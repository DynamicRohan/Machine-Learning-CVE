"""
RUN: LogReg_sigmoid_test0.2
Classifier : Logistic Regression (random_state=0, max_iter=1000)
Kernel PCA : sigmoid kernel, 2 components, SUBSAMPLED (sklearn defaults: gamma=1/n_features, coef0=1)
Test size  : 0.2

APPROXIMATE Kernel PCA: fitted on M stratified training rows, then ALL train/test rows are projected.
"""
import os, gc
import numpy as np, pandas as pd

# ---- EDIT THIS: folder containing the three CSV files ----
DATA_DIR = '/mnt/user-data/uploads/'
FILES = ['Tuesday-WorkingHours_pcap_ISCX.csv',
         'Wednesday-workingHours_pcap_ISCX.csv',
         'Thursday-WorkingHours-Morning-WebAttacks_pcap_ISCX.csv']
TEST_SIZE = 0.2
M = 10000            # rows used to fit Kernel PCA (largest that fit a 4 GB machine)

# 1-6. Load, clean, impute (mean), encode  (same as original code)
Xs, ys = [], []
for f in FILES:
    d = pd.read_csv(os.path.join(DATA_DIR, f), low_memory=False)
    d.columns = d.columns.str.strip()
    ys.append(d.iloc[:, -1].astype(str).str.strip().values)
    Xs.append(d.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64))
    del d; gc.collect()
X = np.concatenate(Xs); del Xs; gc.collect()
y = np.concatenate(ys); del ys
print("Dataset shape:", (X.shape[0], X.shape[1] + 1))
vc = pd.Series(y).value_counts()
print("\nOriginal class distribution:\n", vc)
rare = vc[vc < 2].index
if len(rare):
    m_ = ~np.isin(y, rare); X = X[m_]; y = y[m_]
X[~np.isfinite(X)] = np.nan
means = np.nanmean(X, axis=0)                # same as SimpleImputer(strategy='mean')
r, c = np.where(np.isnan(X)); X[r, c] = means[c]; del r, c; gc.collect()

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)
from sklearn.linear_model import LogisticRegression

le = LabelEncoder(); y = le.fit_transform(y)
print("\nEncoded classes:")
for i, n in enumerate(le.classes_): print(i, "=", n)

# 7. Train/test split
tr, te = train_test_split(np.arange(len(y)), test_size=TEST_SIZE, random_state=0, stratify=y)
X_train, X_test, y_train, y_test = X[tr], X[te], y[tr], y[te]
del X; gc.collect()
print("\nTraining samples:", X_train.shape[0]); print("Testing samples :", X_test.shape[0])

# 8. Scaling
sc = StandardScaler(copy=False)
X_train = sc.fit_transform(X_train); X_test = sc.transform(X_test)

# 9. SUBSAMPLED Kernel PCA: fit on M stratified training rows, project everything in chunks
sub, _ = train_test_split(np.arange(len(y_train)), train_size=M, random_state=0, stratify=y_train)
kpca = KernelPCA(n_components=2, kernel='sigmoid', random_state=0)
kpca.fit(X_train[sub])
def project(A, chunk=4000):
    out = np.empty((len(A), 2))
    for i in range(0, len(A), chunk): out[i:i+chunk] = kpca.transform(A[i:i+chunk])
    return out
X_train_kpca = project(X_train); X_test_kpca = project(X_test)
del X_train, X_test; gc.collect()
print("\nShape after Kernel PCA:", X_train_kpca.shape, X_test_kpca.shape)

# 10. Classifier
classifier = LogisticRegression(random_state=0, max_iter=1000)
classifier.fit(X_train_kpca, y_train)
y_pred = classifier.predict(X_test_kpca)

# 11-12. Metrics
labels = range(len(le.classes_))
print("\nConfusion Matrix:\n"); print(confusion_matrix(y_test, y_pred, labels=labels))
print("\nAccuracy  : {:.4f}".format(accuracy_score(y_test, y_pred)))
print("Precision : {:.4f}".format(precision_score(y_test, y_pred, average='weighted', zero_division=0)))
print("Recall    : {:.4f}".format(recall_score(y_test, y_pred, average='weighted', zero_division=0)))
print("F1 Score  : {:.4f}".format(f1_score(y_test, y_pred, average='weighted', zero_division=0)))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, labels=labels, target_names=le.classes_, zero_division=0))
