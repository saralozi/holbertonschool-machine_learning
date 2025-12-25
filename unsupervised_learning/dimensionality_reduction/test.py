#!/usr/bin/env python3
import numpy as np
pca = __import__('0-pca').pca

X = np.loadtxt("mnist2500_X.txt")
m = X.shape[0]

X_m = X - np.mean(X, axis=0)

W = pca(X_m, var=0.95)
T = np.matmul(X_m, W)

print(T)

X_t = np.matmul(T, W.T)
print(np.sum(np.square(X_m - X_t)) / m)
