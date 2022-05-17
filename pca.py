import numpy as np

def pca(X, k):
    # Data matrix X, assumes 0-centered
    n, m = X.shape
    assert np.allclose(X.mean(axis=0), np.zeros(m))
    # Compute covariance matrix
    C = np.dot(X.T, X) / (n-1)
    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    eigen_vecs = eigen_vecs[:, :k]
    # Project X onto PC space
    X_pca = np.dot(X, eigen_vecs)
    return X_pca

X = np.array([[-1, -1, 0, 2, 0], [-2, 0, 0, 1, 1]])
X = X.T
k = 1
X_pca = pca(X, k)
print(X_pca)

