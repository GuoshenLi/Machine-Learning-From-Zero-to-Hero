import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def pca(X):
    """
    Run principal component analysis.

    Parameters
    ----------
    X : array_like
        The dataset to be used for computing PCA. It has dimensions (m x n)
        where m is the number of examples (observations) and n is
        the number of features.

    Returns
    -------
    U : array_like
        The eigenvectors, representing the computed principal components
        of X. U has dimensions (n x n) where each column is a single
        principal component.

    S : array_like
        A vector of size n, contaning the singular values for each
        principal component. Note this is the diagonal of the matrix we
        mentioned in class.

    Instructions
    ------------
    You should first compute the covariance matrix. Then, you
    should use the "svd" function to compute the eigenvectors
    and eigenvalues of the covariance matrix.

    Notes
    -----
    When computing the covariance matrix, remember to divide by m (the
    number of examples).
    """
    # Useful values
    m, n = X.shape

    # You need to return the following variables correctly.
    U = np.zeros(n)
    S = np.zeros(n)

    # ====================== YOUR CODE HERE ======================
    cov = (1 / m) * X.T @ X
    U, S, V = np.linalg.svd(cov)
    # ============================================================
    return U, S


def featureNormalize(X):
    """
    Normalizes the features in X returns a normalized version of X where the mean value of each
    feature is 0 and the standard deviation is 1. This is often a good preprocessing step to do when
    working with learning algorithms.

    Parameters
    ----------
    X : array_like
        An dataset which is a (m x n) matrix, where m is the number of examples,
        and n is the number of dimensions for each example.

    Returns
    -------
    X_norm : array_like
        The normalized input dataset.

    mu : array_like
        A vector of size n corresponding to the mean for each dimension across all examples.

    sigma : array_like
        A vector of size n corresponding to the standard deviations for each dimension across
        all examples.
    """
    mu = np.mean(X, axis=0)
    X_norm = X - mu

    sigma = np.std(X_norm, axis=0, ddof=1)
    X_norm /= sigma
    return X_norm, mu, sigma


def projectData(X, U, K):
    """
    Computes the reduced data representation when projecting only
    on to the top K eigenvectors.

    Parameters
    ----------
    X : array_like
        The input dataset of shape (m x n). The dataset is assumed to be
        normalized.

    U : array_like
        The computed eigenvectors using PCA. This is a matrix of
        shape (n x n). Each column in the matrix represents a single
        eigenvector (or a single principal component).

    K : int
        Number of dimensions to project onto. Must be smaller than n.

    Returns
    -------
    Z : array_like
        The projects of the dataset onto the top K eigenvectors.
        This will be a matrix of shape (m x k).

    Instructions
    ------------
    Compute the projection of the data using only the top K
    eigenvectors in U (first K columns).
    For the i-th example X[i,:], the projection on to the k-th
    eigenvector is given as follows:

        x = X[i, :]
        projection_k = np.dot(x,  U[:, k])

    """
    # You need to return the following variables correctly.
    Z = np.zeros((X.shape[0], K))
    U_reduced = U[:, :K]
    # ====================== YOUR CODE HERE ======================
    Z = X @ U_reduced
    # =============================================================
    return Z


def recoverData(Z, U, K):
    """
    Recovers an approximation of the original data when using the
    projected data.

    Parameters
    ----------
    Z : array_like
        The reduced data after applying PCA. This is a matrix
        of shape (m x K).

    U : array_like
        The eigenvectors (principal components) computed by PCA.
        This is a matrix of shape (n x n) where each column represents
        a single eigenvector.

    K : int
        The number of principal components retained
        (should be less than n).

    Returns
    -------
    X_rec : array_like
        The recovered data after transformation back to the original
        dataset space. This is a matrix of shape (m x n), where m is
        the number of examples and n is the dimensions (number of
        features) of original datatset.

    Instructions
    ------------
    Compute the approximation of the data by projecting back
    onto the original space using the top K eigenvectors in U.
    For the i-th example Z[i,:], the (approximate)
    recovered data for dimension j is given as follows:

        v = Z[i, :]
        recovered_j = np.dot(v, U[j, :K])

    Notice that U[j, :K] is a vector of size K.
    """
    # You need to return the following variables correctly.
    X_rec = np.zeros((Z.shape[0], U.shape[0]))
    U_reduced = U[:,:K]
    # ====================== YOUR CODE HERE ======================
    X_rec = Z @ U_reduced.T
    # =============================================================
    return X_rec

if __name__ == '__main__':
    # Load the dataset into the variable X
    data = loadmat(os.path.join(os.getcwd(), 'ex7data1.mat'))
    X = data['X']

    #  Visualize the example dataset
    plt.plot(X[:, 0], X[:, 1], 'bo')
    plt.axis([0.5, 6.5, 2, 8])
    plt.gca().set_aspect('equal')
    plt.grid(False)

    #  Before running PCA, it is important to first normalize X
    X_norm, mu, sigma = featureNormalize(X)

    #  Run PCA
    U, S = pca(X_norm)



    #  Project the data onto K = 1 dimension
    K = 1
    Z = projectData(X_norm, U, K)
    print('Projection of the first example: {:.6f}'.format(Z[0, 0]))
    print('(this value should be about    : 1.481274)')

    X_rec  = recoverData(Z, U, K)
    print('Approximation of the first example: [{:.6f} {:.6f}]'.format(X_rec[0, 0], X_rec[0, 1]))
    print('       (this value should be about  [-1.047419 -1.047419])')

    #  Plot the normalized dataset (returned from featureNormalize)
    with sns.axes_style('darkgrid'):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(X_norm[:, 0], X_norm[:, 1], 'bo')
        ax.set_aspect('equal')
        ax.grid(True)
        plt.axis([-3, 2.75, -3, 2.75])

        # Draw lines connecting the projected points to the original points
        ax.plot(X_rec[:, 0], X_rec[:, 1], 'ro', mec='r', mew=2, mfc='none')
        for xnorm, xrec in zip(X_norm, X_rec):
            ax.plot([xnorm[0], xrec[0]], [xnorm[1], xrec[1]], '--k', lw=1)
        plt.savefig('2d_1d_line.png', dpi = 250, bbox_inches = 'tight')
        plt.show()
