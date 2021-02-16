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

def displayData(X, example_width=None, figsize=(10, 10)):
    """
    Displays 2D data in a nice grid.

    Parameters
    ----------
    X : array_like
        The input data of size (m x n) where m is the number of examples and n is the number of
        features.

    example_width : int, optional
        THe width of each 2-D image in pixels. If not provided, the image is assumed to be square,
        and the width is the floor of the square root of total number of pixels.

    figsize : tuple, optional
        A 2-element tuple indicating the width and height of figure in inches.
    """
    # Compute rows, cols
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        n = X.size
        m = 1
        X = X[None]  # Promote to a 2 dimensional array
    else:
        raise IndexError('Input X should be 1 or 2 dimensional.')

    example_width = example_width or int(np.round(np.sqrt(n)))
    example_height = int(n / example_width)

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    fig, ax_array = plt.subplots(display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(example_height, example_width, order='F'), cmap='gray')
        ax.axis('off')

if __name__ == '__main__':
    #  Load Face dataset
    data = loadmat(os.path.join(os.getcwd(), 'ex7faces.mat'))
    X = data['X']
    #  Display the first 100 faces in the dataset
    displayData(X[:100, :], figsize=(8, 8))
    plt.show()

    #  normalize X by subtracting the mean value from each feature
    X_norm, mu, sigma = featureNormalize(X)

    #  Run PCA
    U, S = pca(X_norm)

    #  Project images to the eigen space using the top k eigenvectors
    #  If you are applying a machine learning algorithm
    K = 100
    Z = projectData(X_norm, U, K)

    #  Project images to the eigen space using the top K eigen vectors and
    #  visualize only using those K dimensions
    #  Compare to the original input, which is also displayed

    X_rec = recoverData(Z, U, K)

    # Display normalized data
    displayData(X_norm[:100, :], figsize=(6, 6))
    plt.gcf().suptitle('Original faces')
    plt.savefig('original_faces.png', dpi = 250, bbox_inches='tight')
    plt.show()
    # Display reconstructed data from only k eigenfaces
    displayData(X_rec[:100, :], figsize=(6, 6))
    plt.gcf().suptitle('Recovered faces')
    plt.savefig('recovered_faces.png', dpi = 250, bbox_inches='tight')
    plt.show()