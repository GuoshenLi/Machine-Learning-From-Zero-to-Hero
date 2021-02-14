import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces
import matplotlib.pyplot as plt
import seaborn as sns

def plotData(x, y):
    """
    Plots the data points x and y into a new figure. Plots the data
    points and gives the figure axes labels of population and profit.

    Parameters
    ----------
    x : array_like
        Data point values for x-axis.

    y : array_like
        Data point values for y-axis. Note x and y should have the same size.
    """
    with sns.axes_style("darkgrid"):
        fig = plt.figure(figsize = (6, 3))
        ax = fig.add_subplot()
        ax.plot(x, y, 'o', label = 'Training data')
        ax.legend(loc = 'best')
        return ax


def computeCost(X, y, theta):
    """
    Compute cost for linear regression. Computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y.

    Parameters
    ----------
    X : array_like
        The input dataset of shape (m x n+1), where m is the number of examples,
        and n is the number of features. We assume a vector of one's already
        appended to the features so we have n+1 columns.

    y : array_like
        The values of the function at each data point. This is a vector of
        shape (m, ).

    theta : array_like
        The parameters for the regression function. This is a vector of
        shape (n+1, ).

    Returns
    -------
    J : float
        The value of the regression cost function.

    Instructions
    ------------
    Compute the cost of a particular choice of theta.
    You should set J to the cost.
    """

    # initialize some useful values
    y = y[:,np.newaxis]
    m = y.shape[0]  # number of training examples
    theta = theta[:, np.newaxis]

    # You need to return the following variables correctly
    J = (1 / (2 * m)) * (X @ theta - y).T @ (X @ theta - y)

    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn `theta`. Updates theta by taking `num_iters`
    gradient steps with learning rate `alpha`.

    Parameters
    ----------
    X : array_like
        The input dataset of shape (m x n+1).

    y : array_like
        Value at given features. A vector of shape (m, ).

    theta : array_like
        Initial values for the linear regression parameters.
        A vector of shape (n+1, ).

    alpha : float
        The learning rate.

    num_iters : int
        The number of iterations for gradient descent.

    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).

    J_history : list
        A python list for the values of the cost function after each iteration.

    Instructions
    ------------
    Peform a single gradient step on the parameter vector theta.

    While debugging, it can be useful to print out the values of
    the cost function (computeCost) and gradient here.
    """
    # Initialize some useful values
    y = y[:,np.newaxis]
    m = y.shape[0]  # number of training examples

    # make a copy of theta, to avoid changing the original array, since numpy arrays
    # are passed by reference to functions
    theta = theta.copy()
    theta = theta[:, np.newaxis]
    J_history = []  # Use a python list to save cost in every iteration

    for i in range(num_iters):
        # ==================== YOUR CODE HERE =================================
        theta = theta - (1 / m) * alpha * (X.T @ (X @ theta - y))
        # =====================================================================
        # save the cost J in every iteration
        J_history.append(computeCost(X, y.ravel(), theta.ravel()))

    return theta, J_history


if __name__ == '__main__':
    # Read comma separated data
    data = np.loadtxt(os.path.join(os.getcwd(), 'ex1data1.txt'), delimiter=',')
    X, y = data[:, 0], data[:, 1]

    m = y.size  # number of training examples

    ax = plotData(X, y)
    plt.savefig('data1.png', dpi=250, bbox_inches='tight')
    plt.show()

    X = np.concatenate((np.ones((m, 1)), X[:, np.newaxis]), axis=1)


    # initialize fitting parameters
    theta = np.zeros(2)

    # some gradient descent settings
    iterations = 1500
    alpha = 0.01

    theta, J_history = gradientDescent(X ,y, theta, alpha, iterations)
    print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(theta.ravel()[0], theta.ravel()[1]))

    # plot the linear fit
    ax = plotData(X[:, 1], y)
    ax.plot(X[:, 1], X @ theta, '-')
    ax.legend(['Training data', 'Linear regression'], loc = 'best')
    plt.savefig('linear_reg1.png', dpi=250, bbox_inches='tight')
    plt.show()

