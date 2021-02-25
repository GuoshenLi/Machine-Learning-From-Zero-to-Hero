import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def relu(z):
    mask = (z > 0).astype(np.float32)
    z *= mask
    return z

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def initialize_parameters_he(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """

    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1  # integer representing the number of layers

    for l in range(1, L + 1):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(
            2 / layers_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        ### END CODE HERE ###

    return parameters


def forward_propagation(X, parameters):

    """
    Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.

    Arguments:
    X -- input dataset, of shape (2, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (20, 2)
                    b1 -- bias vector of shape (20, 1)
                    W2 -- weight matrix of shape (3, 20)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)

    Returns:
    A3 -- last activation value, output of the forward propagation, of shape (1,1)
    cache -- tuple, information stored for computing the backward propagation
    """

    np.random.seed(1)

    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache


def compute_cost(a3, Y):
    m = Y.shape[1]
    loss = - (1 / m) * np.sum(Y * np.log(a3) + (1 - Y) * np.log(1 - a3 + 1e-10))

    return loss


def update_parameters(parameters, grads, learning_rate):

    parameters["W1"] -= learning_rate * grads['dW1']
    parameters["b1"] -= learning_rate * grads['db1']
    parameters["W2"] -= learning_rate * grads['dW2']
    parameters["b2"] -= learning_rate * grads['db2']
    parameters["W3"] -= learning_rate * grads['dW3']
    parameters["b3"] -= learning_rate * grads['db3']

    return parameters

def backward_propagation(X, Y, cache):
    """
    Implements the backward propagation of our baseline model to which we added an L2 regularization.

    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation()
    lambd -- regularization hyperparameter, scalar

    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """

    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y

    ### START CODE HERE ### (approx. 1 line)
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    ### END CODE HERE ###
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    ### START CODE HERE ### (approx. 1 line)
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    ### END CODE HERE ###
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    ### START CODE HERE ### (approx. 1 line)
    dW1 = 1. / m * np.dot(dZ1, X.T)
    ### END CODE HERE ###
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients


def compute_acc(Y, Y_):
    return np.mean((Y > 0.5).astype(np.float32) == Y_)

def model(X, Y, learning_rate=0.3, num_iterations=3000, print_cost=True):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
    learning_rate -- learning rate of the optimization
    num_iterations -- number of iterations of the optimization loop
    print_cost -- If True, print the cost every 10000 iterations
    lambd -- regularization hyperparameter, scalar
    keep_prob - probability of keeping a neuron active during drop-out, scalar.

    Returns:
    parameters -- parameters learned by the model. They can then be used to predict.
    """

    costs = []  # to keep track of the cost
    accs = []
    layers_dims = [X.shape[0], 20, 3, 1]

    # Initialize parameters dictionary.
    parameters = initialize_parameters_he(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        a3, cache = forward_propagation(X, parameters)

        # Cost function
        cost = compute_cost(a3, Y)
        acc = compute_acc(a3, Y) * 100
        # Backward propagation.
        grads = backward_propagation(X, Y, cache)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the loss every 10000 iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            print("Training acc after iteration {}: {}%".format(i, acc))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            accs.append(acc)

    # plot the cost
    with sns.axes_style('darkgrid'):
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (x100)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.savefig('cost function.png', dpi=250, bbox_inches='tight')
        plt.show()

    with sns.axes_style('darkgrid'):
        plt.plot(accs)
        plt.ylabel('train_acc')
        plt.xlabel('iterations (x100)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.savefig('train_acc.png', dpi=250, bbox_inches='tight')
        plt.show()

    return parameters


def generate_data():
    x = (np.random.rand(1000) - 0.5) * 10
    y = (np.random.rand(1000) - 0.5) * 10

    data = np.vstack((x, y)).T # 1000 * 2
    label = []
    for (x, y) in data:
        if x ** 2 + y ** 2 < 8:
            label.append(0)
        else:
            label.append(1)
    label = np.array(label)

    return data, label



def visualizeBoundary(X, y, parameters):
    """
    Plots a non-linear decision boundary learned by the SVM and overlays the data on it.

    Parameters
    ----------
    X : array_like
        (m x 2) The training data with two features (to plot in a 2-D plane).

    y : array_like
        (m, ) The data labels.

    model : dict
        Dictionary of model variables learned by SVM.
    """
    ax = plotData(X, y)

    # make classification predictions over a grid of values
    x1plot = np.linspace(min(X[:, 0]), max(X[:, 0]), 1000)
    x2plot = np.linspace(min(X[:, 1]), max(X[:, 1]), 1000)
    X1, X2 = np.meshgrid(x1plot, x2plot)

    vals = np.zeros(X1.shape)
    for i in range(X1.shape[1]):
        this_X = np.stack((X1[:, i], X2[:, i]), axis=1)
        vals[:, i] = forward_propagation(this_X.T, parameters)[0].flatten()

    vals = (vals > 0.5).astype(np.float32)
    ax.contour(X1, X2, vals)
    plt.savefig('Neural Network Decision Boundary.png', dpi=250, bbox_inches='tight')
    plt.show()



def plotData(X, Y):
    pos = Y == 1
    neg = Y == 0
    # Plot Examples
    with sns.axes_style("darkgrid"):  # with里面的用一种背景风格
        fig = plt.figure(figsize = (6, 3))
        ax = fig.add_subplot()
        ax.plot(X[pos, 0], X[pos, 1], 'o', markersize = 5, label = 'pos')
        ax.plot(X[neg, 0], X[neg, 1], 'o', markersize = 5, label = 'neg')
        ax.grid(True)
        ax.legend(loc = 'best')
        # plt.show()

    return ax

if __name__ == '__main__':
    X, Y = generate_data()

    X = X.T
    Y = Y.reshape(1, -1)
    parameters = model(X, Y, learning_rate=0.3, num_iterations=6000, print_cost=True)


    X = X.T
    Y = Y.flatten()
    visualizeBoundary(X, Y, parameters)
