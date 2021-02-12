import svm.utils as utils
import os
import numpy as np
from scipy.io import loadmat
from os.path import join
import matplotlib.pyplot as plt

def linear_kernel():
    # Load from ex6data1
    # You will have X, y as keys in the dict data
    # linear kernel
    data = loadmat(os.path.join(os.getcwd(), 'ex6data1.mat'))
    X, y = data['X'], data['y'][:, 0]

    # Plot training data
    utils.plotData(X, y)



    # You should try to change the C value below and see how the decision
    # boundary varies (e.g., try C = 1000)
    C = 1

    model = utils.svmTrain(X, y, C, 'linearKernel', 1e-3, 20)
    utils.visualizeBoundaryLinear(X, y, model)



def gaussian_kernel():

    # Gaussian kernel
    # Load from ex6data2
    # You will have X, y as keys in the dict data
    data = loadmat(os.path.join(os.getcwd(), 'ex6data2.mat'))
    X, y = data['X'], data['y'][:, 0]

    # Plot training data
    utils.plotData(X, y)

    # SVM Parameters
    C = 1
    sigma = 0.1

    model= utils.svmTrain(X, y, C, 'gaussianKernel', args=(sigma,))
    utils.visualizeBoundary(X, y, model)


if __name__ == '__main__':
    # gaussian kernel:
    gaussian_kernel()

    # linear kernel:
    linear_kernel()