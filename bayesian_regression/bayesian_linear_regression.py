import matplotlib.pyplot as plt
import numpy as np

###########bayesian_linear_regression#######
##t his can only fit linear data#####


if __name__ == '__main__':
    m = 10
    x = np.linspace(- 2, 2, 10).reshape(10, 1)

    x_intercept = np.ones((m,1))

    y = 2 * x + 1

    Y = y + np.random.normal(0,1, np.shape(x))

    X = np.concatenate((x, x_intercept), axis = 1)

    # plt.scatter(x,y)
    # plt.show()
    x_test = np.linspace(-5 , 5, 100).reshape(100,1)
    x_test_intercept = np.ones((100,1))
    X_test = np.concatenate((x_test, x_test_intercept), axis = 1)

    A = (X.T) @ X + np.identity(2)

    mean = X_test @ np.linalg.inv(A) @ (X.T) @ Y

    cov = X_test @ np.linalg.inv(A) @ (X_test.T) + np.identity(100)

    variance = np.diagonal(cov).reshape(-1, 1)

    stand_deviation = np.sqrt(variance)

    upper_bound = mean + 2 * stand_deviation
    lower_bound = mean - 2 * stand_deviation

    plt.plot(x_test, mean,c = 'r')
    plt.plot(x_test, upper_bound, c = 'r')
    plt.plot(x_test, lower_bound, c = 'r')

    plt.fill_between(x_test.reshape(-1), upper_bound.reshape(-1), lower_bound.reshape(-1), facecolor = 'r', alpha = .25)
    plt.scatter(x, Y)
    ### 1-d array###
    plt.savefig('bayesian_reg.png', dpi=250, bbox_inches='tight')
    plt.show()
