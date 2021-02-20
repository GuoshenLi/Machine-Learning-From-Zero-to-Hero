import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import seaborn as sns
class GMM():

    def __init__(self, k=3, max_iter=100):

        self._k = k
        self._max_iter = max_iter
        self._gmm_params = []

    def _compute_likelihood(self, X):
        alpha = self._alpha
        mean_all = []
        cov_all = []
        for i in range(3):
            mean = self._gmm_params[i]['mean']
            cov = self._gmm_params[i]['cov']
            mean_all.append(mean)
            cov_all.append(cov)

        loglikelihood = np.sum(np.log(alpha[0] * multivariate_normal(mean=mean_all[0], cov=cov_all[0]).pdf(X) + \
                     alpha[1] * multivariate_normal(mean=mean_all[1], cov=cov_all[1]).pdf(X) + \
                     alpha[2] * multivariate_normal(mean=mean_all[2], cov=cov_all[2]).pdf(X)))

        return loglikelihood

    def fit(self, X):
        self._init_params(X)
        self.ll = []
        for i in range(self._max_iter):
            self._e_step(X)
            self._m_step(X)
            loglikelihood = self._compute_likelihood(X)

            self.ll.append(loglikelihood)
    def _e_step(self, X):

        n_sample = X.shape[0]
        self.wij = np.zeros((n_sample, self._k))
        for i in range(self._k):
            self.wij[:, i] = self._alpha[i] * self._gaussian_pdf(X, self._gmm_params[i])
        self.wij = self.wij / np.sum(self.wij, axis = 1).reshape(-1, 1)

    def _m_step(self, X):
        n_sample = X.shape[0]
        for j in range(self._k):

            self._alpha[j] = np.sum(self.wij[:, j]) / n_sample
            mean = (self.wij[:, j] @ X) / np.sum(self.wij[:, j])
            covar = (self.wij[:, j] * (X - mean).T).dot(X - mean) / np.sum(self.wij[:, j])
            self._gmm_params[j]["mean"] = mean
            self._gmm_params[j]["cov"] = covar


    def _init_params(self, X):

        n_sample = X.shape[0]

        self._alpha = np.ones(self._k) / self._k
        for _ in range(self._k):
            params = dict()
            params["mean"] = X[int(np.random.choice(n_sample, 1))]
            params["cov"] = np.cov(X.T)
            self._gmm_params.append(params)

    def _gaussian_pdf(self, X, params):

        n_sample, n_feature = X.shape
        mean = params["mean"]
        covar = params["cov"]

        determinant = np.linalg.det(covar)
        Gauss = np.zeros(n_sample)
        coeff = 1 / np.sqrt(np.power(2*np.pi, n_feature) * determinant)
        for i, x in enumerate(X):
            exponent = np.exp(-0.5 * (x - mean).T.dot(np.linalg.inv(covar)).dot(x - mean))
            Gauss[i] = coeff * exponent
        return Gauss


if __name__ == "__main__":
    x1 = np.random.multivariate_normal(mean = [-3, -3], cov = [[1, 0.5],[0.5, 1]], size = 100)
    x2 = np.random.multivariate_normal(mean = [3, 0], cov = [[1, 0],[0, 1]], size = 100)
    x3 = np.random.multivariate_normal(mean = [0, 4], cov = [[1, -0.2],[-0.2, 1]], size = 100)
    with sns.axes_style('darkgrid'):
        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot()
        ax.scatter(x = x1[:,0], y = x1[:,1])
        ax.scatter(x = x2[:,0], y = x2[:,1])
        ax.scatter(x = x3[:,0], y = x3[:,1])
        ax.legend(['class1', 'class2', 'class3'])
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.grid('True')

    X = np.concatenate([x1, x2, x3], axis = 0)

    model = GMM(k = 3, max_iter = 100)
    model.fit(X)

    alpha = model._alpha
    mean_all = []
    cov_all = []
    for i in range(3):
        mean = model._gmm_params[i]['mean']
        cov = model._gmm_params[i]['cov']
        mean_all.append(mean)
        cov_all.append(cov)

    x1plot = np.linspace(min(X[:, 0]), max(X[:, 0]), 1000)
    x2plot = np.linspace(min(X[:, 1]), max(X[:, 1]), 1000)
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros(X1.shape)

    for i in range(X1.shape[1]):
        this_X = np.stack((X1[:, i], X2[:, i]), axis=1)

        vals[:, i] = alpha[0] * multivariate_normal(mean=mean_all[0], cov=cov_all[0]).pdf(this_X)+ \
             alpha[1] * multivariate_normal(mean=mean_all[1], cov=cov_all[1]).pdf(this_X)+ \
             alpha[2] * multivariate_normal(mean=mean_all[2], cov=cov_all[2]).pdf(this_X)

    ax.contour(X1, X2, vals)
    plt.savefig('GMM_Contour.png', dpi=250, bbox_inches='tight')
    plt.show()

    print('predicted mean', mean_all)
    print('predicted conv', cov_all)
    print('predicted alpha', alpha)


    with sns.axes_style('darkgrid'):
        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot()
        ax.plot(model.ll)
        ax.set_xlabel('iteration')
        ax.set_ylabel('loglikelihood')
        ax.grid('True')
        plt.savefig('GMM_likelihood.png', dpi=250, bbox_inches='tight')
        plt.show()