from urllib.request import ProxyBasicAuthHandler
import numpy as np
from scipy.stats import multivariate_normal as mvn  # you may find this useful


# Please implement the fit(), predict(), and negative_log_likelihood() methods
# of this class. You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class GaussianGenerativeModel:
    def __init__(self, is_shared_covariance=False):
        self.is_shared_covariance = is_shared_covariance

    def mean_cov(self):
        self.means = []
        self.cov = []
        self.shared_cov = np.zeros((self.X.shape[1], self.X.shape[1]))
        for i in range(3):
            rows = self.X[self.Y == i]
            self.means.append(np.mean(rows, axis=0))
            if self.is_shared_covariance:
                self.shared_cov += np.cov(rows.T) * rows.shape[0]
            else:
                self.cov.append(np.cov(rows.T))
        self.shared_cov /= self.X.shape[0]

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.mean_cov()
        self.counts = np.zeros(3)
        for y in self.Y:
            self.counts[y] += 1
        self.bias = np.log(self.counts / (self.counts.sum()))

    def gaussian(self, x):
        probs = np.zeros(3)
        for i in range(3):
            if self.is_shared_covariance:
                cov = self.shared_cov 
            else:
                cov = self.cov[i]
            probs[i] = mvn.pdf(x, mean=self.means[i], cov=cov)
        return np.log(probs)

    def predict(self, X_pred):
        probs = np.zeros((X_pred.shape[0], 3))
        for i in range(X_pred.shape[0]):
            probs[i] = self.gaussian(X_pred[i])
        return np.argmax(probs + self.bias, axis=1)

    def negative_log_likelihood(self, X, y):
        loss = 0.0
        for i in range(3):
            rows = self.X[self.Y == i]
            for j in range(rows.shape[0]):
                if self.is_shared_covariance:
                    cov = self.shared_cov 
                else:
                    cov = self.cov[i]
                loss += np.log(mvn.pdf(rows[j], mean=self.means[i], cov=cov))
                loss += np.log(self.counts[i] / (self.counts.sum()))
        return -loss
