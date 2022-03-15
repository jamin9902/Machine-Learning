import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt

# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lam):
        self.eta = eta
        self.lam = lam

    def onehot(self, Y):
        encoded = []
        for label in Y:
            y = np.zeros(3)
            y[label] = 1
            encoded.append(y)
        return np.array(encoded)

    def softmax(self, X, W):
        z = np.dot(X, W.T)
        z_help = logsumexp(z, axis=1)[:, np.newaxis]
        return (z - z_help)

    def calc_loss(self):
        return -np.sum(self.Y * self.softmax(self.X, self.W)) + self.lam * (np.power(self.W, 2).sum())

    def fit(self, X, Y):
        X = np.hstack((np.ones(X.shape[0])[:, np.newaxis], X))
        self.X = X
        self.Y = self.onehot(Y)
        self.W = np.random.randn(3, X.shape[1])
        self.loss = []

        for i in range(20000):
            gradient = np.dot((np.exp(self.softmax(self.X, self.W)) - self.Y).T, self.X) + 2 * self.lam * self.W
            self.loss.append(self.calc_loss())
            self.W -= self.eta * gradient
            
    def predict(self, X_pred):
        X_pred = np.hstack((np.ones(X_pred.shape[0])[:, np.newaxis], X_pred))
        return np.argmax(np.dot(X_pred, self.W.T), axis=1)

    def visualize_loss(self, output_file, show_charts=False):
        plt.plot(self.loss)
        plt.title("Logisitic Regression Loss for λ = " + str(self.lam) + " and η = " + str(self.eta))
        plt.xlabel('Number of Iterations')
        plt.ylabel('Negative Log-Likelihood Loss')
        plt.savefig('loss.png')
        plt.show()
