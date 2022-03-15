#####################
# CS 181, Spring 2022
# Homework 1, Problem 2
# Start Code
##################

import math
from sre_constants import NOT_LITERAL
import matplotlib.cm as cm

from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as c

# set up data
data = [(0., 0.),
        (1., 0.5),
        (2., 1),
        (3., 2),
        (4., 1),
        (6., 1.5),
        (8., 0.5)]

x_train = np.array([d[0] for d in data])
y_train = np.array([d[1] for d in data])

x_test = np.arange(0, 12, .1)

print("y is:")
print(y_train)

def predict_knn(k=1, tau=1):
    """Returns predictions for the values in x_test, using KNN predictor with the specified k."""
    predictions = []

    for i in x_test:    
        k_nearest = []
        for j in range(k):
            nearest = None
            max_kernel = None
            for n in data:
                if n not in k_nearest:
                    c = (i - n[0]) ** 2.0
                    kernel = math.exp(-c / tau)
                    if nearest is None or kernel >= max_kernel:
                        nearest = n
                        max_kernel = kernel
            k_nearest.append(nearest)
        
        sum = 0.0
        for j in k_nearest:
            sum += j[1] / float(k)
        predictions.append(sum)

    return np.array(predictions)


def plot_knn_preds(k):
    plt.xlim([0, 12])
    plt.ylim([0,3])
    
    y_test = predict_knn(k=k)
    
    plt.scatter(x_train, y_train, label = "training data", color = 'black')
    plt.plot(x_test, y_test, label = "predictions using k = " + str(k))

    plt.grid()

    plt.legend()
    plt.title("KNN Predictions with k = " + str(k))
    plt.savefig('k' + str(k) + '.png')
    plt.show()

for k in (1, 3, len(x_train)-1):
    plot_knn_preds(k)