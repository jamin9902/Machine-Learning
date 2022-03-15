#####################
# CS 181, Spring 2022
# Homework 1, Problem 1
# STARTER CODE
##################

import numpy as np
import math
import matplotlib.pyplot as plt

data = [(0., 0.),
        (1., 0.5),
        (2., 1.),
        (3., 2.),
        (4., 1.),
        (6., 1.5),
        (8., 0.5)]

def compute_loss(tau):
    loss = 0.0
    for i in data:
        residual = i[1]
        for j in data:
            if i != j:
                c = (i[0] - j[0]) ** 2.0
                residual -= math.exp(-(c / tau)) * j[1]
        loss += residual ** 2.0
    return loss

for tau in (0.01, 2, 100):
    print("Loss for tau = " + str(tau) + ": " + str(compute_loss(tau)))


x_train = np.array([d[0] for d in data])
y_train = np.array([d[1] for d in data])

x_test = np.arange(0, 12, .1)

def predict_kernel_based(tau):
    """Returns predictions for the values in x_test, using kernel-based predictor with the specified tau."""
    predictions = []

    for i in x_test:
        sum = 0.0    
        for j in data:
            c = (i - j[0]) ** 2.0
            sum += math.exp(-(c / tau)) * j[1]
        predictions.append(sum)

    return np.array(predictions)


def plot_kernel_based_preds(tau):
    plt.xlim([0, 12])
    plt.ylim([0, 7])
    
    y_test = predict_kernel_based(tau=tau)
    
    plt.scatter(x_train, y_train, label = "training data", color = 'black')
    plt.plot(x_test, y_test, label = "predictions using tau = " + str(tau))

    plt.grid()

    plt.legend()
    plt.title("Kernel-Based Predictions with tau = " + str(tau))
    plt.savefig('tau' + str(tau) + '.png')
    plt.show()

for tau in (0.01, 2, 100):
    plot_kernel_based_preds(tau)