"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np

import utils
from utils import euclidean_dist_squared


class KNN:
    X = None
    y = None

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X  # just memorize the training data
        self.y = y

    def predict(self, X_hat):
        """YOUR CODE HERE FOR Q2.1"""
        Y_hat = None  # output of the function
        distance = euclidean_dist_squared(self.X, X_hat)
        dis_index = np.argsort(np.array(distance), axis=0)
        y_temp = np.take(self.y, dis_index)
        y_temp = y_temp[:self.k, :]
        y_count = np.apply_along_axis(np.bincount, axis=0, arr=y_temp)
        Y_hat = np.argmax(y_count, axis=0)
        return Y_hat
