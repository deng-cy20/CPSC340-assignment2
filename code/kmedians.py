import numpy as np
from utils import euclidean_dist_squared


class Kmedians:
    means = None

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        n, d = X.shape
        y = np.ones(n)

        means = np.zeros((self.k, d))
        for kk in range(self.k):
            i = np.random.randint(n)
            means[kk] = X[i]

        while True:
            # iterations of k-medians
            y_old = y

            # Compute L1 norm to each mean
            n, d = X.shape
            T, d = means.shape
            distance_matrix = np.zeros([n, T])
            for i in range(n):
                for j in range(T):
                    distance_matrix[i, j] = abs(X[i, 0] -
                                                means[j, 0]) + abs(X[i, 1] -
                                                                   means[j, 1])
            distance_matrix[np.isnan(distance_matrix)] = np.inf
            y = np.argmin(distance_matrix, axis=1)

            # Update means
            for kk in range(self.k):
                if np.any(
                        y == kk
                ):  # don't update the mean if no examples are assigned to it (one of several possible approaches)
                    means[kk] = np.median(X[y == kk], axis=0)

            changes = np.sum(y != y_old)
            # print('Running K-means, changes in cluster assignment = {}'.format(changes))

            # Stop if no point changed cluster
            if changes == 0:
                break

            # print(self.error(X, y, means))
        self.err, self.errL1 = self.error(X, y, means)
        self.means = means

    def predict(self, X_hat):
        means = self.means
        n, d = X_hat.shape
        T, d = means.shape
        distance_matrix = np.zeros([n, T])
        for i in range(n):
            for j in range(T):
                distance_matrix[i, j] = abs(X_hat[i, 0] -
                                            means[j, 0]) + abs(X_hat[i, 1] -
                                                               means[j, 1])
        distance_matrix[np.isnan(distance_matrix)] = np.inf
        return np.argmin(distance_matrix, axis=1)

    def error(self, X, y, means):
        """YOUR CODE HERE FOR Q4.1"""
        n, d = X.shape
        err = 0
        errL1 = 0
        for i in range(n):
            err = err + (X[i, 0] - means[y[i], 0])**2 + (X[i, 1] -
                                                         means[y[i], 1])**2
            errL1 = errL1 + abs(X[i, 0] - means[y[i], 0]) + abs(X[i, 1] -
                                                                means[y[i], 1])

        return err, errL1
