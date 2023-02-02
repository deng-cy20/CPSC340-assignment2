import numpy as np


class NaiveBayes:
    """
    Naive Bayes implementation.
    Assumes the feature are binary.
    Also assumes the labels go from 0,1,...k-1
    """

    p_y = None
    p_xy = None

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def fit(self, X, y):
        n, d = X.shape

        # Compute the number of class labels
        k = self.num_classes

        # Compute the probability of each class i.e p(y==c), aka "baseline -ness"
        counts = np.bincount(y)
        p_y = counts / n

        # Compute the conditional probabilities i.e.
        # p(x_ij=1 | y_i==c) as p_xy[j, c]
        # p(x_ij=0 | y_i==c) as 1 - p_xy[j, c]
        p_xy = np.zeros((d, k))
        # p_xy[j, c] = p(includes word j ^ newsgroup c|newsgroup c)

        # iterate over X_dataset (each row is one piece of data, each column is one word)
        rows, cols = X.shape
        for i in range(rows):  # for each row i,
            newsgroup = y[i]  # find corresponding newsgroup (y[i])
            for j in range(cols):  # for each bool value indexed j (in that row)
                if X[i][j] == 1:
                    p_xy[j, newsgroup] += 1

        # iterate over p_xy
        rows, cols = p_xy.shape
        for j in range(rows):  # each row gives word info for j-th word
            for c in range(cols):  # for each newsgroup indexed c,
                p_xy[j, c] = p_xy[j, c] / counts[c]

        self.p_y = p_y
        self.p_xy = p_xy

    def predict(self, X):
        n, d = X.shape
        k = self.num_classes
        p_xy = self.p_xy
        p_y = self.p_y

        y_pred = np.zeros(n)
        for i in range(n):

            probs = p_y.copy()  # initialize with the p(y) terms
            for j in range(d):
                if X[i, j] != 0:
                    probs *= p_xy[j, :]
                else:
                    probs *= 1 - p_xy[j, :]

            y_pred[i] = np.argmax(probs)

        return y_pred
