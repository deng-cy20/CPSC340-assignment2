from random_stump import RandomStumpInfoGain
from decision_tree import DecisionTree
import numpy as np
import statistics as stats

import utils


class RandomTree(DecisionTree):
    def __init__(self, max_depth):
        DecisionTree.__init__(
            self, max_depth=max_depth, stump_class=RandomStumpInfoGain
        )

    def fit(self, X, y):
        n = X.shape[0]
        boostrap_inds = np.random.choice(n, n, replace=True)
        bootstrap_X = X[boostrap_inds]
        bootstrap_y = y[boostrap_inds]

        DecisionTree.fit(self, bootstrap_X, bootstrap_y)


class RandomForest:
    num_trees = None
    max_depth = None
    trees = None
    y_matrix = None

    def __init__(self, num_trees, max_depth):
        self.num_trees = num_trees
        self.max_depth = max_depth

    def fit(self, X, y):
        self.trees = []
        for i in range(self.num_trees):
            new_tree = RandomTree(self.max_depth)
            new_tree.fit(X, y)
            self.trees.append(new_tree)

    def predict(self, X):
        rows = len(X)
        self.y_matrix = []
        for tree in self.trees:
            curr_tree_y_values = tree.predict(X)
            self.y_matrix.append(curr_tree_y_values)

        y = np.zeros(rows)
        for i in range(rows):
            all_y_values = []
            for y_row in self.y_matrix:
                all_y_values.append(y_row[i])
            y[i] = stats.mode(all_y_values)

        self.y_matrix = None
        return y
