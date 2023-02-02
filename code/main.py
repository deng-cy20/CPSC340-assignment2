#!/usr/bin/env python
from random_tree import RandomForest, RandomTree
from naive_bayes import NaiveBayes, NaiveBayesLaplace
from knn import KNN
from kmeans import Kmeans
from decision_tree import DecisionTree
from decision_stump import DecisionStumpInfoGain
from utils import load_dataset, plot_classifier, handle, run, main
import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())


@handle("1.2")
def q1_2():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"].astype(bool)
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]
    groupnames = dataset["groupnames"]
    wordlist = dataset["wordlist"]
    """YOUR CODE HERE FOR Q1.2"""
    print(wordlist[72])
    print(len(wordlist))
    print(wordlist)
    for i in range(len(X[802])):
        if X[802][i]:
            print("missing", wordlist[i])
    print(X[802])

    print(groupnames)
    print(y[802])


@handle("1.3")
def q1_3():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    y_hat = model.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Naive Bayes training error: {err_train:.3f}")

    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes validation error: {err_valid:.3f}")


@handle("2.1")
def q2_1():
    dataset = load_dataset("citiesSmall.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]
    """YOUR CODE HERE FOR Q2.1"""
    knn2_1 = KNN(1)
    knn2_1.fit(X, y)
    train_hat = knn2_1.predict(X)
    print(f"Training Error: {1 - np.sum(train_hat == y)/y.size}")
    test_hat = knn2_1.predict(X_test)
    print(f"Test Error: {1 - np.sum(test_hat == y_test)/y_test.size}")

    kn2 = KNeighborsClassifier(1)
    kn2.fit(X, y)
    train_hat = kn2.predict(X)
    print(f"Training Error: {1 - np.sum(train_hat == y)/y.size}")
    test_hat = kn2.predict(X_test)
    print(f"Test Error: {1 - np.sum(test_hat == y_test)/y_test.size}")

    plot_classifier(knn2_1, X, y)
    fname = Path("..", "figs", "q2_trainingboundary.pdf")
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)

    plot_classifier(kn2, X, y)
    fname = Path("..", "figs", "q2_sklearnboundary.pdf")
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)
    y_van = knn2_1.predict(np.array([[49.246292, -123.116226]]))
    print(f"Vancouver Prediction: {y_van}")


@handle("2.2")
def q2_2():
    dataset = load_dataset("ccdebt.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]
    ks = list(range(1, 30, 4))
    """YOUR CODE HERE FOR Q2.2"""
    cv_accs = np.zeros(8)
    test_acc = np.zeros(8)
    train_err = np.zeros(8)
    num = 0
    for k in ks:
        knn2_2 = KNeighborsClassifier(k)
        mean_acc = 0
        for i in range(10):
            maskX = np.ones(X.shape[0], dtype=bool)
            masky = np.ones(y.shape[0], dtype=bool)
            maskX[int(i / 10 * maskX.shape[0]):int((i + 1) / 10 *
                                                   maskX.shape[0])] = 0
            masky[int(i / 10 * masky.shape[0]):int((i + 1) / 10 *
                                                   masky.shape[0])] = 0
            X_train = X[maskX]
            y_train = y[masky]
            X_val = X[~maskX]
            y_val = y[~masky]
            knn2_2.fit(X_train, y_train)
            y_hat = knn2_2.predict(X_val)
            mean_acc = mean_acc + np.sum(y_hat == y_val) / y_val.size
        mean_acc = mean_acc / 10
        cv_accs[num] = mean_acc
        knn2_2.fit(X, y)
        test_hat = knn2_2.predict(X_test)
        test_acc[num] = np.sum(test_hat == y_test) / y_test.size
        knn2_2.fit(X, y)
        train_hat = knn2_2.predict(X)
        train_err[num] = 1 - np.sum(train_hat == y) / y.size
        num = num + 1
    plt.plot(ks, cv_accs)
    plt.plot(ks, test_acc)
    plt.legend(['Cross-Validation', 'Test'])
    plt.xlabel('Choosing of K')
    plt.ylabel('Accuracy')
    fname = Path("..", "figs", "q2_curve.pdf")
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)

    plt.figure()
    plt.plot(ks, train_err)
    plt.legend('Train')
    plt.xlabel('Choosing of K')
    plt.ylabel('Error')
    fname = Path("..", "figs", "q2_traincurve.pdf")
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)


@handle("3")
def q3():
    dataset = load_dataset("vowel.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]
    print(f"n = {X.shape[0]}, d = {X.shape[1]}")

    def evaluate_model(model):
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print(f"    Training error: {tr_error:.3f}")
        print(f"    Testing error: {te_error:.3f}")

    """YOUR CODE FOR Q3"""
    print("Random tree info gain")
    evaluate_model(RandomTree(max_depth=np.inf))

    print("Decision tree info gain")
    evaluate_model(DecisionTree(max_depth=np.inf,
                   stump_class=DecisionStumpInfoGain))

    print("Random forest info gain")
    evaluate_model(RandomForest(num_trees=50, max_depth=np.inf))


@handle("4")
def q4():
    X = load_dataset("clusterData.pkl")["X"]

    model = Kmeans(k=4)
    model.fit(X)
    y = model.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")

    fname = Path("..", "figs", "kmeans_basic_rerun.png")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")


@handle("4.1")
def q4_1():
    X = load_dataset("clusterData.pkl")["X"]
    """YOUR CODE HERE FOR Q4.1"""
    model = Kmeans(k=4)
    best_model = None
    low_err = np.inf
    for i in range(50):
        model.fit(X)
        if (model.err < low_err):
            best_model = model
            low_err = model.err
    print(f"Lowest Error Obtained: {low_err}")
    y = best_model.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")

    fname = Path("..", "figs", "kmeans_lowest_error.png")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")


@handle("4.2")
def q4_2():
    X = load_dataset("clusterData.pkl")["X"]
    """YOUR CODE HERE FOR Q4.2"""
    ks = list(range(1, 11))
    errs = np.zeros(10)
    num = 0
    for k in ks:
        model = Kmeans(k)
        low_err = np.inf
        for i in range(50):
            model.fit(X)
            if (model.err < low_err):
                low_err = model.err
        errs[num] = low_err
        num = num + 1
    plt.plot(ks, errs)
    fname = Path("..", "figs", "kmeans_10_ks.png")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")


@handle("4.3")
def q4_3():
    X = load_dataset("clusterData2.pkl")["X"]
    """YOUR CODE HERE FOR Q4.3"""


if __name__ == "__main__":
    main()
