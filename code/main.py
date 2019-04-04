# standard Python imports
import os
import argparse
import time
import pickle

# 3rd party libraries
import numpy as np                              # this comes with Anaconda
import matplotlib.pyplot as plt                 # this comes with Anaconda
import pandas as pd                             # this comes with Anaconda
from sklearn.tree import DecisionTreeClassifier # see http://scikit-learn.org/stable/install.html
from sklearn.neighbors import KNeighborsClassifier # same as above
from scipy import stats

# CPSC 340 code
import utils
from decision_stump import DecisionStump, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from knn import KNN, CNN

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True,
        choices=["1.1", "2", "2.2", "2.3", "2.4", "3", "3.1", "3.2", "4.1", "4.2", "5"])

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1.1":
        # Load the fluTrends dataset
        df = pd.read_csv(os.path.join('..', 'data', 'fluTrends.csv'))
        X = df.values
        names = df.columns.values
        print(X.min())
        print(X.max())
        print(X.mean())
        print(np.median(X))
        print(stats.mode(X, axis=None))


        print(np.percentile(X, 5))
        print(np.percentile(X, 25))
        print(np.percentile(X, 50))
        print(np.percentile(X, 75))
        print(np.percentile(X, 95))

        print(np.mean(X, axis=0))
        print(np.var(X, axis = 0))


    elif question == "2":

        # 1. Load citiesSmall dataset
        dataset = load_dataset("citiesSmall.pkl")
        X = dataset["X"]
        y = dataset["y"]

        # 2. Evaluate majority predictor model
        y_pred = np.zeros(y.size) + utils.mode(y)

        error = np.mean(y_pred != y)
        print("Mode predictor error: %.3f" % error)

        # 3. Evaluate decision stump
        model = DecisionStumpEquality()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y)
        print("Decision Stump with inequality rule error: %.3f" % error)

        # PLOT RESULT
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q2_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    elif question == "2.2":
        # 1. Load citiesSmall dataset
        dataset = load_dataset("citiesSmall.pkl")
        X = dataset["X"]
        y = dataset["y"]

        # 3. Evaluate decision stump
        model = DecisionStump()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y)
        print("Decision Stump with threshold rule error: %.3f" % error)

        # PLOT RESULT
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q2_2_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == "2.3":
        # 1. Load citiesSmall dataset
        dataset = load_dataset("citiesSmall.pkl")
        X = dataset["X"]
        y = dataset["y"]

        # 2. Evaluate decision tree
        model = DecisionTree(max_depth=2)
        model.fit(X, y)

        y_pred = model.predict(X)
        error = np.mean(y_pred != y)

        print("Error: %.3f" % error)

    elif question == "2.4":
        dataset = load_dataset("citiesSmall.pkl")
        X = dataset["X"]
        y = dataset["y"]
        print("n = %d" % X.shape[0])

        depths = np.arange(1,15) # depths to try

        t = time.time()
        my_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTree(max_depth=max_depth)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors[i] = np.mean(y_pred != y)
        print("Our decision tree took %f seconds" % (time.time()-t))

        plt.plot(depths, my_tree_errors, label="mine")

        t = time.time()
        sklearn_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors[i] = np.mean(y_pred != y)
        print("scikit-learn's decision tree took %f seconds" % (time.time()-t))


        plt.plot(depths, my_tree_errors, label="sklearn")
        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "q2_4_tree_errors.pdf")
        plt.savefig(fname)

        tree = DecisionTreeClassifier(max_depth=1)
        tree.fit(X, y)

        y_pred = tree.predict(X)
        error = np.mean(y_pred != y)

        print("Error: %.3f" % error)


    elif question == "3":
        dataset = load_dataset("citiesSmall.pkl")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]
        model = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=1)
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)

    elif question == "3.1":
        dataset = load_dataset("citiesSmall.pkl")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]
        #print("n = %d" % X.shape[0])
        #print(X_test)

        depths = np.arange(1, 15)  # depths to try

        tr_errors = np.zeros(depths.size)
        te_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTree(max_depth=max_depth)
            model.fit(X, y)
            y_pred = model.predict(X)
            tr_errors[i] = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_errors[i] = np.mean(y_pred != y_test)
        #print(te_errors)

        # print("Training error: %.3f" % tr_errors)
        # print("Testing error: %.3f" % te_errors)

        plt.plot(depths, tr_errors, label="training error")
        plt.plot(depths, te_errors, label="test error")

        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "q3_1 training+testing errors.pdf")
        plt.savefig(fname)

        tree = DecisionTreeClassifier(max_depth=1)
        tree.fit(X, y)

        y_pred = tree.predict(X)
        error = np.mean(y_pred != y)


    elif question == "3.2":
        dataset = load_dataset("citiesSmall.pkl")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]

        half_X = np.split(X, 2)
        X1 = half_X[0]
        X2 = half_X[1]
        #print(X1)
        #print(X2)

        half_y = np.split(y, 2)
        y1 = half_y[0]
        y2 = half_y[1]
        #print(y1)
        #print(y2)

        depths = np.arange(1, 15)  # depths to try

        tr_errors = np.zeros(depths.size)
        te_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTree(max_depth=max_depth)
            model.fit(X, y)
            y_pred = model.predict(X)
            tr_errors[i] = np.mean(y_pred != y)

            model.fit(X1, y1)
            y_pred = model.predict(X2)
            te_errors[i] = np.mean(y_pred != y2)
            #print(te_errors)

            # print("Training error: %.3f" % tr_errors)
            # print("Testing error: %.3f" % te_errors)

        #plt.plot(depths, tr_errors, label="training error")
        plt.plot(depths, te_errors, label="test error")
        print(te_errors)

        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "q3_2 validation set error.pdf")
        plt.savefig(fname)

        tree = DecisionTreeClassifier(max_depth=1)
        tree.fit(X, y)

        y_pred = tree.predict(X)
        error = np.mean(y_pred != y)


    if question == '4.1':
        dataset = load_dataset("citiesSmall.pkl")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]
        k = 1
        model = KNN(k)
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        print("Training error for k = {} is {}".format(k, tr_error))

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)

        print("Testing error for k = {} is {}".format(k, te_error))

        utils.plotClassifier(model, X_test, y_test)

        # fname = os.path.join("..", "figs", "q4_1_our_knn.pdf")
        # plt.savefig(fname)
        # print("\nFigure saved as '%s'" % fname)

        # you can use plt.show() to pause the program and show your figure

    if question == '4.2':
        dataset = load_dataset("citiesBig1.pkl")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]
        k = 1
        model = CNN(k)
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        print("Training error for k = {} is {}".format(k, tr_error))

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)

        print("Testing error for k = {} is {}".format(k, te_error))

        print("Number of variables in subset is {}".format(model.X.shape))

        utils.plotClassifier(model, X_test, y_test)

        fname = os.path.join("..", "figs", "q4_2_our_cnn.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
