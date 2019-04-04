"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils
import pdb

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X  # just memorize the training data
        self.y = y

    def predict(self, Xtest):
        M = Xtest.shape[0]  # number of testing points to classify
        N = self.X.shape[0]  # number of training points

        allDistancesToTrainingPts = utils.euclidean_dist_squared(self.X, Xtest)
        y_pred = np.zeros(M)

        for m in range(M):
            labels = []
            distancesToTrainingPts = allDistancesToTrainingPts[:, m] # gives me all the distances to the training points for this testing point
            sortedDistancesIndices = np.argsort(distancesToTrainingPts) # gives me the indices that would sort the distances

            # undecided about the variable name here
            for n in range(self.k):
                i = sortedDistancesIndices[n] # for loop 0, the 0th closest neighbor's index
                y_i = self.y[i] # for loop 0, the label of the 0th closest neighbor
                labels.append(y_i) # add the label to our list of labels

            y_pred[m] = max(set(labels), key=labels.count) # for each unique element in labels, find out how many times it occurs in the list and return max
            # in the case of a tie, the first maximum will be chosen

        return y_pred



class CNN(KNN):

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : an N by D numpy array
        y : an N by 1 numpy array of integers in {1,2,3,...,c}
        """

        Xcondensed = X[0:1,:]
        ycondensed = y[0:1]

        for i in range(1,len(X)):
            x_i = X[i:i+1,:]
            dist2 = utils.euclidean_dist_squared(Xcondensed, x_i)
            inds = np.argsort(dist2[:,0])
            yhat = utils.mode(ycondensed[inds[:min(self.k,len(Xcondensed))]])

            if yhat != y[i]:
                Xcondensed = np.append(Xcondensed, x_i, 0)
                ycondensed = np.append(ycondensed, y[i])

        self.X = Xcondensed
        self.y = ycondensed
