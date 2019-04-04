def predict(X):
    M, D = X.shape
    X = np.round(X)

    yhat = np.zeros(M)

    for m in range(M):
        if X[m, 1] > 36:
            if X[m, 0] > -97:
                yhat[m] = 0
            else:
                yhat[m] = 1
        else:
            if X[m, 0] > -116:
                yhat[m] = 1
            else:
                yhat[m] = 0

    return yhat
