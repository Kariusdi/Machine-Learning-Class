import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import numpy as np

def HoldOut(df, Y_col, testsize):
    X = df.drop(columns=[Y_col])
    Y = df[Y_col]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=testsize, random_state=0)

    model = LinearRegression()
    model.fit(X_train, Y_train)

    y_pred = model.predict(X_test)

    RMSE = root_mean_squared_error(Y_test, y_pred)
    return RMSE

def CrossValidation(df, Y_col, fold):
    X = df.drop(columns=[Y_col])
    Y = df[Y_col]

    kf = KFold(n_splits=fold, shuffle = True, random_state=0)

    rmse_values = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate RMSE for this fold
        rmse_fold = root_mean_squared_error(y_test, y_pred)
        rmse_values.append(rmse_fold)
    RMSE = np.mean(rmse_values)
    return RMSE