import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import numpy as np
from experiments import Initialize_Data

def HoldOut(df, Y_col, testsize):
    # แยกข้อมูลเป็น X (feature) และ y (target) โดยใช้ฟังก์ชัน Initialize_Data
    X, y = Initialize_Data(df, Y_col)
    # แบ่งข้อมูลเป็น train และ test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=testsize, random_state=0)

    # สร้างและฝึกโมเดล Linear Regression
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # ทำนายค่า y จากข้อมูลทดสอบ
    y_pred = model.predict(X_test)

    # คำนวณค่า RMSE
    RMSE = root_mean_squared_error(Y_test, y_pred)
    return RMSE

def CrossValidation(df, Y_col, fold):
    X, y = Initialize_Data(df, Y_col)

    # สร้าง KFold object สำหรับการทำ Cross-Validation
    kf = KFold(n_splits=fold, shuffle = True, random_state=0)
    
    rmse_values = []
    # แบ่งข้อมูลเป็น k-folds และทำการฝึกโมเดลและทดสอบ
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # สร้างและฝึกโมเดล Linear Regression
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate RMSE for this fold
        rmse_fold = root_mean_squared_error(y_test, y_pred)
        rmse_values.append(rmse_fold)
        
    RMSE = np.mean(rmse_values)
    return RMSE

def Resubstitution(df, Y_col):
    X, y = Initialize_Data(df, Y_col)

    # Initial model with all data
    model = LinearRegression()
    model.fit(X, y)

    Y_pred = model.predict(X)
    
    RMSE = root_mean_squared_error(y, Y_pred)
    return RMSE