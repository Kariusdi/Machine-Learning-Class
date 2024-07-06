import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import numpy as np
from utils.methods import HoldOut, CrossValidation

# Data sampling
def Random_Data(df ,seed, sample_size):
    rng_df = df.sample(n=sample_size, random_state=seed)
    return rng_df

# For sampling observe
def Test_output_handel():
    i = 0
    for i in range(3):
        Random_Data(df, 1+i).to_csv('rngdata{}.csv'.format(i+1), index= False) 
        print(i)
        
def Initialize_Data(df, Y_col):
    X = df.drop(columns=[Y_col])
    y = df[Y_col]
    return X, y

# Make linear model with all data
def LinearRegression_AllData(df, Y_col):
    
    X, y = Initialize_Data(df, Y_col)

    # Initial model with all data
    model = LinearRegression()
    model.fit(X, y)

    # RMSE
    Y_pred = model.predict(X)
    RMSE = root_mean_squared_error(y, Y_pred)

    return RMSE

def Lab3(df, random_state, holdout_split, cross_kfold):
    y= "Weight"

    # average of holdout
    holdOut_log = []
    for seed in range(random_state):
        df = Random_Data(df, seed=seed, sample_size=1000)
        holdOut_log.append(HoldOut(df, y, holdout_split))
        
    avr_holdOut = np.mean(holdOut_log)
    sd_holdOut = np.std(holdOut_log)
    
    # average of holdout
    cross_log = []
    for seed in range(random_state):
        df = Random_Data(df, seed=seed, sample_size=1000)
        cross_log.append(CrossValidation(df, y, cross_kfold))
        
    avr_cross = np.mean(cross_log)
    sd_cross = np.std(cross_log)

    return avr_holdOut, avr_cross, sd_holdOut, sd_cross

def Lab1(df, holdout_split, random_state):
    y= "Weight"
    holdOut_log_main = []
    
    for i in holdout_split:
        # average of holdout for each %
        holdOut_log = []
        for seed in range(random_state):
            df = Random_Data(df, seed=seed, sample_size=1000)
            holdOut_log.append(HoldOut(df, y, i))
        holdOut_log_main.append(np.mean(holdOut_log))
   
    avg_of_holdout = np.mean(holdOut_log_main)
    sd = np.std(holdOut_log_main)
    
    return holdOut_log_main, avg_of_holdout, sd

def Lab2(df, cross_kfold, random_state):
    y= "Weight"
    cross_log_main = []
    
    for i in cross_kfold:
        # average of cross validation
        cross_log = []
        for seed in range(random_state):
            df = Random_Data(df, seed=seed, sample_size=1000)
            cross_log.append(CrossValidation(df, y, i))
        cross_log_main.append(np.mean(cross_log))
        
    avr_of_cross = np.mean(cross_log_main)
    sd = np.std(cross_log_main)
    
    return cross_log_main, avr_of_cross, sd


if __name__ == "__main__":
    
    # Prepare data set
    # Get data frame from csv
    df = pd.read_csv('./datasets/HeightWeight.csv')
    AllData_RMSE = LinearRegression_AllData(df, "Weight")
    print('Reference (RMSE): ', AllData_RMSE)

    crossval_arr = [5, 10, 20]
    holdout_arr = [0.3, 0.5, 0.8]

    # Function Arguments
    # Lab1 and 2 => (dataframe, holdout ratio or cross folds array, seed_round)
    # Lab3 => (dataframe, seed_round, holdout ratio, cross folds)
    
    # -----------------
    ##____Warning____##
    ## if seed round > 1000, be careful for lagging
    # -----------------

    
    # # Test Lab 3
    # avr_holdOut, avr_cross, sd_holdOut, sd_cross = Lab3(df, random_state=100, holdout_split=0.5, cross_kfold=10)
    # print(avr_holdOut, avr_cross, sd_holdOut, sd_cross)
    
    # # Test Lab 1
    # _, avg_of_holdout, sd = Lab1(df, holdout_split=holdout_arr,random_state=1000)
    # print(avg_of_holdout, sd)
    
    # # Test Lab 2
    # _, avg_of_holdout, sd = Lab2(df, cross_kfold=crossval_arr, random_state=100)
    # print(avg_of_holdout, sd)
         
    