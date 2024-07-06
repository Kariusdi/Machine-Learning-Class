import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import numpy as np
from utils.methods import *
from utils.data import *

def Lab1(df, holdout_split, random_state, sample_size):
    y = "Weight"
    holdOut_log_main = []
    
    for i in holdout_split:
        holdOut_log = []
        for seed in range(random_state):
            df = random_data(df, seed=seed, sample_size=sample_size)
            holdOut_log.append(HoldOut(df, y, i))
        holdOut_log_main.append(np.mean(holdOut_log))
   
    avg_of_holdout = np.mean(holdOut_log_main)
    sd = np.std(holdOut_log_main)
    
    return holdOut_log_main, avg_of_holdout, sd

def Lab2(df, cross_kfold, random_state, sample_size):
    y = "Weight"
    cross_log_main = []
    
    for i in cross_kfold:
        cross_log = []
        for seed in range(random_state):
            df = random_data(df, seed=seed, sample_size=sample_size)
            cross_log.append(CrossValidation(df, y, i))
        cross_log_main.append(np.mean(cross_log))
        
    avr_of_cross = np.mean(cross_log_main)
    sd = np.std(cross_log_main)
    
    return cross_log_main, avr_of_cross, sd

def Lab3(df, random_state, holdout_split, cross_kfold, sample_size):
    y = "Weight"
    
    holdOut_log = []
    for seed in range(random_state):
        df = random_data(df, seed=seed, sample_size=sample_size)
        holdOut_log.append(HoldOut(df, y, holdout_split))
        
    avg_holdOut = np.mean(holdOut_log)
    sd_holdOut = np.std(holdOut_log)
    
    cross_log = []
    for seed in range(random_state):
        df = random_data(df, seed=seed, sample_size=sample_size)
        cross_log.append(CrossValidation(df, y, cross_kfold))
        
    avg_cross = np.mean(cross_log)
    sd_cross = np.std(cross_log)

    return avg_holdOut, avg_cross, sd_holdOut, sd_cross

def Lab4(df, random_state, holdout_split, cross_kfold, sample_size):
    y = "Weight"
    avg_holdOut, avg_cross, sd_holdOut, sd_cross = Lab3(df, random_state, holdout_split, cross_kfold, sample_size)
    
    resub_log = []
    for seed in range(random_state):
        df = random_data(df, seed=seed, sample_size=sample_size)
        resub_log.append(Resubstitution(df, y))
    
    avg_resub = np.mean(resub_log)
    sd_resub = np.std(resub_log)
        
    return avg_holdOut, avg_cross, avg_resub, sd_holdOut, sd_cross, sd_resub

if __name__ == "__main__":
    
    # Prepare data set
    # Get data frame from csv
    df = pd.read_csv('./datasets/HeightWeight.csv')
    AllData_RMSE = Resubstitution(df, "Weight")
    print('Reference (RMSE): ', AllData_RMSE)


    # Function Arguments
    # Lab1 and 2 => (dataframe, holdout ratio or cross folds array, seed_round)
    # Lab3 => (dataframe, seed_round, holdout ratio, cross folds)
    
    # -----------------
    ##____Warning____##
    ## if seed round > 1000, be careful for lagging
    # -----------------
    
    # # Lab 1
    # holdout_arr = [0.3, 0.5, 0.8]
    # _, avg_of_holdout, sd = Lab1(df, holdout_split=holdout_arr,random_state=1000, sample_size=1000)
    # print(avg_of_holdout, sd)
    
    # # Lab 2
    # crossval_arr = [5, 10, 20]
    # _, avg_of_holdout, sd = Lab2(df, cross_kfold=crossval_arr, random_state=100, sample_size=1000)
    # print(avg_of_holdout, sd)
    
    # # Lab 3
    # avg_holdOut, avg_cross, sd_holdOut, sd_cross = Lab3(df, random_state=100, holdout_split=0.5, cross_kfold=10, sample_size=1000)
    # print(avg_holdOut, avg_cross, sd_holdOut, sd_cross)
    
    # # Lab 4
    # avg_holdOut, avg_cross, avg_resub, sd_holdOut, sd_cross, sd_resub = Lab4(df, random_state=100, holdout_split=0.5, cross_kfold=10, sample_size=1000)
    # print(avg_holdOut, avg_cross, avg_resub, sd_holdOut, sd_cross, sd_resub)