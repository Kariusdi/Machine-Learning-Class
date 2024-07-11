import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import numpy as np
from utils.methods import *
from utils.data import *

#Lab1: Focuses on Hold-Out validation with different split ratios.
def Lab1(df, holdout_split, random_state, sample_size):
    y = "Weight"    # Target variable
    holdOut_log_main = {split: [] for split in holdout_split}
    
    # Loop over each holdout split ratio
    for i in holdout_split: # Loop over each random seed
        for seed in range(random_state):    # Generate random sample data
            df = random_data(df, seed=seed, sample_size=sample_size)    # Perform holdout validation and store the result
            holdOut_log_main[i].append(HoldOut(df, y, i))
   
    
    holdout_df = pd.DataFrame(holdOut_log_main) # Create a DataFrame to store the results
    holdout_df.loc["Average RMSE"] = holdout_df.mean()   # Calculate and store average RMSE
    holdout_df.loc["SD"] = holdout_df.std() # Calculate and store standard deviation of RMSE

    return holdout_df
    
#Lab2: Focuses on Cross-Validation with different k-fold values.
def Lab2(df, cross_kfold, random_state, sample_size):
    y = "Weight"
    cross_log_main = {folds: [] for folds in cross_kfold}   # Initialize a dictionary to store results for each k-fold
    
    for i in cross_kfold: 
        for seed in range(random_state):
            # Generate random sample data
            df = random_data(df, seed=seed, sample_size=sample_size)
            # Perform cross-validation and store the result
            cross_log_main[i].append(CrossValidation(df, y, i))
        
    # Create a DataFrame to store the results
    crossval_df = pd.DataFrame(cross_log_main)
    # Calculate and store average RMSE
    crossval_df.loc["Average RMSE"] = crossval_df.mean()
    # Calculate and store standard deviation of RMSE
    crossval_df.loc["SD"] = crossval_df.std()

    return crossval_df  
    
#Lab3: Combines Hold-Out and Cross-Validation for comparison.
def Lab3(df, random_state, holdout_split, cross_kfold, sample_size):
    y = "Weight"
    holdOut_log = []
    for seed in range(random_state):
        df = random_data(df, seed=seed, sample_size=sample_size)
        holdOut_log.append(HoldOut(df, y, holdout_split))

    # Calculate average and standard deviation of holdout RMSE
    avg_holdOut = np.mean(holdOut_log)
    sd_holdOut = np.std(holdOut_log)
    holdout_df = pd.DataFrame(holdOut_log, columns=[f'holdout: {holdout_split}'])
    holdout_df.loc["Average RMSE"] = avg_holdOut
    holdout_df.loc["SD"] = sd_holdOut
   

    cross_log = []  # List to store cross-validation results

    for seed in range(random_state):
        df = random_data(df, seed=seed, sample_size=sample_size)
        cross_log.append(CrossValidation(df, y, cross_kfold))
        
    # Calculate average and standard deviation of cross-validation RMSE
    avg_cross = np.mean(cross_log)
    sd_cross = np.std(cross_log)
    crossval_df = pd.DataFrame(cross_log, columns=[f'Cross_val: {cross_kfold}'])
    crossval_df.loc["Average RMSE"] = avg_cross
    crossval_df.loc["SD"] = sd_cross
   
    # Combine holdout and cross-validation results into one DataFrame
    combined_df = pd.concat([holdout_df, crossval_df], axis=1)

    return avg_holdOut, avg_cross, sd_holdOut, sd_cross, combined_df

#Lab4: Adds Resubstitution to the comparison for a more comprehensive analysis.
def Lab4(df, random_state, holdout_split, cross_kfold, sample_size):
    y = "Weight"
    avg_holdOut, avg_cross, sd_holdOut, sd_cross, combined_df = Lab3(df, random_state, holdout_split, cross_kfold, sample_size)
    
    resub_log = []  # List to store resubstitution results
    for seed in range(random_state):
        df = random_data(df, seed=seed, sample_size=sample_size)
        resub_log.append(Resubstitution(df, y))
    
    # Calculate average and standard deviation of resubstitution RMSE
    avg_resub = np.mean(resub_log)
    sd_resub = np.std(resub_log)
    resub_df = pd.DataFrame(resub_log, columns=['Resubstitution'])
    resub_df.loc["Average RMSE"] = avg_resub
    resub_df.loc["SD"] = sd_resub

    # Combine resubstitution results with previous combined DataFrame
    combined_df = pd.concat([combined_df, resub_df], axis=1)

    return avg_holdOut, avg_cross, avg_resub, sd_holdOut, sd_cross, sd_resub, combined_df

if __name__ == "__main__":
    
    # Prepare data set
    # Get data frame from csv
    df = pd.read_csv('E:\ML\Machine-Learning-Class\Performance-Estimation\Experiments-python\HeightWeight.csv')
    # Compute reference RMSE using Resubstitution on the full data
    AllData_RMSE = Resubstitution(df, "Weight")
    print('Reference (RMSE): ', AllData_RMSE)


    # Function Arguments
    # Lab1 and 2 => (dataframe, holdout ratio or cross folds array, seed_round)
    # Lab3 => (dataframe, seed_round, holdout ratio, cross folds)
    
    # -----------------
    ##____Warning____##
    ## if seed round > 1000, be careful for lagging
    # -----------------
    
    # Lab 1
    holdout_arr = [0.9, 0.8, 0.5, 0.2, 0.1]
    result_lab1 = Lab1(df, holdout_split=holdout_arr,random_state=100, sample_size=1000)
    print("\nHoldout Results Table:")
    print(result_lab1)

    
    # Lab 2
    crossval_arr = [20, 10, 5]
    result_lab2 = Lab2(df, cross_kfold=crossval_arr, random_state=100, sample_size=1000)
    print("\nCross-Validation Results Table:")
    print(result_lab2)
    
    # Lab 3
    avg_holdOut, avg_cross, sd_holdOut, sd_cross, combined_df = Lab3(df, random_state=100, holdout_split=0.5, cross_kfold=10, sample_size=100)
    print("\n Lab 3 ")
    print(combined_df)
    print(f"Lab 3 Average Holdout RMSE: ", avg_holdOut)
    print(f"Lab 3 Average Cross-Validation RMSE: ", avg_cross)
    print(f"Lab 3 Holdout RMSE SD: ", sd_holdOut)
    print(f"Lab 3 Cross-Validation RMSE SD: ", sd_cross)
    
    
    # Lab 4
    avg_holdOut, avg_cross, avg_resub, sd_holdOut, sd_cross, sd_resub, combined_df = Lab4(df, random_state=100, holdout_split=0.5, cross_kfold=10, sample_size=10000)
    print("\n Lab 4 ")
    print(combined_df)
    print(f"Lab 4 Average Holdout RMSE: ", avg_holdOut)
    print(f"Lab 4 Average Cross-Validation RMSE: ", avg_cross)
    print(f"Lab 4 Average Resubstitution RMSE: ", avg_resub)
    print(f"Lab 4 Holdout RMSE SD: ", sd_holdOut)
    print(f"Lab 4 Cross-Validation RMSE SD: ", sd_cross)
    print(f"Lab 4 Resubstitution RMSE SD: ", sd_resub)
