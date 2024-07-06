# Data sampling
def random_data(df ,seed, sample_size):
    rng_df = df.sample(n=sample_size, random_state=seed)
    return rng_df

# For sampling observe
def Test_output_handel(df):
    i = 0
    for i in range(3):
        random_data(df, 1+i).to_csv('rngdata{}.csv'.format(i+1), index= False) 
        print(i)
        
def Initialize_Data(df, Y_col):
    X = df.drop(columns=[Y_col])
    y = df[Y_col]
    return X, y