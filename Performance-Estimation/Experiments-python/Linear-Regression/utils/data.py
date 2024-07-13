# Data sampling
def random_data(df ,seed, sample_size):
    # Randomly sample rows from the DataFrame based on the given sample size and seed
    rng_df = df.sample(n=sample_size, random_state=seed)
    return rng_df

# For sampling observe
def Test_output_handel(df):
    i = 0

    # Perform sampling three times, incrementing the seed by 1 each time
    for i in range(3):
        random_data(df, 1+i).to_csv('rngdata{}.csv'.format(i+1), index= False) 
        print(i)

# Function to initialize data
def Initialize_Data(df, Y_col):
    # Separate features (X) and target (y) from the DataFrame
    X = df.drop(columns=[Y_col])
    y = df[Y_col]
    return X, y