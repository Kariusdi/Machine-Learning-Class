# Performance Estimation using Python

From the experiments that we have tested with **_WEKA_** by using 3 methods, which are

- Hold Out
- K-fold Cross Validation
- Resubstitution

In this time, we are gonna use the same experiments but **_implementing_** with Python instead.

> we are not gonna implement it from scratch, we are gonna use libs for this.

## Proposition

1. เขียนโปรแกรมเพื่อทดสอบความเที่ยงตรง (Precision) และการทดสอบความแม่นยํา (Accuracy) ของวิธี Resubstitution, Holdout และ Cross Validation โดยใช้ข้อมูล height weight โดยให้ออกแบบการทดลองเอง

What we have to do is design the experiment by ourself. So we need to define

- **Seed** (How many step to we want to test)
- **Sample Size**
- **Train / Test percentage** (for Hold Out method)
- **K-fold size** (for Cross Validation method)

and the dataset that we use is **HeightWeight.csv**

We have done the experiment planning, so LET'S START!! ⚡️

## <mark>Firstly, **_Initializing Methods_**</mark>

Of course, we're gonna have 3 methods for all experiments

- Hold Out

```python
def HoldOut(df, Y_col, testsize):
    X = df.drop(columns=[Y_col])
    Y = df[Y_col]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=testsize, random_state=0)

    model = LinearRegression()
    model.fit(X_train, Y_train)

    y_pred = model.predict(X_test)

    RMSE = root_mean_squared_error(Y_test, y_pred)
    return RMSE
```

- K-fold Cross Validation

```python
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
```

- Resubstitution

```python
def Resubstitution(df, Y_col):
    X, y = Initialize_Data(df, Y_col)

    # Initial model with all data
    model = LinearRegression()
    model.fit(X, y)

    Y_pred = model.predict(X)

    RMSE = root_mean_squared_error(y, Y_pred)
    return RMSE
```

> we use RMSE to define the error of the model.

## <mark>Experiment 1 and 2</mark>

result from WEKA

![ex1-2](../assets/ex1-2.png)
</br>

```python
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
```

## <mark>Experiment 3</mark>

result from WEKA

![ex3](../assets/ex3.png)
</br>

```python
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
```

## <mark>Experiment 4</mark>

result from WEKA

![ex4](../assets/ex4.png)
</br>

```python
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
```
