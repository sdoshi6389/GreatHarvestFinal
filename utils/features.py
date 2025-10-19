import pandas as pd, numpy as np

def add_time_features(df):
    df = df.copy()

    #captures day of week
    df["dow"] = df["date"].dt.weekday

    #captures week number (either 1-52 or 1-53 depending on year)
    df["week"] = df["date"].dt.isocalendar().week.astype(int)

    #captures calendar month
    df["month"] = df["date"].dt.month

    #captures year
    df["year"] = df["date"].dt.year

    #cyclic encodings - transform variables into sin/cosine pairs for ml models so data appears in a continuous, smooth,
    #circular representation rather than a discontinuous numeric representation (like the numbers 0 - 6 for days of week)
    df["dow_sin"] = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"] = np.cos(2*np.pi*df["dow"]/7)
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)   

    return df

def add_lags_rolls(df: pd.DataFrame, lags, roll_windows, group_cols=["store","product"]):

    #sorts the data so that each (store, product) group is in chronological order
    df = df.sort_values(group_cols + ["date"]).copy()

    #generates lag columns - capture previous values of y
    #for each lag value L, a new column lag_L is added that stores the sales value L days ago for that (store, product)
    #stores values in increments of L for the index
    for L in lags:
        df[f"lag_{L}"] = df.groupby(group_cols)["y"].shift(L)

    #rolling statistics - rolling means and rolling standard deviations summarize past performance over a window
    #data is shifted by 1 so it doesn't use the current days data (strictly uses past data)
    for W in roll_windows:

        #compute rolling mean of y for each (store, product)
        df[f"rollmean_{W}"] = (
            df.groupby(group_cols, group_keys=False)["y"]
              .apply(lambda s: s.shift(1).rolling(W).mean())
        )

        #compute rolling standard deviation of y for each (store, product)
        df[f"rollstd_{W}"] = (
            df.groupby(group_cols, group_keys=False)["y"]
              .apply(lambda s: s.shift(1).rolling(W).std())
        )

    return df

def train_val_test_split_by_date(df, val_days, test_days):

    #find the latest date in the dataset
    last_date = df["date"].max()

    #determines where the test set begins
    test_start = last_date - pd.Timedelta(days=test_days-1)

    #determines where the validation set begins
    val_start  = test_start - pd.Timedelta(days=val_days)

    #all data before the validation set, this data is the training data
    train = df[df["date"] < val_start]

    #all data from the validation set to the test set, this data is the validation data
    val = df[(df["date"] >= val_start) & (df["date"] < test_start)]

    #all data from the test set onwards, this data is the test data
    test = df[df["date"] >= test_start]
    
    return train, val, test
