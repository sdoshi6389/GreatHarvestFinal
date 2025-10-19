import pandas as pd

def naive_forecast(df):

    #sorts values before applying any operation and sorting by store, product, and date ensures previous day refers to 
    #the prior date of that specific product at the appropriate store
    df = df.sort_values(["store","product","date"]).copy()

    #naive prediction takes previous days sales as the prediction for today
    df["pred_naive"] = df.groupby(["store","product"])["y"].shift(1)

    return df

def seasonal_naive(df, season=7):

    #sorts values before applying any operation and sorting by store, product, and date ensures previous day refers to 
    #the prior date of that specific product at the appropriate store
    df = df.sort_values(["store","product","date"]).copy()

    #naive seasonal takes previous weeks sales (exactly 7 days prior) as the prediction for today
    df["pred_snaive"] = df.groupby(["store","product"])["y"].shift(season)
    
    return df
