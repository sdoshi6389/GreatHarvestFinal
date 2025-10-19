import numpy as np

#compute the average of absolute difference between actual and predicted values, ignoring nan values
def mae(y, yhat):
    return float(np.nanmean(np.abs(y - yhat)))

#compute the square root of the mean squared differences between actual and predicted values, ignoring nan values
def rmse(y, yhat):
    return float(np.sqrt(np.nanmean((y - yhat)**2)))

#compute the symmetric mean absolute percentage error between actual and predicted values, ignoring nan values
def smape(y, yhat):
    denom = (np.abs(y) + np.abs(yhat)) / 2.0
    val = np.abs(y - yhat) / np.where(denom==0, 1.0, denom)
    return float(100*np.nanmean(val))
