"""Advanced Econometrics Group Assignment"""
import itertools
import numpy as np
import pandas as pd
import copy
import time
import warnings

from matplotlib import pyplot as plt

from methods import *

np.random.seed(42)

def data_manipulation(df,test_df,obs_cut=0):
    """ Adjust the training and testing data lengths

    Parameters
    ----------
    df          :   pd.DataFrame
    test_df     :   pd.DataFrame
    obs_cut     :   int

    Returns
    -------
    train       :   pd.DataFrame
    test        :   pd.DataFrame

    """
    # If we are not cutting, we can use the original data
    if obs_cut ==0:
        return df, test_df

    # Generate counts and adjust them to match desired cut
    counts = df.count() - obs_cut
    train = copy.copy(df)
    test = pd.DataFrame(index=np.arange(18), columns=df.columns)
    for s in counts.index:
        if counts.loc[s] < 1:
            test.drop([s], axis=1, inplace=True)
            train.drop([s], axis=1, inplace=True)
        else:
            test.loc[:, s] = df.loc[:, s].iloc[counts.loc[s]:counts.loc[s]+18]
            train.loc[:, s].iloc[counts.loc[s]:] = np.nan

    return train, test


def model_init(df, series=None):
    """ Helper function to generate the model dictionary based on how many obs
    to cut off

    Parameters
    ----------
    df      :       pd.DataFrame
                    data to use for training the model
    finalcut:       int (defualt=0)
                    How many observations to cut off from the end
    series  :       list (default None)
                    which subset of series to use

    Returns
    -------
    models  :       dict
                    dictionary with key = methods, and objects for methods
    """
    series = series if series is not None else df.columns.to_list()
    data = df.loc[:, series]

    return {
        'expS': expSmoothing(data),
        'theta': thetaMethod(data),
        'rho': rhoMethod(data),
        'sarima': arimaMethod(data),
        'delta': deltaMethod(data),
        'card': CARDMethod(data),
        'trend': trendMethod(data),
        'MA': maMethod(data),
    }


def model_estimation(models, test_data, verbose=True):
    """

    Parameters
    ----------
    models      :       dict
                        Unfitted model classes
    test_data   :       pd.DataFrame
                        data that are the true values for the forecasts
    verbose     :       boolean (default True)
                        whether to print timings

    Returns
    -------
    models      :   dict
                    dictionary of the fitted models
    """
    data = test_data.loc[:, models[list(models.keys())[0]].data.columns]
    start = time.time()
    for key, mod in models.items():
        t = time.time()
        models[key].fit()
        models[key].forecast(data)
        if verbose: print("Completed {} in {:.3f}s".format(key,time.time() - t))
    if verbose: print("Overall time: {:.3f}s".format(time.time() - start))
    return models


def sMAPE(trueval, fcast, *args):
    num = np.abs(trueval - fcast)
    denom = np.abs(trueval) + np.abs(fcast)
    return 100 * 2 / len(fcast) * np.sum(num / denom)


def MASE(trueval, fcast, data, m=12):
    in_samp = data.dropna()
    in_samp_diff = in_samp.diff(periods=m).dropna()
    numerator = np.sum(np.abs(trueval - fcast))
    denominator = ((1 / (len(in_samp) - m)) * np.sum(np.abs(in_samp_diff)))
    return 1 / len(fcast) * numerator / denominator

if __name__ == '__main__':
    pass