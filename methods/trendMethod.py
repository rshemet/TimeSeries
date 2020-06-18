import numpy as np
import pandas as pd

# Imports for the individual submodels
from scipy import stats
from statsmodels.regression.linear_model import OLS

from methods.model import model

class trendMethod(model):
    def __init__(self,data,test_num=0):
        """ Class for the trend method

        Parameters
        ----------
        @Parameters data        -   (required) pd.DataFrame of the data series,
                                    each column is a new series. Not all series
                                    need to have the same length
        @Parameters test_num    -   (default 0) int of number of observations
                                    at the end of series to use for testing

        Returns
        ----------
        @Returns trend   -   trend forecasting object
        """

        # Initiate parent class and inherit all attributes and methods
        super().__init__(data=data,args=None,test_num=test_num)


    def fit(self):
        """Fit OLS model

        Parameters
        ----------

        Returns
        ----------
        None
        """
        self.fitted = {}
        self.fitted_vals = {}

        for series in self.data.columns:
            x = self.data.loc[:,series].dropna()
            self.last_observed = len(x)
            t = np.hstack([np.ones(len(x))[:, np.newaxis],np.arange(len(x))[:, np.newaxis]])
            coefs = np.linalg.inv(t.T@t)@(t.T@x)
            self.fitted[series] = coefs

        self.fit_success = True

    def forecast(self,true_vals):
        """Function to forecast using the previously fitted models

        Parameters
        ----------
        @Parameter true_vals    -   (default None) optional pd.DataFrame of the
                                    values to forecast using the data. Assumes
                                    they are adjacent to existing data, and that
                                    the column dimension matches.

        Returns
        ----------
        None
        """
        assert self.fit_success == True, "Please fit model before forecasting"
        assert self.data.shape[1] == true_vals.shape[1], "Dimension mismatch"

        self.forecasts = {}

        steps = true_vals.shape[0]

        for series in self.data.columns:

            cols = [series] + ["Trend"]
            self.forecasts[series] = pd.DataFrame(index=np.arange(steps),columns=cols)
            self.forecasts[series].loc[:,series] = true_vals.loc[:,series]

            t_hat = np.arange(self.last_observed, self.last_observed+steps)[:, np.newaxis]
            y_hat = np.hstack([np.ones(steps)[:, np.newaxis],t_hat]) @ self.fitted[series]

            self.forecasts[series].loc[:, "Trend"] = y_hat

        self.forecasts_generated = True