import numpy as np
import pandas as pd

# Imports for the individual submodels
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX

from methods.model import model

class maMethod(model):
    def __init__(self,data,test_num=0):
        """ Class for the MA method

        Parameters
        ----------
        @Parameters data        -   (required) pd.DataFrame of the data series,
                                    each column is a new series. Not all series
                                    need to have the same length
        @Parameters test_num    -   (default 0) int of number of observations
                                    at the end of series to use for testing

        Returns
        ----------
        @Returns MA   -   MA forecasting object
        """

        # Initiate parent class and inherit all attributes and methods
        super().__init__(data=data,args=None,test_num=test_num)


    def fit(self):
        """Fit MA model

        Parameters
        ----------

        Returns
        ----------
        None
        """
        self.fitted = {}

        for series in self.data.columns:
            x = self.data.loc[:,series].dropna()
            self.fitted[series] = {
                "MA1": SARIMAX(x, order = (0, 0, 1), trend="ct").fit(disp = False),
                "MA2": SARIMAX(x, order = (0, 0, 2), trend="ct").fit(disp = False),
                "MA3": SARIMAX(x, order = (0, 0, 3), trend="ct").fit(disp = False),
                #"MA6": SARIMAX(x, order = (0, 0, 6), trend="ct").fit(disp = False),
                #"MA9": SARIMAX(x, order = (0, 0, 9), trend="ct").fit(disp = False),
                #"MA12": SARIMAX(x, order = (0, 0, 12), trend="ct").fit(disp = False)
            }

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

            cols = [series] + list(self.fitted[series].keys())
            res = pd.DataFrame(index=np.arange(steps), columns=cols)

            res.loc[:,series] = true_vals.loc[:,series]

            for key in self.fitted[series].keys():
                y_hat = self.fitted[series][key].forecast(steps).array
                res.loc[:, key] = y_hat
            self.forecasts[series] = res

        self.forecasts_generated = True