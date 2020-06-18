import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX

from methods.model import model

class arimaMethod(model):
    def __init__(self,data,test_num=0):
        """ Class for ARIMA method

        Parameters
        ----------
        @Parameters data        -   (required) pd.DataFrame of the data series,
                                    each column is a new series. Not all series
                                    need to have the same length
        @Parameters test_num    -   (default 0) int of number of observations
                                    at the end of series to use for testing

        Returns
        ----------
        @Returns arima   -   rho forecasting object
        """

        # Initiate parent class and inherit all attributes and methods
        super().__init__(data=data,args=None,test_num=test_num)

    def fit(self):
        """Fit ARIMA model

        Parameters
        ----------

        Returns
        ----------
        None
        """
        self.fitted = {}
        for i,series in enumerate(self.data.columns):
            x = self.data.loc[:self.train_ix[series]-1,series]
            y = np.log(x)

            self.fitted[series] = {
            "ar1_trend" : SARIMAX(y, order=(1,0,0), trend="ct").fit(maxiter=250, disp=0),
            "arma11_trend" : SARIMAX(y, order=(1,0,1), trend="ct").fit(maxiter=250, disp=0),
            "arima_111" : SARIMAX(y, order=(1, 1, 1), trend="c").fit(maxiter=250, disp=0),
            "arima_011" : SARIMAX(y, order=(0, 1, 1), trend="c").fit(maxiter=250, disp=0),
            "arima_011_nc" : SARIMAX(y, order=(0, 1, 1), trend="n").fit(maxiter=250, disp=0),
            "arima_000_010" : SARIMAX(y, order=(0,0,0), seasonal_order=(0, 1, 0, 12), trend="c").fit(disp=0),
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

        for i, series in enumerate(self.data.columns):
            # Generate the dataframe in which to save the forecasts
            cols = [series] + list(self.fitted[series].keys())
            res = pd.DataFrame(index=np.arange(steps),columns=cols)
            #Currently - using log of true values and also output log of forecasts -
            #not sure but maybe we can change to raw true values and raw forecasts.
            res.loc[:,series] = true_vals.loc[:,series]
            #self.forecasts[series].loc[:,series] = true_vals.loc[:,series]
            for key in self.fitted[series].keys():
                f = self.fitted[series][key].forecast(steps).array
                res.loc[:, key] = np.exp(f)
            self.forecasts[series] = res
        self.forecasts_generated = True

