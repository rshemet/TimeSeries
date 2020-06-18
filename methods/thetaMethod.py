import numpy as np
import pandas as pd

# Imports for the individual submodels
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
from statsmodels.tsa.tsatools import add_trend
from statsmodels.regression.linear_model import OLS
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tools.eval_measures import mse

from methods.model import model

from matplotlib import pyplot as plt

class thetaMethod(model):
    def __init__(self, data, test_num=0):
        """ Class for the theta method

        Parameters
        ----------
        @Parameters data        -   (required) pd.DataFrame of the data series,
                                    each column is a new series. Not all series
                                    need to have the same length
        @Parameters test_num    -   (default 0) int of number of observations
                                    at the end of series to use for testing

        Returns
        ----------
        @Returns theta          -   theta method estimation object
        """

        # Initiate parent class and inherit all attributes and methods
        super().__init__(data=data, args=None, test_num=test_num)
        self.forecasts = {}
        self.fitted = {}
        self.best_theta = {}

    @staticmethod
    def estimate(x,theta):
        """Small function to estimate the a0 and b0 parameters for the
        theta methods

        Parameters
        ----------
        @param  x       -   pd.DataFrame of timeseries.
        @param  theta   -   float of the theta value

        Returns
        ----------
        @returns params -   pd.DataFrame of parameters
        """
        t = x.shape[0]
        rhs = np.hstack([np.ones((t, 1)), np.arange(t).reshape((t,1))])
        rhs = pd.DataFrame(rhs, index=x.index, columns=["a0", "b0"])
        lhs = x * (1 - theta)
        mod = OLS(lhs, rhs).fit()
        return mod.params


    def fit(self, folds=3, thetas=(-2, -1, 0, 0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2)):
        """Function to theta models based on Kevin Sheppard's code. Selects the
        best theta for the series based on KFold cross-validation

        Parameters
        ----------
        @Parameters thetas  -   tuple of float theta values to evaluate

        Returns
        ----------
        None
        """

        # Initialise the KFold object
        kf = TimeSeriesSplit(n_splits=folds)

        for i, series in enumerate(self.data.columns):
            x = self.data.loc[:self.train_ix[series] - 1, series]

            mspes = {t: np.empty((folds, 1)) for t in thetas}
            p = pd.DataFrame(None, index=["a0", "b0"], dtype=np.double)
            params = {i: p for i in range(folds)}

            fold_ix = 0
            for tr_ix, te_ix in kf.split(x):
                # Set up data
                x_tr, x_te = x.iloc[tr_ix], x.iloc[te_ix]

                t = x_tr.shape[0]
                k = x_te.shape[0]

                for theta in thetas:
                    # Estimate the different theta models
                    params[fold_ix][theta] = self.estimate(x_tr, theta)
                    # Forecast for different theta models:
                    b0 = params[fold_ix][theta]["b0"]
                    # New RHS for forecasting
                    rhs_oos = np.ones((k, 2))
                    rhs_oos[:, 1] = np.arange(k) + t + 1
                    # Exp. Smoothing term
                    fit_args = {"disp": False, "iprint": -1, "low_memory": True}
                    ses = ExponentialSmoothing(x_tr).fit(**fit_args)
                    alpha = ses.params.smoothing_level
                    # Actual forecasting
                    ses_forecast = ses.forecast(k)
                    trend = (np.arange(k) + 1 / alpha - ((1 -alpha) ** t) / alpha)
                    trend *= 0.5 * b0
                    forecast = np.array(ses_forecast + trend)
                    mspes[theta][fold_ix] = mse(x_te, forecast)

                fold_ix += 1

            # Evaluate the KFold
            for k, v in mspes.items():
                mspes[k] = np.mean(v)

            self.best_theta[series] = min(mspes, key=mspes.get)
            self.fitted[series] = self.estimate(x, self.best_theta[series])
            self.fit_success = True

    def fit1(self, thetas=(-2, -1, 0, 0.5, 0.75, 1.25, 1.5, 2)):
        """Function to theta models based on Kevin Sheppards code. Selects the
        best theta for the series based on KFold cross-validation

        Parameters
        ----------
        @Parameters thetas  -   tuple of float theta values to evaluate

        Returns
        ----------
        None
        """
        self.fitted = {t: {} for t in thetas}
        self.fitted_vals = {t: {} for t in thetas}

        for i, series in enumerate(self.data.columns):
            x = self.data.loc[:self.train_ix[series] - 1, series]
            # Set up the trend data by adding cols for constant and time trend
            joined = add_trend(x, "ct", prepend=False)
            joined.loc[:, "trend"] -= 1

            # set up the OLS
            rhs = joined.iloc[:, -2:]
            rhs.columns = ["a0", "b0"]

            for theta in thetas:
                res = OLS(x * (1 - theta), rhs).fit()
                self.fitted[theta][series] = res
                self.fitted_vals[theta][series] = pd.concat([res.fittedvalues + x * theta, rhs])

        self.fit_success = True

    def forecast(self, true_vals):
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
        assert self.fit_success, "Please fit model before forecasting"
        assert self.data.shape[1] == true_vals.shape[1], "Dimension mismatch"

        steps = true_vals.shape[0]

        for series in self.data.columns:
            # Set up
            x = self.data.loc[:self.train_ix[series] - 1, series]
            k = true_vals.loc[:,series].shape[0]
            t = x.shape[0]

            # Generate the dataframe in which to save the forecasts
            res = pd.DataFrame(index=np.arange(steps),columns=[series, "Theta"])
            res.loc[:, series] = true_vals.loc[:, series]

            # Smoothing parameter
            fit_args = {"disp": False, "iprint": -1, "low_memory": True}
            ses = ExponentialSmoothing(x).fit(**fit_args)
            alpha = ses.params.smoothing_level
            ses_forecast = ses.forecast(k)

            # New RHS for forecasting
            rhs_oos = np.ones((k, 2))
            rhs_oos[:, 1] = np.arange(k) + t + 1
            b0 = self.fitted[series]["b0"]
            trend = (np.arange(k) + 1 / alpha - ((1 - alpha) ** t) / alpha)
            trend *= 0.5 * b0
            res.loc[:, "Theta"] = (ses_forecast + trend).values
            self.forecasts[series] = res

            """
            temp = res.copy()
            temp.index += x.index[-1]

            plt.figure()
            plt.plot(temp.loc[:, series], label="True Forecast", color='black')
            plt.plot(x, label='Fitting Data', color='Gray')
            plt.plot(temp.loc[:, "Theta"], label="Forecast")
            plt.legend()
            plt.show()
            """

        self.forecasts_generated = True
