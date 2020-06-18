import numpy as np
import pandas as pd

# Imports for the individual submodels
from scipy import stats
from statsmodels.tsa.stattools import acf

from methods.model import model

class deltaMethod(model):
    def __init__(self,data,test_num=0):
        """ Class for the Delta method

        Parameters
        ----------
        @Parameters data        -   (required) pd.DataFrame of the data series,
                                    each column is a new series. Not all series
                                    need to have the same length
        @Parameters test_num    -   (default 0) int of number of observations
                                    at the end of series to use for testing

        Returns
        ----------
        @Returns delta   -   delta forecasting object
        """

        # Initiate parent class and inherit all attributes and methods
        super().__init__(data=data,args=None,test_num=test_num)


    def fit(self):
        """Fit delta model

        Parameters
        ----------

        Returns
        ----------
        None
        """
        self.fitted = {}
        self.fitted_vals = {}
        out = {
            'I_rho': '',\
            'I_A': '',\
            'I_ln': '',\
            'Params': ''
        }

        def seasonal_matrix(arr):
            arr = arr.dropna()
            a = np.asarray(arr)
            tau = a.shape[0] // m
            a = a[-(tau*12):]
            a = a.reshape((tau, m))
            return a

        def anova(arr):
            a = seasonal_matrix(arr)
            tau = a.shape[0]
            zbar = a.mean(0)
            v_zbar = zbar.var()
            v = a.ravel().var()
            stat = m * (tau-1) / (m - 1) * v_zbar / (v - v_zbar)
            i_a = stat > stats.f(m - 1, m * (tau-1)).ppf(.9)
            return i_a

        def r(arr):
            arr = arr.dropna()
            rhos = acf(arr, nlags=12, fft=True)
            test = arr.shape[0] * rhos[-1] ** 2 / (1 + (rhos[1:-1]**2).sum())
            return test > stats.chi2(1).ppf(0.9)

        def sgnmin(a, b):
            return np.sign(a) * min(np.abs(a), np.abs(b)) * ((a * b) > 0)

        #go through data series, creating indicators:
        for series in self.data.columns:
            x = self.data.loc[:,series].dropna()
            I_ln = (x > 1).all() #should we take log?
            y = np.log(x) if I_ln else x
            I_rho = y.diff().var() < (1.2 * y.var()) #should we use differenced series?
            z = y.diff() if I_rho else y

            #next three functions copied from KS GitHub
            'maybe shouldnt use 12?'
            m = 12

            I_A = anova(z) #check for presence of seasonality

            I_R = r(z) #uses the funky r formula to check for seasonal autocorrelation

            #delta method begins by sorting series into ordered list & taking trend means
            zj = z.dropna()
            zi = np.asarray(zj.iloc[np.argsort(np.abs(zj))])
            #sgnmin adds robustness to series

            if I_rho and not I_A: #if we have non-stationary (unit root) and non-seasonal data

                #take means of entire data (ex. 1 largest, ex. 3 largest, only 6 most recent, YoY diff)
                d1 = zi[:-1].mean()
                d2 = zi[:-3].mean()
                dr = zj.iloc[-6:].mean()
                dm = zj.mean()
                ds = y.diff(12).mean() / 12
                dm = sgnmin(dm, ds)
                dr_star = sgnmin(dr, dm)
                out['Params'] = [dr_star, d1, d2]

            elif I_rho and I_A: #if we have non-stationary and seasonal data

                #first transform data to seasonal matrix
                zij = seasonal_matrix(zj)
                zbar = zij.mean(1) #mean across each seasonal cycle

                #take means of entire data (ex. 1 largest, ex. 3 largest, only 6 most recent, YoY diff)
                zbari = np.asarray(zbar[np.argsort(np.abs(zbar))])

                d1 = zbari[:-1].mean()
                d2 = zbari[:-3].mean()
                dr = zbar[-6:].mean()
                dm = zj.mean()
                ds = y.diff(12).mean() / 12
                dr_star = sgnmin(dr,dm)

                w = np.array([1,2,3,9])/15
                s_star = w @ zij[-4:]
                s = s_star - s_star.mean()

                out['Params'] = [dr_star, d1, d2, s]

            elif not I_rho and not I_A: #if stationary and non-seasonal

                #we use simple average
                mu_m = zj.iloc[-m:].mean()
                mu_6m = zj.iloc[-6*m:].mean() #maybe error could be thrown here? do min(6*m,series.shape[0])
                out['Params'] = [mu_m, mu_6m]

            elif not I_rho and I_A: #if stationary and seasonal

                #first transform data to seasonal matrix
                zij = seasonal_matrix(zj)

                w = np.array([1, 2, 3, 4, 5, 20]) / 35
                s_star = w @ zij[-6:]
                s = s_star - s_star.mean()

                zbar = zij.mean(1)

                out['Params'] = [zbar, s]

            #for delta, we need to know seasonality and stationarity
            out['I_rho'] = I_rho
            out['I_A'] = I_A
            out['I_ln'] = I_ln
            self.fitted[series] = out
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

        def sgnmin(a, b):
            return np.sign(a) * min(np.abs(a), np.abs(b)) * ((a * b) > 0)

        for series in self.data.columns:

            cols = [series] + ["Delta Method"]
            self.forecasts[series] = pd.DataFrame(index=np.arange(steps),columns=cols)
            self.forecasts[series].loc[:,series] = true_vals.loc[:,series]

            #extract indicators created in fit() method
            I_rho = self.fitted[series]['I_rho']
            I_A = self.fitted[series]['I_A']
            I_ln = self.fitted[series]['I_ln']

            x = self.data.loc[:,series].dropna()
            y = np.log(x) if I_ln else x
            y_hat = np.zeros(steps)
            # Generate the dataframe in which to save the forecasts

            if I_rho and not I_A: #if we have non-stationary (unit root) and non-seasonal data

                #then forecasts use our d_star, d1 and d2 measures:
                dr_star = self.fitted[series]['Params'][0]
                d1 = self.fitted[series]['Params'][1]
                d2 = self.fitted[series]['Params'][2]

                y_hat[0] = y.iloc[-1] + sgnmin(dr_star, d1)
                for i in range(1, steps):
                    y_hat[i] = y_hat[i-1] + sgnmin(dr_star, d2)

            elif I_rho and I_A: #if we have non-stationary and seasonal data

                #then forecasts use our d_star, d1, d2 and s measures:
                dr_star = self.fitted[series]['Params'][0]
                d1 = self.fitted[series]['Params'][1]
                d2 = self.fitted[series]['Params'][2]
                s = self.fitted[series]['Params'][3]

                y_hat[0] = y.iloc[-1] + sgnmin(dr_star, d1) + s[0]
                for i in range(1, steps):
                    loc = i - 12 * (i // 12)
                    y_hat[i] = y_hat[i-1] + sgnmin(dr_star, d2) + s[loc]

            elif not I_rho and not I_A: #if stationary and non-seasonal

                #we need the average values of last m and last 6*m ovservations
                mu_m = self.fitted[series]['Params'][0]
                mu_6m = self.fitted[series]['Params'][1]

                y_hat[0] = mu_m
                for i in range(1, steps):
                    y_hat[i] = 0.5 * (mu_1 + mu_6m)

            elif not I_rho and I_A: #if stationary and seasonal

                #then forecasts use our zbar and s measures:

                zbar = self.fitted[series]['Params'][0]
                s = self.fitted[series]['Params'][1]

                y_hat[0] = zbar[-1] + s[0]
                mu_1 = zbar[-1]
                mu_6 = zbar[-6:].mean()
                for i in range(1, steps):
                    loc = i - 12 * (i // 12)
                    y_hat[i] = 0.5 * (mu_1 + mu_6) + s[loc]

            self.forecasts[series].loc[:, "Delta Method"] = np.exp(y_hat) if I_ln else y_hat

        self.forecasts_generated = True
