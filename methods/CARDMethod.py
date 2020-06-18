import numpy as np
import pandas as pd

from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import acf
from scipy import stats

from methods.model import model
from methods.deltaMethod import deltaMethod
from methods.rhoMethod import rhoMethod


class CARDMethod(model):
    def __init__(self,data,test_num=0):
        """ Class for CARD method

        Parameters
        ----------
        @Parameters data        -   (required) pd.DataFrame of the data series,
                                    each column is a new series. Not all series
                                    need to have the same length
        @Parameters test_num    -   (default 0) int of number of observations
                                    at the end of series to use for testing

        Returns
        ----------
        @Returns card   -   card forecasting object
        """

        # Initiate parent class and inherit all attributes and methods
        super().__init__(data=data,args=None,test_num=test_num)


    def fit(self):
        """Fit CARD model

        Parameters
        ----------

        Returns
        ----------
        None
        """

        rho = rhoMethod(self.data)
        delta = deltaMethod(self.data)

        rho.fit()
        delta.fit()

        self.fittedRho = rho
        self.fittedDelta = delta

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

        self.fittedRho.forecast(true_vals)
        self.fittedDelta.forecast(true_vals)

        for i, series in enumerate(self.data.columns):
            # Generate the dataframe in which to save the forecasts
            cols = [series] + ["CARD Method"]
            self.forecasts[series] = pd.DataFrame(index=np.arange(steps),columns=cols)
            #Currently - using log of true values and also output log of forecasts -
            #not sure but maybe we can change to raw true values and raw forecasts.
            self.forecasts[series].loc[:,series] = true_vals.loc[:,series]
            #self.forecasts[series].loc[:,series] = true_vals.loc[:,series]

            x = self.data.loc[:,series].dropna()
            I_ln = (x > 1).all()
            yj = np.log(x) if I_ln else x
            I_rho = yj.diff().var() < (1.2 * yj.var())
            z = yj.copy()
            if(I_rho):
                z.loc[:] = yj.loc[:].diff()
            m = 12
            def r(arr):
                arr = arr.dropna()
                rhos = acf(arr, nlags=12, fft=True)
                test = arr.shape[0] * rhos[-1] ** 2 / (1 + (rhos[1:-1]**2).sum())
                return test > stats.chi2(1).ppf(0.9)

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

            I_A = anova(z)

            I_R = r(z)

            y_rho = pd.Series(list(self.data.loc[:,series].dropna().iloc[-13:]) + list(self.fittedRho.forecasts[series]["Rho Method"]))
            y_delta = pd.Series(list(self.data.loc[:,series].dropna().iloc[-13:]) + list(self.fittedDelta.forecasts[series]["Delta Method"]))

            yhat_avg = 0.5 * (y_rho + y_delta)

            tph = yhat_avg.shape[0]
            t = yj.shape[0]
            rhs = {"const": pd.Series(np.ones_like(yhat_avg))}
            I_4 = t > 4 * m
            I_5 = I_rho * (m in (4,12,13))
            if I_rho:
                rhs["rho"] = yhat_avg.shift(1)
            if I_rho and I_R and I_4:
                rhs["rho_m"] = yhat_avg.shift(12)
                rhs["rho_m_1"] = yhat_avg.shift(13)
            if I_A:
                seasons = pd.Series(np.arange(t) % 12, dtype="category")
                dummies = pd.get_dummies(seasons, drop_first=True)
                for col in dummies:
                    rhs[f"gamma_{col}"] = dummies[col]
            else:
                s = np.sin(2*np.pi * np.arange(1, tph+1) / m)
                c = np.cos(2*np.pi * np.arange(1, tph+1) / m)
                rhs["s"] = pd.Series(s)
                rhs["c"] = pd.Series(c)
            I_6 = m != 24 and t > 3*m and (tph - len(rhs)) > 10
            if I_6:
                dt = np.ones_like(yhat_avg)
                dt = 1.0 * (np.arange(1, tph+1) >= t - 0.5 * min(4 * m, tph))
                rhs["dt"] = pd.Series(dt)
                if I_5:
                    rhs["tdt"] = pd.Series(np.arange(1, tph+1) * dt)
            rhs = pd.DataFrame(rhs)
            joined = pd.concat([yhat_avg, rhs], axis=1)
            joined = joined.dropna()
            lhs, rhs = joined.iloc[:,0], joined.iloc[:,1:]
            res = OLS(lhs, rhs).fit()

            result = res.fittedvalues
            result.index -= 13
            self.forecasts[series].loc[:,"CARD Method"] = result
        self.forecasts_generated = True

