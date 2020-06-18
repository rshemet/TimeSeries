import numpy as np
import pandas as pd
from scipy import stats

# Imports for the individual submodels
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import acf

from methods.model import model

class rhoMethod(model):
    def __init__(self,data,test_num=0):
        """ Class for the Rho method

        Parameters
        ----------
        @Parameters data        -   (required) pd.DataFrame of the data series,
                                    each column is a new series. Not all series
                                    need to have the same length
        @Parameters test_num    -   (default 0) int of number of observations
                                    at the end of series to use for testing

        Returns
        ----------
        @Returns rho   -   rho forecasting object
        """

        # Initiate parent class and inherit all attributes and methods
        super().__init__(data=data,args=None,test_num=test_num)

    def fit(self):
        """Fit rho model

        Parameters
        ----------

        Returns
        ----------
        None
        """
        self.fittedLhs = {}
        self.fittedRhs = {}
        self.fittedYj = {}
        for i,series in enumerate(self.data.columns):
            x = self.data.loc[:self.train_ix[series]-1,series]
            I_ln = (x > 1).all()
            y = x.copy()
            y = np.log(x) if I_ln else x
            I_rho = y.diff().var() < (1.2 * y.var())
            z = y.copy()
            if(I_rho):
                z.loc[:] = y.loc[:].diff()
            m = 12
            def seasonal_matrix(arr):
                arr.dropna()
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

            def r(arr):
                arr = arr.dropna()
                rhos = acf(arr, nlags=12, fft=True)
                test = arr.shape[0] * rhos[-1] ** 2 / (1 + (rhos[1:-1]**2).sum())
                return test > stats.chi2(1).ppf(0.9)
            yj = y
            I_R = r(z)
            I_ar = I_rho
            I_UR = I_tr = False
            rhs = {"mu": pd.Series(np.ones_like(yj), index=yj.index)}
            if I_ar:
                rhs["rho"] = yj.shift(1)
                if I_R:
                    rhs["rho_m"] = yj.shift(12)
            if I_tr:
                rhs["trend"] = np.arange(1, yj.shape[0] + 1) / m
            if I_A:
                d = np.arange(yj.shape[0], dtype="int")
                d = d - 12 * (d // 12)
                d = [f"s_{v}" for v in d]
                d = pd.Series(d, dtype="category")
                dummies =  pd.get_dummies(d, drop_first=True)
                for col in dummies:
                    rhs[col] = dummies[col]
            rhs = pd.DataFrame(rhs)

            joined = pd.concat([yj, rhs],axis=1).dropna()
            lhs, rhs = joined.iloc[:,0],  joined.iloc[:,1:]

            self.fittedLhs[series] = lhs
            self.fittedRhs[series] = rhs
            self.fittedYj[series] = yj

        self.fit_success = True

    def forecast(self,true_vals,m=12):
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
            cols = [series] + ["Rho Method"]
            self.forecasts[series] = pd.DataFrame(index=np.arange(steps),columns=cols)
            #Currently - using log of true values and also output log of forecasts -
            #not sure but maybe we can change to raw true values and raw forecasts.
            self.forecasts[series].loc[:,series] = true_vals.loc[:,series]
            #self.forecasts[series].loc[:,series] = true_vals.loc[:,series]

            I_UR = I_tr = False
            x = self.data.loc[:,series].dropna()
            I_ln = (x > 1).all()

            lhs = self.fittedLhs[series]
            rhs = self.fittedRhs[series]
            res = OLS(lhs, rhs).fit()
            yj = self.fittedYj[series]

            try: rho = res.params["rho"]
            except KeyError: rho = 0

            if rho > 0.5 and (rho + 2 * res.bse["rho"]) > 0.9:
                lhs = yj.diff()
                I_UR = True
                rhs = rhs.drop("rho", axis=1)
                joined = pd.concat([lhs, rhs], axis=1).dropna()
                lhs, rhs = joined.iloc[:,0],  joined.iloc[:,1:]

                res = OLS(lhs, rhs).fit()
            elif rho < 0:
                lhs = yj
                rhs = rhs.drop("rho", axis=1)
                if "rho_m" in rhs:
                    rhs = rhs.drop("rho_m", axis=1)
                joined = pd.concat([lhs, rhs], axis=1).dropna()
                lhs, rhs = joined.iloc[:,0],  joined.iloc[:,1:]
                res = OLS(lhs, rhs).fit()

            eps = res.resid
            t = eps.shape[0]

            if I_UR and (t - res.df_model) > 10:
                trend = t - np.arange(t)
                stat = OLS(eps, trend).fit().tvalues[0]
                pval = 2 * (1 - stats.t(t-1).cdf(np.abs(stat)))
                if pval < .01:
                    I_tr = True
                    rhs["trend"] = np.arange(1, lhs.shape[0] + 1) / m
                    res = OLS(lhs, rhs).fit()

            if I_tr and "rho" in res.params and res.params.rho < -0.5:
                rhs = rhs.drop("trend", axis=1)
                res = OLS(lhs, rhs).fit()

            mu = res.params["mu"]
            mu_se = res.bse["mu"]
            sigma = np.sqrt(res.mse_resid)
            t = lhs.shape[0]
            mu_tilde = max(0, mu - 1.645 * sigma / np.sqrt(t - 1))
            yj = np.exp(yj) if I_ln else yj
            y_hat = yj.iloc[-1] + np.arange(1, steps+1) * mu_tilde
            combined = pd.Series(list(yj) + list(y_hat))

            y_rho = combined

            self.forecasts[series].loc[:, "Rho Method"] = y_hat

        self.forecasts_generated = True

