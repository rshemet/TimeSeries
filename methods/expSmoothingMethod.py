import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing as ExpSmoothing

from methods.model import model

class expSmoothing(model):
    def __init__(self,data,test_num=0):
        """ Class for the exponential smoothing models

        Parameters
        ----------
        @Parameters data        -   (required) pd.DataFrame of the data series,
                                    each column is a new series. Not all series
                                    need to have the same length
        @Parameters test_num    -   (default 0) int of number of observations
                                    at the end of series to use for testing

        Returns
        ----------
        @Returns expSmoothing   -   exponential smoothing estimation object
        """

        # Initiate parent class and inherit all attributes and methods
        super().__init__(data=data,args=None,test_num=test_num)

    def fit(self,hw_args={"trend":True,"initialization_method":"heuristic"}):
        """Function to fit all possible exponential smoothing models
        found in Kevin Sheppards jupyter notebook. Overwrites parent fit class

        Parameters
        ----------
        @Parameters hw_args -   dictionary of arguments to regulate the
                                holt-winters usage

        Returns
        ----------
        None
        """

        # Reduce the display output of minimizers and reduce memory usage
        fit_args = {"disp":False,"iprint":-1,"low_memory":True}

        # Initiate results as dictionaries of model defining parameters
        self.fitted = {
            "simple_log":{'trend':False},
            "holt":{'trend':True},
            "holt_damped":{'trend':True,'damped_trend':True},
            "hw_annual": {"seasonal": 12, "trend": True, "initialization_method": "heuristic"},
            "hw_semi":{"seasonal":6,"trend":True,"initialization_method":"heuristic"},
            "hw_quarterly":{"seasonal":3,"trend":True,"initialization_method":"heuristic"},
        }

        for mod in self.fitted:
            # Take arguments that define the model
            args = self.fitted[mod]
            # Apply model to each column i.e. series with only existing (no NaN) data
            models = self.data.apply(lambda x:ExpSmoothing(np.log(x.dropna()),**args).fit(**fit_args))
            # Convert the dataframe into a dictionary for forecasting
            self.fitted[mod] = models.to_dict()

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
            # Generate dataframe for all the individual forecasts
            cols = [series] + list(self.fitted.keys())
            self.forecasts[series] = pd.DataFrame(index=np.arange(steps),columns=cols)
            self.forecasts[series].loc[:,series] = true_vals.loc[:,series]

            # Forecast for all the models
            for mod in self.fitted.keys():
                fcast = self.fitted[mod][series].forecast(steps=steps)
                fcast.index = np.arange(steps)
                # From earlier fitting, the forecasts are in log terms
                self.forecasts[series].loc[:,mod] = np.exp(fcast)

        self.forecasts_generated = True