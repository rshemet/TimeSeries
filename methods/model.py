import numpy as np
import pandas as pd
import statsmodels.stats.api as TESTS
# Evaluation methods
import statsmodels.tools.eval_measures as METRICS
from matplotlib import pyplot as plt


class model(object):
    def __init__(self, data, args=None, test_num=0):
        """Overall parent class for all of the forecasting objects. THis will
        define all of the common functions to be used and naming conventions.
        Each subclass inherits all of the parents methods and attributes but
        can then overwrite methods such as fit() to its specific needs.

        Parameters
        ----------
        @Parameters data        -   (required) pd.DataFrame of the data series,
                                    each column is a new series. Not all series
                                    need to have the same length
        @Parameters args        -   (default None) specific arguments for the
                                    submethods in the forecast method
        @Parameters test_num    -   (default 0) int of number of observations
                                    at the end of series to use for testing
                                    (e.g. KFold cross-validation)

        Returns
        ----------
        @Returns forecaster     -   generic forecaster object
        """

        # Generic saving activity
        self.data = data
        self.args = args
        self.test_num = test_num

        # Find the index where the training data stops for each series
        self.train_ix = {}
        self.count = {}
        for series in self.data.columns:
            self.count[series] = self.data.loc[:, series].count()
            self.train_ix[series] = self.count[series] - self.test_num

    def fit(self, args=None):
        """ Fit method for all functions should have the same outcomes.

        Parameters
        ----------
        @Parameters args    -   (default None) dictionary of arguments
                                that are required for the fitting method

        Returns
        ----------
        None
        """

        # REQUIRED OUTPUTS
        # Save the fitted outputs e.g. objects, arrays
        self.fitted = {}
        # Set fit_success for future checking
        self.fit_success = True

    def forecast(self, true_vals=None):
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
        if true_vals is not None:
            assert self.data.shape[1] == true_vals.shape[1], "Dimension mismatch"

        ### REQUIRED OUTPUTS ###
        # Dictionary with pd.DataFrame of forecast values for each series.
        # Can have multiple columns per series, must have at least one column
        # for the true values, with column_name == series name
        # Contains ONLY forecasts, not existing series with forecasts appended
        self.forecasts = {}

        self.fcast_true = true_vals
        self.forecasts_generated = True

    def forecast_analysis(self, stats=None, tests=None, subset=None):
        """Function to evaluate the forecasts based on the MSPE criteria

        Parameters
        ----------
        @Parameter stats        -   (default None) list of strings of the
                                    name of criteria to include in the
                                    evaluation e.g. ['mspe','jb_stat','mae']
        @Parameters subset      -   list (default=None) of the column names for
                                    series that should be analysed
        @Parameters save_summary-   (default None) optional path of file to
                                    save a summary of the analysis to

        Returns
        ----------
        @Returns analysis       -   dict of pd.DataFrames with the analysis for
                                    each of the models
        """

        assert self.forecasts_generated == True, "Require forecasts first"

        # Test if subset selection is empty, if so use all series
        subset = subset if subset is not None else list(self.forecasts.keys())

        # All possible stats to look at (have the same input structure)
        tot_stats = {
            "mse": METRICS.mse,
            "rmse": METRICS.rmse,
            "maxabs": METRICS.maxabs,
            "mae": METRICS.meanabs,
            "medianabs": METRICS.medianabs,
            "bias": METRICS.bias,
            "medianbias": METRICS.medianbias,
            "vare": METRICS.vare,
            "stde": METRICS.stde
        }

        # All possible tests to consider
        tot_tests = {
            "jb": TESTS.jarque_bera,
            "acorr_lm": TESTS.acorr_lm,
            "het_arch": TESTS.het_arch
        }

        m4 = {

        }

        # Check if none, if so use all available stats and tests
        stats = stats if stats is not None else list(tot_stats.keys())
        tests = tests if tests is not None else list(tot_tests.keys())
        m4 = list(m4.keys())

        analysis = {}
        for series in subset:
            # Eliminate the "true value" column from evaluation
            models = self.forecasts[series].columns.to_list()
            models.remove(series)

            # Specify that we want stats and pvals for each test
            ix_test = [[test + "_stat", test + "_pval"] for test in tests]
            ix_test = [i for j in ix_test for i in j]

            # Results frame
            res = pd.DataFrame(index=stats + ix_test, columns=models)

            # Need to have the true values as an array to simplify calculations
            trueval = self.forecasts[series].loc[:, series]
            trueval_array = np.hstack(len(models) * [trueval.values[:, None]])

            # Calculate the stats
            for stat in stats:
                res.loc[stat, :] = tot_stats[stat](trueval_array,
                                                   self.forecasts[series].loc[:, models], axis=0)

            residuals = self.forecasts[series].loc[:, models].copy()
            residuals = residuals.subtract(trueval, axis='index')

            for test in tests:
                for m in models:
                    temp = tot_tests[test](residuals.loc[:, m])
                    res.loc[test + "_stat", m] = temp[0]
                    res.loc[test + "_pval", m] = temp[1]

            # Save the work
            analysis[series] = res

        return analysis

    def model_eval(self, stats=None, tests=None, subset=None):
        """Function to evaluate the forecasts based on the MSPE criteria

        Parameters
        ----------
        @Parameter stats        -   (default None) list of strings of the
                                    name of criteria to include in the
                                    evaluation e.g. ['mspe','jb_stat','mae']
        @Parameters subset      -   list (default=None) of the column names for
                                    series that should be analysed
        @Parameters save_summary-   (default None) optional path of file to
                                    save a summary of the analysis to

        Returns
        ----------
        @Returns analysis       -   dict of pd.DataFrames with the analysis for
                                    each of the models
        """

        assert self.forecasts_generated == True, "Require forecasts first"

        # Test if subset selection is empty, if so use all series
        subset = subset if subset is not None else list(self.forecasts.keys())

        # All possible stats to look at (have the same input structure)
        tot_stats = {
            "MSE": METRICS.mse,
            "RMSE": METRICS.rmse,
            "Max Abs": METRICS.maxabs,
            "MAE": METRICS.meanabs,
            "Median Abs": METRICS.medianabs,
            "Bias": METRICS.bias,
            "Median Bias": METRICS.medianbias,
            "Var(e)": METRICS.vare,
            "Std(e)": METRICS.stde
        }

        # All possible tests to consider
        tot_tests = {
            "jb": TESTS.jarque_bera,
            "acorr_lm": TESTS.acorr_lm,
            "het_arch": TESTS.het_arch
        }

        def sMAPE(trueval, fcast, *args):
            num = np.abs(trueval - fcast)
            denom = np.abs(trueval) + np.abs(fcast)
            return 100 * 2 / len(fcast) * np.sum(num / denom)

        def MASE(trueval, fcast, series,m=12):
            in_samp = self.data.loc[:, series].dropna()
            in_samp_diff = in_samp.diff(periods=m).dropna()
            numerator = np.sum(np.abs(trueval - fcast))
            denominator = ((1 / (len(in_samp) - m)) * np.sum(np.abs(in_samp_diff)))
            return 1 / len(fcast) * numerator / denominator

        m4_tests = {
            "sMAPE": sMAPE,
            "MASE": MASE,
        }

        # Check if none, if so use all available stats and tests
        stats = stats if stats is not None else list(tot_stats.keys())
        tests = tests if tests is not None else list(tot_tests.keys())
        m4 = list(m4_tests.keys())

        models = self.forecasts[subset[0]].columns.to_list()
        models.remove(subset[0])
        empty_df = pd.DataFrame(index=subset, columns=stats + tests + m4)
        res = {i: empty_df.copy(deep=True) for i in models}

        for s in subset:
            trueval = self.forecasts[s].loc[:, s]
            for m in res.keys():
                fcast = self.forecasts[s].loc[:, m]
                residuals = fcast.subtract(trueval, axis='index')

                for stat in stats:
                    res[m].loc[s, stat] = tot_stats[stat](trueval, fcast)

                for test in tests:
                    # Save the p-values of the tests in question
                    temp = tot_tests[test](residuals)
                    res[m].loc[s, test] = temp[1]

                for stat in m4:
                    res[m].loc[s, stat] = m4_tests[stat](trueval, fcast, s)

        return res

    def plotForecasts(self, subset=None, subplots=False, block=False, save=None):
        """Function to plot the individual forecast series

        Parameters
        ----------
        @Parameters subset  -   list (default=None) of the column names for
                                series that should be plotted
        @Parameters subplots-   Boolean (default=False) of whether to plot
                                each series in an individual subplot
        @Parameters block   -   Boolean (defualt=False) whether to stop
                                code to show figure
        @Parameters save    -   String (default None) for filename to save the
                                image to

        Returns
        ----------
        None
        """

        # Test if subset selection is empty, if so plot all series
        subset = subset if subset is not None else self.data.columns

        # Check whether sufficient subsets for subplots
        if subplots: assert len(subset) > 1, "For subplots more than 1 series needed"

        # Dimensions of the desired graph
        rows = len(subset) if subplots else 1

        # Check whether there have been forecasts generated
        assert self.forecasts_generated == True, "Forecasts necessary to plot"

        fig, ax = plt.subplots(rows, 1)

        for i, series in enumerate(subset):
            # Add the true values and legends
            true_val = self.forecasts[series].loc[:, series]
            if subplots:
                ax[i].plot(np.arange(true_val.shape[0]), true_val, label=series, color='black')
            else:
                label = str(series) + " True Values"
                ax.plot(np.arange(true_val.shape[0]), true_val, label=label, color='black')

            # Go through the individual models
            model_list = self.forecasts[series].columns.to_list()
            model_list.remove(series)
            for model in model_list:
                f_casts = self.forecasts[series].loc[:, model]
                # Plot the models and add legends
                if subplots:
                    ax[i].plot(np.arange(f_casts.shape[0]), f_casts, label=model, linestyle='--')
                    ax[i].legend(loc="upper left")
                    ax[i].set_xticks(np.arange(f_casts.shape[0]))
                else:
                    label = str(series) + " " + str(model)
                    ax.plot(np.arange(f_casts.shape[0]), f_casts, label=label, linestyle='--')
                    ax.legend(loc="upper left")
                    ax.set_xticks(np.arange(f_casts.shape[0]))

        plt.tight_layout()
        if save is not None:
            fig.set_size_inches(12, 12)
            plt.savefig(fname=save)
        plt.show(block=block)