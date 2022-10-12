from typing import Union
import pandas as pd
import numpy as np
import pickle
import sklearn
import sklearn.ensemble
import sklearn.linear_model
import sklearn.svm
import sklearn.tree
import xgboost

from sklearn.metrics import r2_score, max_error, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV, RandomizedSearchCV

from skltest.skl_regressor.config import _MODELS, _SPACES, _METRICS
from skltest.utils.utils import check_metric, check_max_n_iter, adjusted_r2_score, split_dict


class SklRegressorTest:
    """
    A class used to represent test several Scikit-Learn regression models

    ...

    Attributes
    ----------
    m_input : np
        Data input matrix [n x m]
    m_output : np
        Data output array [n x 1]
    m_train_percentage : float
        Percentage of data to train the model
    m_models : dict
        Machine learning models to be analyzed
    m_spaces : dict
        Possible hyperparameters values to each model
    m_best_model : dict
        Best machine learning model after boosting, with already settled the best hyperparameters
    m_best_params : dict
        Best combination of hyperparameters for each model
    m_r2_score : dict
        R2 score achieved by the model after boosting
    m_adj_r2_score : dict
        Adjusted R2 score achieved by the model after boosting
    m_max_error : dict
        Maximum error achieved by the model after boosting
    m_mean_absolute_error : dict
        Mean absolute error achieved by the model after boosting
    m_root_mean_absolute_error : dict
        Root mean absolute error achieved by the model after boosting
    m_mean_squared_error : dict
        Maximum absolute error achieved by the model after boosting
    m_root_mean_squared_error : dict
        Root maximum absolute error achieved by the model after boosting
    m_mean_absolute_percentage_error : dict
        Mean absolute percentage error achieved by the model after boosting
    m_random_state : dict
        Best random state to split the train and test data

    Methods
    -------
    set_desired_models(models:str="all") -> None
        Set the Scikit-Learn model to be analyzed or all of the available models

    initialize_parameters() -> None
        Initialize attributes with the default values

    test_random_states(n_random_states:int=10, desired_metric:str="max_error", verbose:bool=False) -> None
        Analyze the best random state to split the data and train with the default hyperparameters

    test_spaces(n_random_states:int=10, rkf_cv_n_splits:int=5, rkf_cv_n_repeats:int=10, n_rand_iter:int=10,
     desired_metric:str="max_error", verbose:bool=False) -> None
        Boost hyperparameters through a k-fold cross-validation holdout

    test_all(n_random_states:int=10, rkf_cv_n_splits:int=5, rkf_cv_n_repeats:int=10, n_rand_iter:int=10,
     desired_metric:str="max_error", verbose:bool=False) -> None
        Analyze the best performance between  test_random_states and test_spaces methods at once

    write_log(self, path:str="", filename:str="skl_regressor_test_summary") -> pd
        Save a summary file with the best hyperparameters configuration and statistical data from the models
    
    test_random_states_until(maxerror:float=1.0, n_iter:int=1e4, desired_metric:str="max_error", verbose:bool=False) -> None
        Resample the train/test and train the model with default configuration until achieve an error lower than specified

    test_spaces_until(rkf_cv_n_splits:int=5, rkf_cv_n_repeats:int=10, n_rand_iter:int=10, maxerror:float=1.0,
    n_iter:int=1e4, desired_metric:str="max_error", verbose:bool=False) -> None
        Resample the train/test and train the model boost hyperparameters configuration until achieve an error
        lower than specified

    test_all_until(rkf_cv_n_splits:int=5, rkf_cv_n_repeats:int=10, n_rand_iter:int=10, maxerror:float=1.0,
    n_iter:int=1e4, desired_metric:str="max_error", verbose:bool=False) -> None
        Analyze the best performance between  test_random_states_until and test_spaces_until methods at once

    summary() -> None
        Print performance data of the the best configuration achieved

    save_models(path:str="")
        Save the final model
    """

    def __init__(self, m_input: np = None, m_output: np = None, m_train_percentage: float = 0.8):
        """
        Parameters
        ----------
        m_input : np
            Data input matrix [n x m]
        m_output : np
            Data output array [n x 1]
        m_train_percentage : float
            Percentage of data to train the model
        """
        self.m_input = m_input
        self.m_output = m_output
        self.m_train_percentage = m_train_percentage

    def set_desired_models(self, models: Union[str, list] = "all") -> None:
        """
        Set the Scikit-Learn model to be analyzed or all of the available models

        Parameters
        ----------
        models : str
            Models to be evaluated. Pass the name of the Scikit-Learn implementation or "all" if you desire to run all
            implemented methdos
        """
        if type(models) == str:
            if models == "all":
                self.m_models = _MODELS
                self.m_spaces = _SPACES
            else:
                try:
                    self.m_models = {models: _MODELS[models]}
                    self.m_spaces = {models: _SPACES[models]}
                except:
                    exit("Unexpected model: must be Scikit-Learn regression model name or all.")
        elif type(models) == list:
            self.m_models = dict()
            self.m_spaces = dict()
            for model in models:
                try:
                    self.m_models[model] = _MODELS[model]
                    self.m_spaces[model] = _SPACES[model]
                except:
                    exit("Unexpected model: must be Scikit-Learn regression model name.")
        else:
            exit("Unexpected model type. Must be string or list.")
        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        """
        Initialize attributes with the default value
        """
        self.m_best_model = self.m_models.copy()
        self.m_best_params = self.m_models.copy()
        self.m_r2_score = self.m_models.copy()
        self.m_adj_r2_score = self.m_models.copy()
        self.m_max_error = self.m_models.copy()
        self.m_mean_absolute_error = self.m_models.copy()
        self.m_root_mean_absolute_error = self.m_models.copy()
        self.m_mean_squared_error = self.m_models.copy()
        self.m_root_mean_squared_error = self.m_models.copy()
        self.m_mean_absolute_percentage_error = self.m_models.copy()
        self.m_random_state = self.m_models.copy()
        for key in self.m_models:
            self.m_best_params[key] = "default"
            self.m_r2_score[key] = 1e10
            self.m_adj_r2_score[key] = 1e10
            self.m_max_error[key] = 1e10
            self.m_mean_absolute_error[key] = 1e10
            self.m_root_mean_absolute_error[key] = 1e10
            self.m_mean_squared_error[key] = 1e10
            self.m_root_mean_squared_error[key] = 1e10
            self.m_mean_absolute_percentage_error[key] = 1e10
            self.m_random_state[key] = 1e10

    def test_random_states(self, n_random_states: int = 10, desired_metric: str = "max_error",
                           verbose: bool = False) -> None:
        """
        Analyze the best random state to split the data and train with the default hyperparameters

        Parameters
        ----------
        n_random_states : int
            Number of random states to evaluate the dafault model
        desired_metric : str
            Metric to be minimized -> ["max_error", "mean_absolute_error", "root_mean_squared_error", "mean_absolute_percentage_error"]
        verbose: bool
            Print intermediate steps.
        """
        desired_metric = check_metric(all_metrics=_METRICS,
                                      desired_metric=desired_metric,
                                      std_metric="max_error")
        random_state_interval = [int(i) for i in range(1, n_random_states + 1, 1)]

        print("\nTesting random states")
        for random_state in random_state_interval:
            if verbose:
                print("Random state " + str(random_state))
            xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(self.m_input,
                                                                                    self.m_output,
                                                                                    train_size=self.m_train_percentage,
                                                                                    random_state=random_state)
            for key in self.m_models.keys():
                model = self.m_models[key]
                model.fit(xtrain, ytrain)
                ypred = model.predict(xtest)

                metrics = {"max_error": self.m_max_error[key],
                           "mean_absolute_error": self.m_mean_absolute_error[key],
                           "root_mean_squared_error": self.m_root_mean_squared_error[key],
                           "mean_absolute_percentage_error": self.m_mean_absolute_percentage_error[key]}

                if metrics[desired_metric] > _METRICS[desired_metric](ytest, ypred):
                    self.m_r2_score[key] = r2_score(ytest, ypred)
                    self.m_adj_r2_score[key] = adjusted_r2_score(self.m_r2_score[key], xtest)
                    self.m_max_error[key] = max_error(ytest, ypred)
                    self.m_mean_absolute_error[key] = mean_absolute_error(ytest, ypred)
                    self.m_root_mean_absolute_error[key] = pow(self.m_mean_absolute_error[key], 0.5)
                    self.m_mean_squared_error[key] = mean_squared_error(ytest, ypred)
                    self.m_root_mean_squared_error[key] = pow(self.m_mean_squared_error[key], 0.5)
                    self.m_mean_absolute_percentage_error[key] = mean_absolute_percentage_error(ytest, ypred)
                    self.m_random_state[key] = random_state

        print("Random state default analysis finished")

    def test_spaces(self, n_random_states: int = 10, rkf_cv_n_splits: int = 5, rkf_cv_n_repeats: int = 10,
                    n_rand_iter: int = 10,
                    desired_metric: str = "max_error", verbose: bool = False) -> None:
        """
        Boost hyperparameters through a k-fold cross-validation holdout

        Parameters
        ----------
        n_random_states : int
            Number of random states to evaluate the dafault model
        rkf_cv_n_splits : int
            Number of splits of cross-validation
        rkf_cv_n_repeats : int
            Number of repeats of cross-validation
        n_rand_iter : int
            Number of samples of the grid of possible hyperparameters combinations
        desired_metric : str
            Metric to be minimized -> ["max_error", "mean_absolute_error", "root_mean_squared_error", "mean_absolute_percentage_error"]
        verbose: bool
            Print intermediate steps.
        """
        desired_metric = check_metric(all_metrics=_METRICS,
                                      desired_metric=desired_metric,
                                      std_metric="max_error")
        random_state_interval = [int(i) for i in range(1, n_random_states + 1, 1)]

        print("\nTesting spaces")
        for random_state in random_state_interval:
            if verbose:
                print("Iter " + str(random_state))
            xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(self.m_input,
                                                                                    self.m_output,
                                                                                    train_size=self.m_train_percentage,
                                                                                    random_state=random_state)
            for key in self.m_models.keys():
                model = self.m_models[key]
                search_n_iter = check_max_n_iter(desired_n_iter=n_rand_iter,
                                                 space=self.m_spaces[key])

                cv = RepeatedKFold(n_splits=rkf_cv_n_splits, n_repeats=rkf_cv_n_repeats, random_state=random_state)
                search = RandomizedSearchCV(estimator=model,
                                            param_distributions=self.m_spaces[key],
                                            cv=cv,
                                            n_iter=search_n_iter,
                                            random_state=random_state)
                search.fit(xtrain, ytrain)

                model = search.best_estimator_
                model.fit(xtrain, ytrain)
                ypred = model.predict(xtest)

                metrics = {"max_error": self.m_max_error[key],
                           "mean_absolute_error": self.m_mean_absolute_error[key],
                           "root_mean_squared_error": self.m_root_mean_squared_error[key],
                           "mean_absolute_percentage_error": self.m_mean_absolute_percentage_error[key]}

                if metrics[desired_metric] > _METRICS[desired_metric](ytest, ypred):
                    self.m_r2_score[key] = r2_score(ytest, ypred)
                    self.m_adj_r2_score[key] = adjusted_r2_score(self.m_r2_score[key], xtest)
                    self.m_max_error[key] = max_error(ytest, ypred)
                    self.m_mean_absolute_error[key] = mean_absolute_error(ytest, ypred)
                    self.m_root_mean_absolute_error[key] = pow(self.m_mean_absolute_error[key], 0.5)
                    self.m_mean_squared_error[key] = mean_squared_error(ytest, ypred)
                    self.m_root_mean_squared_error[key] = pow(self.m_mean_squared_error[key], 0.5)
                    self.m_mean_absolute_percentage_error[key] = mean_absolute_percentage_error(ytest, ypred)
                    self.m_random_state[key] = random_state
                    self.m_best_model[key] = model
                    self.m_best_params[key] = split_dict(search.best_params_)

        print("K-fold cross-validation holdout analysis finished")

    def test_all(self, n_random_states: int = 10, rkf_cv_n_splits: int = 5, rkf_cv_n_repeats: int = 10,
                 n_rand_iter: int = 10,
                 desired_metric: str = "max_error", verbose: bool = False) -> None:
        """
        Analyze the best performance between  test_random_states and test_spaces methods at once

        Parameters
        ----------
        n_random_states : int
            Number of random states to evaluate the dafault model
        rkf_cv_n_splits : int
            Number of splits of cross-validation
        rkf_cv_n_repeats : int
            Number of repeats of cross-validation
        n_rand_iter : int
            Number of samples of the grid of possible hyperparameters combinations
        desired_metric : str
            Metric to be minimized -> ["max_error", "mean_absolute_error", "root_mean_squared_error", "mean_absolute_percentage_error"]
        verbose: bool
            Print intermediate steps.
        """
        print("\n\nTesting all")
        self.test_random_states(n_random_states=n_random_states,
                                desired_metric=desired_metric,
                                verbose=verbose)
        self.test_spaces(n_random_states=n_random_states,
                         rkf_cv_n_splits=rkf_cv_n_splits,
                         rkf_cv_n_repeats=rkf_cv_n_repeats,
                         n_rand_iter=n_rand_iter,
                         desired_metric=desired_metric,
                         verbose=verbose)
        print("Testing all finished")

    def write_log(self, path: str = "", filename: str = "skl_regressor_test_summary") -> pd:
        """
        Analyze the best random state to split the data and train with the default hyperparameters

        Parameters
        ----------
        path : str
            Path to the file. It is assumed that it will be saved in the current directory.
        filename: str
            Name of the summary file
        
        Return
        ------
        df : pd
            Summary dataframe
        """
        df = pd.DataFrame()
        df["Model"] = self.m_models.keys()
        df["Random State"] = np.array(list(self.m_random_state.values()))
        df["Best Parameters"] = np.array(list(self.m_best_params.values()))
        df["R2 score"] = np.array(list(self.m_r2_score.values()))
        df["Adj. R2 score"] = np.array(list(self.m_adj_r2_score.values()))
        df["Max. Error"] = np.array(list(self.m_max_error.values()))
        df["Mean Absolute Error"] = np.array(list(self.m_mean_absolute_error.values()))
        df["Root Mean Absolute Error"] = np.array(list(self.m_root_mean_absolute_error.values()))
        df["Mean Squared Error"] = np.array(list(self.m_mean_squared_error.values()))
        df["Root Mean Squared Error"] = np.array(list(self.m_root_mean_squared_error.values()))
        df["Mean Absolute Percentage Error"] = np.array(list(self.m_mean_absolute_percentage_error.values()))

        df = df.sort_values(by="Model")
        if path == "" or path[-1] == "/":
            df.to_excel(path + filename + ".xlsx", index=False)
        else:
            df.to_excel(path + "/" + filename + ".xlsx", index=False)
        return df

    def test_random_states_until(self, maxerror: float = 1.0, n_iter: int = 1e4,
                                 desired_metric: str = "max_error", verbose: bool = False) -> None:
        """
        Resample the train/test and train the model with default configuration until achieve an error lower than specified

        Parameters
        ----------
        maxerror:float
            Maximum absolute error alowed
        n_iter:int
            Maximum number of iterations
        desired_metric : str
            Metric to be minimized -> ["max_error", "mean_absolute_error", "root_mean_squared_error", "mean_absolute_percentage_error"]
        verbose: bool
            Print intermediate steps.
        """
        print("\nTesting random states until {} iterations".format(n_iter))
        desired_metric = check_metric(all_metrics=_METRICS,
                                      desired_metric=desired_metric,
                                      std_metric="max_error")
        for key in self.m_models.keys():
            if verbose:
                print(key)
            random_state = 1
            while random_state <= n_iter:
                xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(self.m_input,
                                                                                        self.m_output,
                                                                                        train_size=self.m_train_percentage,
                                                                                        random_state=random_state)
                model = self.m_best_model[key]
                model.fit(xtrain, ytrain)
                ypred = model.predict(xtest)

                metrics = {"max_error": self.m_max_error[key],
                           "mean_absolute_error": self.m_mean_absolute_error[key],
                           "root_mean_squared_error": self.m_root_mean_squared_error[key],
                           "mean_absolute_percentage_error": self.m_mean_absolute_percentage_error[key]}

                if metrics[desired_metric] > _METRICS[desired_metric](ytest, ypred):
                    self.m_r2_score[key] = r2_score(ytest, ypred)
                    self.m_adj_r2_score[key] = adjusted_r2_score(self.m_r2_score[key], xtest)
                    self.m_max_error[key] = max_error(ytest, ypred)
                    self.m_mean_absolute_error[key] = mean_absolute_error(ytest, ypred)
                    self.m_root_mean_absolute_error[key] = pow(self.m_mean_absolute_error[key], 0.5)
                    self.m_mean_squared_error[key] = mean_squared_error(ytest, ypred)
                    self.m_root_mean_squared_error[key] = pow(self.m_mean_squared_error[key], 0.5)
                    self.m_mean_absolute_percentage_error[key] = mean_absolute_percentage_error(ytest, ypred)
                    self.m_random_state[key] = random_state
                    if self.m_max_error[key] <= maxerror:
                        break
                random_state += 1
            if verbose:
                print("Model: " + key +
                      " | Iterations: " + str(self.m_random_state[key]) +
                      " | Max. Error: " + str(self.m_max_error[key]))
        print("Random state default analysis until {} iterations finished".format(n_iter))

    def test_spaces_until(self, rkf_cv_n_splits: int = 5, rkf_cv_n_repeats: int = 10, n_rand_iter: int = 10,
                          maxerror: float = 1.0, n_iter: int = 1e4, desired_metric: str = "max_error",
                          verbose: bool = False) -> None:
        """
        Resample the train/test and train the model boost hyperparameters configuration until achieve an error
        lower than specified

        Parameters
        ----------
        rkf_cv_n_splits : int
            Number of splits of cross-validation
        rkf_cv_n_repeats : int
            Number of repeats of cross-validation
        n_rand_iter : int
            Number of samples of the grid of possible hyperparameters combinations
        maxerror:float
            Maximum absolute error alowed
        n_iter:int
            Maximum number of iterations
        desired_metric : str
            Metric to be minimized -> ["max_error", "mean_absolute_error", "root_mean_squared_error", "mean_absolute_percentage_error"]
        verbose: bool
            Print intermediate steps.
        """
        print("\nTesting spaces until {} iterations".format(n_iter))
        desired_metric = check_metric(all_metrics=_METRICS,
                                      desired_metric=desired_metric,
                                      std_metric="max_error")
        for key in self.m_models.keys():
            if verbose:
                print(key)
            random_state = 1
            while random_state <= n_iter:
                xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(self.m_input,
                                                                                        self.m_output,
                                                                                        train_size=self.m_train_percentage,
                                                                                        random_state=random_state)
                model = self.m_models[key]
                search_n_iter = check_max_n_iter(desired_n_iter=n_rand_iter,
                                                 space=self.m_spaces[key])

                cv = RepeatedKFold(n_splits=rkf_cv_n_splits, n_repeats=rkf_cv_n_repeats, random_state=random_state)
                search = RandomizedSearchCV(estimator=model,
                                            param_distributions=self.m_spaces[key],
                                            cv=cv,
                                            n_iter=search_n_iter,
                                            random_state=random_state)
                search.fit(xtrain, ytrain)

                model = search.best_estimator_
                model.fit(xtrain, ytrain)
                ypred = model.predict(xtest)

                metrics = {"max_error": self.m_max_error[key],
                           "mean_absolute_error": self.m_mean_absolute_error[key],
                           "root_mean_squared_error": self.m_root_mean_squared_error[key],
                           "mean_absolute_percentage_error": self.m_mean_absolute_percentage_error[key]}

                if metrics[desired_metric] > _METRICS[desired_metric](ytest, ypred):
                    self.m_r2_score[key] = r2_score(ytest, ypred)
                    self.m_adj_r2_score[key] = adjusted_r2_score(self.m_r2_score[key], xtest)
                    self.m_max_error[key] = max_error(ytest, ypred)
                    self.m_mean_absolute_error[key] = mean_absolute_error(ytest, ypred)
                    self.m_root_mean_absolute_error[key] = pow(self.m_mean_absolute_error[key], 0.5)
                    self.m_mean_squared_error[key] = mean_squared_error(ytest, ypred)
                    self.m_root_mean_squared_error[key] = pow(self.m_mean_squared_error[key], 0.5)
                    self.m_mean_absolute_percentage_error[key] = mean_absolute_percentage_error(ytest, ypred)
                    self.m_random_state[key] = random_state
                    self.m_best_model[key] = model
                    self.m_best_params[key] = split_dict(search.best_params_)
                    if self.m_max_error[key] <= maxerror:
                        break
                random_state += 1
            if verbose:
                print("Model: " + key +
                      " | Iterations: " + str(self.m_random_state[key]) +
                      " | Max. Error: " + str(self.m_max_error[key]))
        print("K-fold cross-validation holdout analysis until {} iterations finished".format(n_iter))

    def test_all_until(self, rkf_cv_n_splits: int = 5, rkf_cv_n_repeats: int = 10, n_rand_iter: int = 10,
                       maxerror: float = 1.0,
                       n_iter: int = 1e4, desired_metric: str = "max_error", verbose: bool = False) -> None:
        """
        Analyze the best performance between  test_random_states_until and test_spaces_until methods at once

        Parameters
        ----------
        rkf_cv_n_splits : int
            Number of splits of cross-validation
        rkf_cv_n_repeats : int
            Number of repeats of cross-validation
        n_rand_iter : int
            Number of samples of the grid of possible hyperparameters combinations
        maxerror:float
            Maximum absolute error alowed
        n_iter:int
            Maximum number of iterations
        desired_metric : str
            Metric to be minimized -> ["max_error", "mean_absolute_error", "root_mean_squared_error", "mean_absolute_percentage_error"]
        verbose: bool
            Print intermediate steps.
        """
        print("\n\nTesting all until {} iterations".format(n_iter))
        self.test_random_states_until(maxerror=maxerror,
                                      n_iter=n_iter,
                                      desired_metric=desired_metric,
                                      verbose=verbose)
        self.test_spaces_until(rkf_cv_n_splits=rkf_cv_n_splits,
                               rkf_cv_n_repeats=rkf_cv_n_repeats,
                               n_rand_iter=n_rand_iter,
                               maxerror=maxerror,
                               n_iter=n_iter,
                               desired_metric=desired_metric,
                               verbose=verbose)
        print("Testing all until {} iterations finished".format(n_iter))

    def summary(self) -> None:
        """
        Show the performance of the best model achieved for each method.
        """
        df = pd.DataFrame()
        df["Model"] = self.m_models.keys()
        df["Max. Error"] = np.array(list(self.m_max_error.values()))
        df["R2"] = np.array(list(self.m_r2_score.values()))
        df["Adj. R2"] = np.array(list(self.m_adj_r2_score.values()))
        df["MAE"] = np.array(list(self.m_mean_absolute_error.values()))
        df["RMAE"] = np.array(list(self.m_root_mean_absolute_error.values()))
        df["MSE"] = np.array(list(self.m_mean_squared_error.values()))
        df["RMSE"] = np.array(list(self.m_root_mean_squared_error.values()))
        df["MAPE"] = np.array(list(self.m_mean_absolute_percentage_error.values()))
        df = df.sort_values(by="Max. Error", ascending=True)

        print("\n\n----------------------------------------------------------------------------------------------"
              "-----------------------")
        print("| {:^30} | {:^10} | {:^7} | {:^7} | {:^7} | {:^7} | {:^7} | {:^7} | {:^7} |".format(df.columns[0],
                                                                                                   df.columns[1],
                                                                                                   df.columns[2],
                                                                                                   df.columns[3],
                                                                                                   df.columns[4],
                                                                                                   df.columns[5],
                                                                                                   df.columns[6],
                                                                                                   df.columns[7],
                                                                                                   df.columns[8]))
        print("--------------------------------------------------------------------------------------------------"
              "-------------------")
        for i in range(len(df)):
            print("| {:30} | {:10.3f} | {:7.2f} | {:7.2f} | {:7.2f} | {:7.2f} | {:7.2f} | {:7.2f} |"
                  " {:7.2f} |".format(df.values[i, 0], df.values[i, 1], df.values[i, 2],
                                      df.values[i, 3], df.values[i, 4], df.values[i, 5],
                                      df.values[i, 6], df.values[i, 7], df.values[i, 8]))
        print("---------------------------------------------------------------------------------------------------"
              "------------------")

    def save_models(self, path: str = "") -> None:
        """
        Save the final model

        Parameters
        ----------
        path : str
            Path to the file. It is assumed that it will be saved in the current directory.
        """
        if path == "" or path[-1] == "/":
            _path = path
        else:
            _path = path + "/"
        for key in self.m_models.keys():
            pickle.dump(self.m_best_model[key],
                        open("{}model_{}.pkl".format(_path, key), 'wb'))
