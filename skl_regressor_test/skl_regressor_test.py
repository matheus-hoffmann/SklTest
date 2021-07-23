from typing import Union
import pandas as pd
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.linear_model
import sklearn.svm
import sklearn.tree
import xgboost

from sklearn.metrics import r2_score, max_error, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV, RandomizedSearchCV
from skl_regressor_test.utils import *


# The keys are the method name according to Scikit-Learn documentation
_MODELS = {"SVR": sklearn.svm.SVR(),
           "BaggingRegressor": sklearn.ensemble.BaggingRegressor(),
           "NuSVR": sklearn.svm.NuSVR(),
           "RandomForestRegressor": sklearn.ensemble.RandomForestRegressor(),
           "XGBRegressor": xgboost.XGBRegressor(),
           "GradientBoostingRegressor": sklearn.ensemble.GradientBoostingRegressor(),
           "ExtraTreesRegressor": sklearn.ensemble.ExtraTreesRegressor(),
           "AdaBoostRegressor": sklearn.ensemble.AdaBoostRegressor(),
           "KNeighborsRegressor": sklearn.neighbors.KNeighborsRegressor(),
           "DecisionTreeRegressor": sklearn.tree.DecisionTreeRegressor(),
           "HuberRegressor": sklearn.linear_model.HuberRegressor(),
           "LinearSVR": sklearn.svm.LinearSVR(),
           "RidgeCV": sklearn.linear_model.RidgeCV(),
           "BayesianRidge": sklearn.linear_model.BayesianRidge(),
           "Ridge": sklearn.linear_model.Ridge(),
           "LinearRegression": sklearn.linear_model.LinearRegression(),
           "ElasticNetCV": sklearn.linear_model.ElasticNetCV(),
           "LassoCV": sklearn.linear_model.LassoCV(),
           "LassoLarsIC": sklearn.linear_model.LassoLarsIC(),
           "LassoLarsCV": sklearn.linear_model.LassoLarsCV(),
           "Lars": sklearn.linear_model.Lars(),
           "LarsCV": sklearn.linear_model.LarsCV(),
           "SGDRegressor": sklearn.linear_model.SGDRegressor(),
           "ElasticNet": sklearn.linear_model.ElasticNet(),
           "Lasso": sklearn.linear_model.Lasso(),
           "RANSACRegressor": sklearn.linear_model.RANSACRegressor(),
           "OrthogonalMatchingPursuitCV": sklearn.linear_model.OrthogonalMatchingPursuitCV(),
           "PassiveAggressiveRegressor": sklearn.linear_model.PassiveAggressiveRegressor(),
           "OrthogonalMatchingPursuit": sklearn.linear_model.OrthogonalMatchingPursuit(),
           "ExtraTreeRegressor": sklearn.tree.ExtraTreeRegressor(),
           "DummyRegressor": sklearn.dummy.DummyRegressor(),
           "LassoLars": sklearn.linear_model.LassoLars()}
_SPACES = {"SVR": {'kernel': ['linear', 'rbf', 'sigmoid'],
                      'gamma': [0.00001 * pow(10, x) for x in range(10)],
                      'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 20000, 30000, 40000, 50000,
                            100000],
                      'epsilon': [0.01 * pow(10, x) for x in range(3)]},
              "BaggingRegressor": {'n_estimators': [10, 50, 100, 500, 500, 1000, 5000],
                                   'max_samples': np.arange(0.1, 1.1, 0.1)},
              "NuSVR": {'nu': [0.01, 0.05, 0.1, 0.5, 1],
                        'kernel': ['linear', 'rbf', 'sigmoid'],
                        'gamma': [0.00001 * pow(10, x) for x in range(10)],
                        'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 20000, 30000, 40000,
                              50000,
                              100000]},
              "RandomForestRegressor": {'n_estimators': [2, 20, 100, 200, 2000],
                                        'criterion': ['mse', 'mae'],
                                        'min_samples_split': [int(x) for x in np.linspace(start=2, stop=40, num=5)],
                                        'min_samples_leaf': [int(x) for x in np.linspace(start=1, stop=20, num=5)],
                                        'max_features': ['auto', 'sqrt', 'log2']},
              "XGBRegressor": {'max_depth': range(9, 12),
                               'min_child_weight': range(5, 8),
                               'subsample': [i / 10. for i in range(7, 11)],
                               'colsample': [i / 10. for i in range(7, 11)],
                               'booster': ["gbtree", "gblinear", "dart"],
                               'learning_rate': [.3, .2, .1, .05, .01, .005]},
              "GradientBoostingRegressor": {'loss': ['ls', 'lad', 'huber', 'quantile'],
                                            'learning_rate': [1, 0.1, 0.01, 0.001],
                                            'n_estimators': [2, 20, 100, 200, 2000],
                                            'subsample': [1, 0.1, 0.01, 0.001],
                                            'criterion': ['mse', 'friedman_mse'],
                                            'min_samples_split': [int(x) for x in np.linspace(start=2, stop=40, num=5)],
                                            'min_samples_leaf': [int(x) for x in np.linspace(start=1, stop=20, num=5)],
                                            'max_features': ['auto', 'sqrt', 'log2']},
              "ExtraTreesRegressor": {'n_estimators': [2, 20, 100, 200, 2000],
                                      'criterion': ['mse', 'mae'],
                                      'min_samples_split': [int(x) for x in np.linspace(start=2, stop=40, num=5)],
                                      'min_samples_leaf': [int(x) for x in np.linspace(start=1, stop=20, num=5)],
                                      'max_features': ['auto', 'sqrt', 'log2']},
              "AdaBoostRegressor": {'n_estimators': [50, 100, 200, 500, 1000],
                                    'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
                                    'loss': ['linear', 'square', 'exponential']},
              "KNeighborsRegressor": {'n_neighbors': [2, 3, 4, 5, 6],
                                      'weights': ['uniform', 'distance'],
                                      'algorithm': ["auto", "ball_tree", "kd_tree", "brute"],
                                      'metric': ["euclidean", "manhattan", "chebyshev"]},
              "DecisionTreeRegressor": {'criterion': ['mse', 'friedman_mse', 'mae'],
                                        'min_samples_split': [int(x) for x in np.linspace(start=2, stop=40, num=5)],
                                        'min_samples_leaf': [int(x) for x in np.linspace(start=1, stop=20, num=5)],
                                        'max_features': ['auto', 'sqrt', 'log2']},
              "HuberRegressor": {"epsilon": [1, 1.35, 1.5, 2, 5, 10],
                                 "max_iter": [10, 100, 500, 1000],
                                 "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10]},
              "LinearSVR": {'loss': ["epsilon_insensitive", "squared_epsilon_insensitive"],
                            'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 20000, 30000, 40000,
                                  50000, 100000],
                            'epsilon': [0, 0.01, 0.1, 1]},
              "RidgeCV": {"fit_intercept": [True, False],
                          "normalize": [True, False],
                          "gcv_mode": ["auto", "svd", "eigen"]},
              "BayesianRidge": {'n_iter': [1, 10, 100, 300, 500, 1000, 5000, 10000, 100000],
                                'alpha_1': [100, 10, 1, 1e-3, 1e-6, 1e-9],
                                'alpha_2': [100, 10, 1, 1e-3, 1e-6, 1e-9],
                                'lambda_1': [100, 10, 1, 1e-3, 1e-6, 1e-9],
                                'lambda_2': [100, 10, 1, 1e-3, 1e-6, 1e-9],
                                'compute_score': [True, False],
                                'fit_intercept': [True, False],
                                'normalize': [True, False],
                                'copy_X': [True, False]},
              "Ridge": {"alpha": [0.1, 1, 10, 100, 500, 1000],
                        "fit_intercept": [True, False],
                        "normalize": [True, False],
                        "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]},
              "LinearRegression": {'fit_intercept': [True, False],
                                   'normalize': [True, False],
                                   'copy_X': [True, False],
                                   'positive': [True, False]},
              "ElasticNetCV": {"l1_ratio": [0, 0.25, 0.5, 0.75, 1],
                               "fit_intercept": [True, False],
                               "normalize": [True, False]},
              "LassoCV": {"fit_intercept": [True, False],
                          "normalize": [True, False]},
              "LassoLarsIC": {"criterion": ["bic", "aic"],
                              "fit_intercept": [True, False],
                              "normalize": [True, False]},
              "LassoLarsCV": {"fit_intercept": [True, False],
                              "normalize": [True, False]},
              "Lars": {"fit_intercept": [True, False],
                       "normalize": [True, False]},
              "LarsCV": {"fit_intercept": [True, False],
                         "normalize": [True, False]},
              "SGDRegressor": {"loss": ["squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
                               "penalty": ["l1", "l2", "elasticnet"],
                               "alpha": [0.005, 0.02, 0.03, 0.05, 0.06, 0.1, 1, 10, 100, 500, 1000],
                               "l1_ratio": [0, 0.25, 0.5, 0.75, 1],
                               "fit_intercept": [True, False],
                               "learning_rate": ["constant", "optimal", "invscaling", "adaptive"]},
              "ElasticNet": {"alpha": [0.005, 0.02, 0.03, 0.05, 0.06, 0.1, 1, 10, 100, 500, 1000],
                             "l1_ratio": [0, 0.25, 0.5, 0.75, 1],
                             "fit_intercept": [True, False],
                             "normalize": [True, False]},
              "Lasso": {"alpha": [0.005, 0.02, 0.03, 0.05, 0.06, 0.1, 1, 10, 100, 500, 1000],
                        "fit_intercept": [True, False],
                        "normalize": [True, False]},
              "RANSACRegressor": {"min_samples": [0, 0.1, 0.5, 0.9, 1, 5, 10, 50],
                                  "loss": ["absolute_loss", "squared_loss"]},
              "OrthogonalMatchingPursuitCV": {"fit_intercept": [True, False],
                                              "normalize": [True, False]},
              "PassiveAggressiveRegressor": {
                  'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 20000, 30000, 40000, 50000,
                        100000],
                  "fit_intercept": [True, False],
                  'epsilon': [0.01 * pow(10, x) for x in range(3)],
              },
              "OrthogonalMatchingPursuit": {"fit_intercept": [True, False],
                                            "normalize": [True, False]},
              "ExtraTreeRegressor": {'criterion': ['mse', 'friedman_mse', 'mae'],
                                     'min_samples_split': [int(x) for x in np.linspace(start=2, stop=40, num=5)],
                                     'min_samples_leaf': [int(x) for x in np.linspace(start=1, stop=20, num=5)],
                                     'max_features': ['auto', 'sqrt', 'log2']},
              "DummyRegressor": {"strategy": ["mean", "median"]
                                 },
              "LassoLars": {"alpha": [0.005, 0.02, 0.03, 0.05, 0.06, 0.1, 1, 10, 100, 500, 1000],
                            "fit_intercept": [True, False],
                            "normalize": [True, False]}
              }


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
    m_max_absolute_error : dict
        Minimum maximum absolute error achieved by the model after boosting
    m_max_error : dict
        Maximum error achieved by the model after boosting
    m_mean_absolute_error : dict
        Mean absolute error achieved by the model after boosting
    m_mean_squared_error : dict
        Maximum absolute error achieved by the model after boosting
    m_random_state : dict
        Best random state to split the train and test data

    Methods
    -------
    set_desired_models(models:str="all") -> None
        Set the Scikit-Learn model to be analyzed or all of the available models
    initialize_parameters() -> None
        Initialize attributes with the default values
    test_random_states(n_random_states:int=10) -> None
        Analyze the best random state to split the data and train with the default hyperparameters
    test_spaces(n_random_states:int=10, rkf_cv_n_splits:int=5, rkf_cv_n_repeats:int=10, n_rand_iter:int=10) -> None
        Boost hyperparameters through a k-fold cross-validation holdout
    test_all(n_random_states:int=10, rkf_cv_n_splits:int=5, rkf_cv_n_repeats:int=10, n_rand_iter:int=10) -> None
        Analyze the best performance between  test_random_states and test_spaces methods at once
    write_log(self, path:str="", filename:str="skl_regressor_test_summary") -> pd
        Save a summary file with the best hyperparameters configuration and statistical data from the models
    test_random_states_until(maxerror:float=1.0, n_iter:int=1e4) -> None
        Resample the train/test and train the model with default configuration until achieve an error lower than specified
    test_spaces_until(rkf_cv_n_splits:int=5, rkf_cv_n_repeats:int=10, n_rand_iter:int=10, maxerror:float=1.0,
    n_iter:int=1e4) -> None
        Resample the train/test and train the model boost hyperparameters configuration until achieve an error
        lower than specified
    test_all_until(rkf_cv_n_splits:int=5, rkf_cv_n_repeats:int=10, n_rand_iter:int=10, maxerror:float=1.0, 
    n_iter:int=1e4) -> None
        Analyze the best performance between  test_random_states_until and test_spaces_until methods at once
    """

    def __init__(self, m_input:np=None, m_output:np=None, m_train_percentage:float=0.8):
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

    def set_desired_models(self, models:Union[str, list]="all") -> None:
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
        self.m_max_absolute_error = self.m_models.copy()
        self.m_max_error = self.m_models.copy()
        self.m_mean_absolute_error = self.m_models.copy()
        self.m_mean_squared_error = self.m_models.copy()
        self.m_random_state = self.m_models.copy()
        for key in self.m_models:
            self.m_best_params[key] = "default"
            self.m_r2_score[key] = 1e10
            self.m_max_absolute_error[key] = 1e10
            self.m_max_error[key] = 1e10
            self.m_mean_absolute_error[key] = 1e10
            self.m_mean_squared_error[key] = 1e10
            self.m_random_state[key] = 1e10

    def test_random_states(self, n_random_states:int=10) -> None:
        """
        Analyze the best random state to split the data and train with the default hyperparameters

        Parameters
        ----------
        n_random_states : int
            Number of random states to evaluate the dafault model
        """
        random_state_interval = [int(i) for i in range(1, n_random_states+1, 1)]

        print("Testing random states")
        for random_state in random_state_interval:
            print("Random state " + str(random_state))
            xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(self.m_input,
                                                                                    self.m_output,
                                                                                    train_size=self.m_train_percentage,
                                                                                    random_state=random_state)
            for key in self.m_models.keys():
                model = self.m_models[key]
                model.fit(xtrain, ytrain)
                ypred = model.predict(xtest)
                if self.m_max_absolute_error[key] > np.nanmax(np.absolute(ytest - ypred)):
                    self.m_r2_score[key] = r2_score(ytest, ypred)
                    self.m_max_absolute_error[key] = np.nanmax(np.absolute(ytest - ypred))
                    self.m_max_error[key] = max_error(ytest, ypred)
                    self.m_mean_absolute_error[key] = mean_absolute_error(ytest, ypred)
                    self.m_mean_squared_error[key] = mean_squared_error(ytest, ypred)
                    self.m_random_state[key] = random_state

        print("Random state default analysis finished")

    def test_spaces(self, n_random_states:int=10, rkf_cv_n_splits:int=5, rkf_cv_n_repeats:int=10, n_rand_iter:int=10) -> None:
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
        """
        random_state_interval = [int(i) for i in range(1, n_random_states + 1, 1)]

        print("Testing spaces")
        for random_state in random_state_interval:
            print("Iter " + str(random_state))
            xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(self.m_input,
                                                                                    self.m_output,
                                                                                    train_size=self.m_train_percentage,
                                                                                    random_state=random_state)
            for key in self.m_models.keys():
                print(key)
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
                if self.m_max_absolute_error[key] > np.nanmax(np.absolute(ytest - ypred)):
                    self.m_r2_score[key] = r2_score(ytest, ypred)
                    self.m_max_absolute_error[key] = np.nanmax(np.absolute(ytest - ypred))
                    self.m_max_error[key] = max_error(ytest, ypred)
                    self.m_mean_absolute_error[key] = mean_absolute_error(ytest, ypred)
                    self.m_mean_squared_error[key] = mean_squared_error(ytest, ypred)
                    self.m_random_state[key] = random_state
                    self.m_best_model[key] = model
                    self.m_best_params[key] = split_dict(search.best_params_)

        print("Random state default analysis finished")

    def test_all(self, n_random_states:int=10, rkf_cv_n_splits:int=5, rkf_cv_n_repeats:int=10, n_rand_iter:int=10) -> None:
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
        """
        self.test_random_states(n_random_states=n_random_states)
        self.test_spaces(n_random_states=n_random_states,
                         rkf_cv_n_splits=rkf_cv_n_splits,
                         rkf_cv_n_repeats=rkf_cv_n_repeats,
                         n_rand_iter=n_rand_iter)

    def write_log(self, path:str="", filename:str="skl_regressor_test_summary") -> pd:
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
        df["Max. Absolute Error"] = np.array(list(self.m_max_absolute_error.values()))
        df["Max. Error"] = np.array(list(self.m_max_error.values()))
        df["Mean Absolute Error"] = np.array(list(self.m_mean_absolute_error.values()))
        df["Mean Squared Error"] = np.array(list(self.m_mean_squared_error.values()))

        df = df.sort_values(by="Model")
        if path == "" or path[-1] == "/":
            df.to_excel(path+filename+".xlsx", index=False)
        else:
            df.to_excel(path + "/" + filename + ".xlsx", index=False)
        return df

    def test_random_states_until(self, maxerror:float=1.0, n_iter:int=1e4) -> None:
        """
        Resample the train/test and train the model with default configuration until achieve an error lower than specified

        Parameters
        ----------
        maxerror:float
            Maximum absolute error alowed
        n_iter:int
            Maximum number of iterations
        """
        for key in self.m_models.keys():
            random_state = 1
            while random_state <= n_iter:
                xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(self.m_input,
                                                                                    self.m_output,
                                                                                    train_size=self.m_train_percentage,
                                                                                    random_state=random_state)
                model = self.m_best_model[key]
                model.fit(xtrain, ytrain)
                ypred = model.predict(xtest)

                if self.m_max_absolute_error[key] > np.nanmax(np.absolute(ytest - ypred)):
                    self.m_r2_score[key] = r2_score(ytest, ypred)
                    self.m_max_absolute_error[key] = np.nanmax(np.absolute(ytest - ypred))
                    self.m_max_error[key] = max_error(ytest, ypred)
                    self.m_mean_absolute_error[key] = mean_absolute_error(ytest, ypred)
                    self.m_mean_squared_error[key] = mean_squared_error(ytest, ypred)
                    self.m_random_state[key] = random_state
                    if self.m_max_absolute_error[key] <= maxerror:
                        break
                random_state += 1

            print("Model: " + key +
                  " | Iterations: " + str(random_state) +
                  " | Max. Error: " + str(self.m_max_absolute_error[key]))

    def test_spaces_until(self, rkf_cv_n_splits:int=5, rkf_cv_n_repeats:int=10, n_rand_iter:int=10,
                                 maxerror:float=1.0, n_iter:int=1e4) -> None:
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
        """
        for key in self.m_models.keys():
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
                if self.m_max_absolute_error[key] > np.nanmax(np.absolute(ytest - ypred)):
                    self.m_r2_score[key] = r2_score(ytest, ypred)
                    self.m_max_absolute_error[key] = np.nanmax(np.absolute(ytest - ypred))
                    self.m_max_error[key] = max_error(ytest, ypred)
                    self.m_mean_absolute_error[key] = mean_absolute_error(ytest, ypred)
                    self.m_mean_squared_error[key] = mean_squared_error(ytest, ypred)
                    self.m_random_state[key] = random_state
                    self.m_best_model[key] = model
                    self.m_best_params[key] = split_dict(search.best_params_)
                    if self.m_max_absolute_error[key] <= maxerror:
                        break
                random_state += 1

            print("Model: " + key +
                  " | Iterations: " + str(random_state) +
                  " | Max. Error: " + str(self.m_max_absolute_error[key]))
    
    def test_all_until(self, rkf_cv_n_splits:int=5, rkf_cv_n_repeats:int=10, n_rand_iter:int=10,maxerror:float=1.0, 
                       n_iter:int=1e4) -> None:
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
        """
        self.test_random_states_until(maxerror=maxerror,
                                      n_iter=n_iter)
        self.test_spaces_until(rkf_cv_n_splits=rkf_cv_n_splits,
                               rkf_cv_n_repeats=rkf_cv_n_repeats,
                               n_rand_iter=n_rand_iter,
                               maxerror=maxerror,
                               n_iter=n_iter)
