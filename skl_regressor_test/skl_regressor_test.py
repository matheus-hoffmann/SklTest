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
    def __init__(self, m_input:np=None, m_output:np=None, m_train_percentage:float=0.8):
        self.m_input = m_input
        self.m_output = m_output
        self.m_train_percentage = m_train_percentage

    def set_desired_models(self, models:str="all") -> None:
        if models == "all":
            self.m_models = _MODELS
            self.m_spaces = _SPACES
        else:
            try:
                self.m_models = {models: _MODELS[models]}
                self.m_spaces = {models: _SPACES[models]}
            except:
                exit("Unexpected model: must be Scikit-Learn regression model name or all.")
        self.set_dependent_parameters()

    def set_dependent_parameters(self) -> None:
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

    def test_all(self, method:str="all", n_random_states:int=10, rkf_cv_n_splits:int=5, rkf_cv_n_repeats:int=10, n_rand_iter:int=10) -> None:
        if method == "all":
            self.test_random_states(n_random_states=n_random_states)
            self.test_spaces(n_random_states=n_random_states,
                             rkf_cv_n_splits=rkf_cv_n_splits,
                             rkf_cv_n_repeats=rkf_cv_n_repeats,
                             n_rand_iter=n_rand_iter)
        elif method == "random_states":
            self.test_random_states(n_random_states=n_random_states)
        elif method == "spaces":
            self.test_spaces(n_random_states=n_random_states,
                             rkf_cv_n_splits=rkf_cv_n_splits,
                             rkf_cv_n_repeats=rkf_cv_n_repeats,
                             n_rand_iter=n_rand_iter)
        else:
            exit("Error: Unexpected method. Must be: \"all\", \"random_states\" or \"spaces\"")

    def write_log(self, path:str="", filename:str="skl_regressor_test_summary") -> None:
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
        df.to_excel(path+filename+".xlsx", index=False)
        print(df.head())