import warnings
from skltest.utils.utils import read_data
from skltest.skl_regressor.skl_regressor_test import SklRegressorTest

# Read data file
df, input_data, output_data = read_data("../datasets/regression_dataset.csv")

# Create SklRegressorTest
SklRegressors = SklRegressorTest(m_input=input_data,
                                 m_output=output_data,
                                 m_train_percentage=0.8)

# Hidde warnings to run code faster - Not recommended
warnings.filterwarnings("ignore")

# Define Scikit-Learn regression models to run and initialization
models = "all"
SklRegressors.set_desired_models(models=models)

# Analyze the best random state with default hyperparameters
SklRegressors.test_random_states(n_random_states=10,
                                 desired_metric="root_mean_squared_error",
                                 verbose=False)

# Boost hyperparameters through a k-fold cross-validation holdout
SklRegressors.initialize_parameters()
SklRegressors.test_spaces(n_random_states=2,
                          rkf_cv_n_splits=2,
                          rkf_cv_n_repeats=1,
                          n_rand_iter=1,
                          desired_metric="root_mean_squared_error",
                          verbose=False)

# Analyze the best performance between  test_random_states and test_spaces methods at once
SklRegressors.initialize_parameters()
SklRegressors.test_all(n_random_states=2,
                       rkf_cv_n_splits=2,
                       rkf_cv_n_repeats=1,
                       n_rand_iter=1,
                       desired_metric="root_mean_squared_error",
                       verbose=False)

# Resample train/test until the error get lower than maxerror
SklRegressors.initialize_parameters()
SklRegressors.test_random_states_until(maxerror=2.0,
                                       n_iter=2,
                                       desired_metric="root_mean_squared_error",
                                       verbose=False)

# Resample train/test and boost model hyperparameters until get lower than maxerror
SklRegressors.initialize_parameters()
SklRegressors.test_spaces_until(rkf_cv_n_splits=2,
                                rkf_cv_n_repeats=1,
                                n_rand_iter=2,
                                maxerror=2.0,
                                n_iter=2,
                                desired_metric="root_mean_squared_error",
                                verbose=False)

# Analyze the best performance between  test_random_states_until and test_spaces_until methods at once
SklRegressors.initialize_parameters()
SklRegressors.test_all_until(rkf_cv_n_splits=2,
                             rkf_cv_n_repeats=1,
                             n_rand_iter=2,
                             maxerror=2.0,
                             n_iter=2,
                             desired_metric="root_mean_squared_error",
                             verbose=False)

# Save a summary file with the best hyperparameters configuration and statistical data from the models
summary_df = SklRegressors.write_log(path="",
                                     filename="skl_regressor_test_summary")

# Print all methods and metrics
SklRegressors.summary()
