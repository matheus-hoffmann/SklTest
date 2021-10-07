# Skl Regressor Test
Python library to compare more than 30 regression models available in Scikit-learn at once. It is possible to evaluate the influence of successive resampling and optimize the hyperparameters through K-fold cross-validation holdout.

## Instalation
This installation tutorial will guide you through running a local application using the conda environment and the PyCharm as IDE. First, download the full ![repository](https://github.com/matheus-hoffmann/skl_regressor_test) as a ZIP file and extract to a folder named **skl_regressor_test**.

After that, open a conda terminal e follow these steps:
1. Create your own virtual environment with the correct python version:
 
```bash
conda create -n skl_regressor_test python=3.8
```

2. Activate your virtual environment in order to work in this safe environment:

```bash
conda activate skl_regressor_test
```

3. Navigate to the `setup.py` folder in your terminal:

```bash
cd [PATH]/skl_regressor_test
```

4. Install the library:

```bash
pip install -e .
```

## Configuring PyCharm
In the same conda command window, get the path where the lib got installed:

```bash
conda env list
```

Search for the `skl_regressor_test` env in the list, and copy the full path.

Open PyCharm IDE follow this steps: `File -> Settings -> Project -> Project Interpreter ou Python Interpreter`.

After that, select the virtual environment name (`skl_regressor_test`) and apply.

## First steps
Read your csv data file:

```python
df, input_data, output_data = read_data("80_19_1.csv")
```

Creating a SklRegressorTest object:

```python
SklRegressors = SklRegressorTest(m_input=input_data,
                                 m_output=output_data,
                                 m_train_percentage=0.8)
```

Set the Scikit-Learn regressor models you want to test:

```python
SklRegressors.set_desired_models(models="all")
```

Get the best `random_state` for each model in a given interval:

```python
SklRegressors.test_random_states(n_random_states=10,
                                 desired_metric="root_mean_squared_error")
```

Clear the results in order to try to perform another method:

```python
SklRegressors.initialize_parameters()
```

Get the best combination of hyperparameters through a k-fold cross-validation holdout in a given interval:

```python
SklRegressors.test_spaces(n_random_states=10,
                          rkf_cv_n_splits=5,
                          rkf_cv_n_repeats=10,
                          n_rand_iter=20,
                          desired_metric="root_mean_squared_error")
```

Get the best model configuration after try the two method above:

```python
SklRegressors.test_all(n_random_states=10,
                       rkf_cv_n_splits=5,
                       rkf_cv_n_repeats=10,
                       n_rand_iter=20,
                       desired_metric="root_mean_squared_error")
```

Try to achieve a maximum absolute error (`maxerror`) lower than an estimated value before a given number of iterations `n_iter` just resampling the train/test data:

```python
SklRegressors.test_random_states_until(maxerror=1.0,
                                       n_iter=1000,
                                       desired_metric="root_mean_squared_error")
```


Try to achieve a maximum absolute error (`maxerror`) lower than an estimated value before a given number of iterations `n_iter` resampling the train/test data and performing the k-fold cross-validation holdout:

```python
SklRegressors.test_spaces_until(rkf_cv_n_splits=5,
                                rkf_cv_n_repeats=10,
                                n_rand_iter=20,
                                maxerror=1.0,
                                n_iter=100,
                                desired_metric="root_mean_squared_error")
```

Get the best model configuration after try the two method above:

```python
SklRegressors.test_all_until(rkf_cv_n_splits=5,
                             rkf_cv_n_repeats=10,
                             n_rand_iter=20,
                             maxerror=1.0,
                             n_iter=100,
                             desired_metric="root_mean_squared_error")
```

Write a summary file with the best configuration of hyperparameters and statistical data from this best model:

```python
summary_df = SklRegressors.write_log(path="",
                                     filename="skl_regressor_test_summary")
```

Print the calculated methods and their respective R² and Maximum Absolute Error:

```python
SklRegressors.summary()
```

Your output will be similar to this:

```python
---------------------------------------------------------------------------------------------------------------------
|             Model              | Max. Error |   R2    | Adj. R2 |   MAE   |  RMAE   |   MSE   |  RMSE   |  MAPE   |
---------------------------------------------------------------------------------------------------------------------
| XGBRegressor                   |      1.552 |    1.00 |    1.00 |    0.85 |    0.92 |    0.88 |    0.94 |    0.40 |
| GradientBoostingRegressor      |      1.783 |    1.00 |    1.00 |    0.80 |    0.90 |    0.89 |    0.94 |    0.15 |
| ExtraTreesRegressor            |      2.343 |    1.00 |    1.00 |    0.95 |    0.98 |    1.39 |    1.18 |    0.17 |
| ExtraTreeRegressor             |      2.422 |    1.00 |    1.00 |    1.13 |    1.06 |    2.04 |    1.43 |    0.24 |
| RandomForestRegressor          |      2.799 |    1.00 |    1.00 |    1.40 |    1.18 |    2.55 |    1.60 |    0.10 |
| BaggingRegressor               |      3.075 |    1.00 |    1.00 |    1.17 |    1.08 |    2.26 |    1.50 |    0.16 |
| DecisionTreeRegressor          |      3.212 |    1.00 |    1.00 |    1.27 |    1.13 |    2.90 |    1.70 |    0.22 |
| AdaBoostRegressor              |      3.754 |    1.00 |    1.00 |    1.96 |    1.40 |    5.07 |    2.25 |    0.10 |
| LassoCV                        |      7.041 |    0.99 |    0.99 |    4.57 |    2.14 |   25.00 |    5.00 |    2.07 |
| OrthogonalMatchingPursuitCV    |      7.088 |    0.99 |    0.99 |    4.46 |    2.11 |   24.32 |    4.93 |    2.03 |
| Ridge                          |      7.151 |    0.99 |    0.98 |    5.09 |    2.26 |   28.91 |    5.38 |    1.30 |
| SGDRegressor                   |      7.336 |    0.99 |    0.99 |    4.64 |    2.16 |   25.80 |    5.08 |    2.19 |
| HuberRegressor                 |      7.369 |    0.99 |    0.99 |    4.65 |    2.16 |   26.16 |    5.11 |    2.20 |
| RidgeCV                        |      7.375 |    0.99 |    0.99 |    4.66 |    2.16 |   25.84 |    5.08 |    2.20 |
| BayesianRidge                  |      7.418 |    0.99 |    0.99 |    4.67 |    2.16 |   26.08 |    5.11 |    2.21 |
| Lars                           |      7.443 |    0.99 |    0.99 |    4.68 |    2.16 |   26.22 |    5.12 |    2.22 |
| LassoLarsCV                    |      7.443 |    0.99 |    0.99 |    4.68 |    2.16 |   26.22 |    5.12 |    2.22 |
| LarsCV                         |      7.443 |    0.99 |    0.99 |    4.68 |    2.16 |   26.22 |    5.12 |    2.22 |
| LinearRegression               |      7.443 |    0.99 |    0.99 |    4.68 |    2.16 |   26.22 |    5.12 |    2.22 |
| RANSACRegressor                |      7.443 |    0.99 |    0.99 |    4.68 |    2.16 |   26.22 |    5.12 |    2.22 |
| Lasso                          |      7.620 |    0.99 |    0.99 |    4.68 |    2.16 |   25.52 |    5.05 |    1.40 |
| ElasticNetCV                   |      8.007 |    0.99 |    0.98 |    5.03 |    2.24 |   28.13 |    5.30 |    1.20 |
| LassoLarsIC                    |      8.231 |    0.99 |    0.98 |    4.83 |    2.20 |   29.24 |    5.41 |    1.11 |
| PassiveAggressiveRegressor     |      9.769 |    0.99 |    0.99 |    5.08 |    2.25 |   35.22 |    5.93 |    0.38 |
| LassoLars                      |     15.637 |    0.97 |    0.95 |    7.12 |    2.67 |   65.97 |    8.12 |    0.92 |
| OrthogonalMatchingPursuit      |     15.926 |    0.96 |    0.95 |   10.22 |    3.20 |  119.10 |   10.91 |    1.85 |
| KNeighborsRegressor            |     18.435 |    0.94 |    0.93 |   11.44 |    3.38 |  158.97 |   12.61 |    0.67 |
| LinearSVR                      |     25.860 |    0.89 |    0.85 |   12.34 |    3.51 |  217.04 |   14.73 |    0.90 |
| ElasticNet                     |     30.823 |    0.82 |    0.76 |   15.18 |    3.90 |  341.11 |   18.47 |    0.89 |
| SVR                            |     64.718 |    0.13 |   -0.16 |   32.01 |    5.66 | 1660.40 |   40.75 |    1.09 |
| NuSVR                          |     70.428 |    0.10 |   -0.20 |   33.68 |    5.80 | 1722.84 |   41.51 |    1.85 |
| DummyRegressor                 |     74.721 |   -0.01 |   -0.35 |   35.97 |    6.00 | 1925.02 |   43.88 |    2.21 |
---------------------------------------------------------------------------------------------------------------------
```
