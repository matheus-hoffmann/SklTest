# Skl Regressor Test
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
SklRegressors.test_random_states(n_random_states=10)
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
                          n_rand_iter=20)
```

Get the best model configuration after try the two method above:

```python
SklRegressors.test_all(n_random_states=10,
                       rkf_cv_n_splits=5,
                       rkf_cv_n_repeats=10,
                       n_rand_iter=20)
```

Try to achieve a maximum absolute error (`maxerror`) lower than an estimated value before a given number of iterations `n_iter` just resampling the train/test data:

```python
SklRegressors.test_random_states_until(maxerror=1.0,
                                       n_iter=1000)
```


Try to achieve a maximum absolute error (`maxerror`) lower than an estimated value before a given number of iterations `n_iter` resampling the train/test data and performing the k-fold cross-validation holdout:

```python
SklRegressors.test_spaces_until(rkf_cv_n_splits=5,
                                rkf_cv_n_repeats=10,
                                n_rand_iter=20,
                                maxerror=1.0,
                                n_iter=100)
```

Get the best model configuration after try the two method above:

```python
SklRegressors.test_all_until(rkf_cv_n_splits=5,
                             rkf_cv_n_repeats=10,
                             n_rand_iter=20,
                             maxerror=1.0,
                             n_iter=100)
```

Write a summary file with the best configuration of hyperparameters and statistical data from this best model:

```python
summary_df = SklRegressors.write_log(path="",
                                     filename="skl_regressor_test_summary")
```
