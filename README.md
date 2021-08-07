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
