import numpy as np
import pandas as pd


def doe_normalize(matrix: np) -> tuple[np, list, list]:
    """
    Normalize each column of the matrix between [-1, 1] and return the keys to recompose the original data.

    Parameters
    ----------
    matrix : np
        Original data matrix

    Return
    ------
    normalized_matrix : np
        Normalized matrix
    x0_list : lst
        Value of the real variable at the center of the experimental region
    dx_list : lst
        Value of the deviation from x0
    """
    x0_list = []
    dx_list = []
    normalized_matrix = np.zeros(matrix.shape)
    for col in range(matrix.shape[1]):
        x_max = max(matrix[:, col])
        x_min = min(matrix[:, col])
        x0_list.append(0.5 * (x_max + x_min))
        dx_list.append(0.5 * (x_max - x_min))
        if dx_list[-1] < 1e-6:
            dx_list[-1] = dx_list[-1] + 1e-6
        normalized_matrix[:, col] = (matrix[:, col] - x0_list[-1]) / dx_list[-1]
    return normalized_matrix, x0_list, dx_list


def read_data(filepath: str) -> tuple[pd, np, np]:
    """
    Read the data file and return the dataframe and the input/output arrays.
    It is assumed that is single output (SISO or MISO) analysis, and the last column must be the output.

    Parameters
    ----------
    filepath : str
        Path to the data file

    Return
    ------
    df : np
        Original data in a dataframe
    input_data : np
        Normalized input matrix [n x m]
    output_data : np
        Output array [n x 1]
    """
    df = pd.read_csv(filepath)
    original_data = df.to_numpy()
    original_input_data = original_data[:, :-1]
    input_data, x0_list, dx_list = doe_normalize(original_input_data)
    output_data = original_data[:, -1]
    return df, input_data, output_data


def check_max_n_iter(desired_n_iter:int, space:dict) -> int:
    """
    Check if the desired number of iterations is higher than the number of hyperparameters combinations.

    Parameters
    ----------
    desired_n_iter : int
        Desired number of iterations in a RandomizedSearchCV
    space : dict
        Possible values of each hyperparameter of a Scikit-Learn method

    Return
    ------
    Number of iterations to run in RandomizedSearchCV
    """
    n_comb_keys = 1
    for value in space.values():
        n_comb_keys *= len(value)
    if desired_n_iter > n_comb_keys:
        return n_comb_keys
    else:
        return desired_n_iter


def split_dict(dictionary:dict) -> str:
    """
    Convert a dict to a string, separating each key by a /

    Parameters
    ----------
    dictionary : dict
        Dictionary to be converted

    Return
    ------
    final_text : str
        String with the dictionary converted
    """
    keys = list(dictionary.keys())
    values = list(dictionary.values())
    final_text = ""
    for i in range(len(keys) - 1):
        final_text += str(keys[i])
        final_text += ": "
        final_text += str(values[i])
        final_text += "/"
    final_text += str(keys[-1])
    final_text += ": "
    final_text += str(values[-1])
    return final_text


def adjusted_r2_score(r2:float, x:np) -> float:
    """
    Compute the adjusted R² score

    Parameters
    ----------
    r2 : float
        Data R² previously computed
    x : np
        Input data array

    Return
    ------
    r2a : float
        Adjusted R²
    """
    n = x.shape[0]
    p = x.shape[1]
    den = (n - p - 1)
    if den == 0:
        den += 1e-6
    r2a = 1 - (1 - r2) * (n - 1) / den
    if r2a > 1:
        r2a = 1.0
    return r2a


def check_metric(all_metrics:dict, desired_metric:str, std_metric:str) -> str:
    """
    Check if a metric is available in a dict

    Parameters
    ----------
    all_metrics : dict
        Dictionary with the available metrics as keys
    desired_metric : str
        The metric that the user want to use
    std_metric : str
        A standard metric that is going to be used if the desired_metric is not available

    Return
    ------
    metric : str
        The metric to be used
    """
    available_metrics = list(all_metrics.keys())
    metric = std_metric
    if desired_metric in available_metrics:
        metric = desired_metric
    return metric
