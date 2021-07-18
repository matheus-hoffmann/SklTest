import numpy as np
import pandas as pd


def doe_normalize(matrix: np) -> tuple[np, list, list]:
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
    df = pd.read_csv(filepath)
    original_data = df.to_numpy()
    original_input_data = original_data[:, :-1]
    input_data, x0_list, dx_list = doe_normalize(original_input_data)
    output_data = np.reshape(original_data[:, -1], (-1, 1))
    return df, input_data, output_data


def check_max_n_iter(desired_n_iter:int, space:dict) -> int:
    n_comb_keys = 1
    for value in space.values():
        n_comb_keys *= len(value)
    if desired_n_iter > n_comb_keys:
        return n_comb_keys
    else:
        return desired_n_iter


def split_dict(dictionary:dict) -> str:
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
