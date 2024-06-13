import numpy as np
import pandas as pd


def should_convert_column_to_numpy(series: pd.Series):
    def is_string_repr_of_array(entry):
        if isinstance(entry, str):
            if entry.startswith("[") and entry.endswith("]"):
                return True
        return False

    def is_list_of_elements_mostly_floats(entry):
        def list_elem_is_float_or_str_nan(list_element):
            return (
                isinstance(list_element, float)
                or isinstance(list_element, int)
                or (isinstance(list_element, str) and list_element == "NaN")
            )

        if isinstance(entry, list):
            if all([list_elem_is_float_or_str_nan(elem) for elem in entry]):
                return True
        return False

    if len(series) > 1:
        val = series[1]
    else:
        val = series[0]

    if is_string_repr_of_array(val) or is_list_of_elements_mostly_floats(val):
        return True
    else:
        return False


def column_to_numpy(x):
    """Convert string repr of numpy arrays to numpy arrays."""

    def convert_str_to_numpy(str_repr):
        str_repr = str_repr.strip("[]")
        str_repr = str_repr.replace("'NaN'", "nan")
        return np.fromstring(str_repr, sep=", ", dtype=float)

    def convert_list_to_numpy(list_repr):
        list_repr = [np.inf if x == "Infinity" else x for x in list_repr]
        return np.array(list_repr, dtype=np.float32)

    if x is None:
        return np.nan
    if type(x) is float or type(x) is int or type(x) is np.ndarray:
        return x
    elif type(x) is str:
        return convert_str_to_numpy(x)
    elif isinstance(x, list):
        return convert_list_to_numpy(x)
    else:
        raise ValueError(f"Cannot convert row, unknown type {type(x)}")
