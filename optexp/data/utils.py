import numpy as np
import pandas as pd


def pprint_dict(adict):
    """Pretty printing for logging dictionaries.

    Ignores keys containing lists, prints floating points in scientific notation, shows
    accuracy in percentage.
    """

    def fmt_entry(k, v):
        if isinstance(v, float):
            if "Accuracy" in k or "acc" in k:
                return f"{k}={100*v:.1f}"
            return f"{k}={v:.2e}"
        return f"{k}={v}"

    dict_str = ", ".join(
        fmt_entry(k, v) for k, v in sorted(adict.items()) if not hasattr(v, "__len__")
    )
    return "{" + dict_str + "}"


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
    if isinstance(x, (float, int, np.ndarray)):
        return x
    if isinstance(x, str):
        return convert_str_to_numpy(x)
    if isinstance(x, list):
        return convert_list_to_numpy(x)
    raise ValueError(f"Cannot convert row, unknown type {type(x)}")


def should_convert_column_to_numpy(series: pd.Series):
    def is_string_repr_of_array(entry):
        if isinstance(entry, str):
            if entry.startswith("[") and entry.endswith("]"):
                return True
        return False

    def is_list_of_elements_mostly_floats(entry):
        def list_elem_is_float_or_str_nan(list_element):
            return isinstance(list_element, (float, int)) or (
                isinstance(list_element, str) and list_element == "NaN"
            )

        return isinstance(entry, list) and all(
            list_elem_is_float_or_str_nan(elem) for elem in entry
        )

    val = series[1] if len(series) > 1 else series[0]

    return is_string_repr_of_array(val) or is_list_of_elements_mostly_floats(val)
