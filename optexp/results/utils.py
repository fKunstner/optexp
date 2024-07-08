import numpy as np
import pandas as pd


def pprint_dict(metric_dict: dict[str, float | list[float]]) -> str:
    def format_entry(k, v):
        if isinstance(v, float):
            if "accuracy" in k.lower():
                return k, f"{100*v:.1f}"
            return k, f"{v:.2e}"

        if hasattr(v, "__len__"):
            if len(v) > 3:
                return k, [f"{v[0]:.2e}", "...", f"{v[-1]:.2e}"]
            return k, [f"{x:.2e}" for x in v]
        return k, v

    formatted_entries = [format_entry(k, v) for k, v in metric_dict.items()]
    return "{" + ", ".join(f"{k}={v}" for k, v in formatted_entries) + "}"


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

    def is_list_of_float_like(entry):
        def list_elem_is_float_or_str_nan(x):
            if isinstance(x, (float, int)):
                return True
            return isinstance(x, str) and x == "NaN"

        return isinstance(entry, list) and all(
            list_elem_is_float_or_str_nan(elem) for elem in entry
        )

    more_than_one_entry = len(series) > 1
    if more_than_one_entry:
        entry_to_check = series[1]
    else:
        entry_to_check = series[0]

    return is_string_repr_of_array(entry_to_check) or is_list_of_float_like(
        entry_to_check
    )


def numpyfy(df: pd.DataFrame) -> pd.DataFrame:
    df.replace("Infinity", np.inf, inplace=True)
    for key in df.columns:
        if should_convert_column_to_numpy(df[key]):
            df[key] = df[key].apply(column_to_numpy)
    return df


def flatten_dict(x):
    return pd.io.json._normalize.nested_to_record(x)  # pylint: disable=protected-access
