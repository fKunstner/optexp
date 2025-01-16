import json
from pathlib import Path

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
            return isinstance(x, str) and x in ["NaN", "inf"]

        return isinstance(entry, list) and all(
            list_elem_is_float_or_str_nan(elem) for elem in entry
        )

    def find_first_entry_with_actual_data(series):
        if len(series) == 1:
            return series[0]

        for i in range(len(series)):
            if series[i] is not None and series[i] is not np.nan:
                return series[i]

        return series[0]

    entry = find_first_entry_with_actual_data(series)
    return is_string_repr_of_array(entry) or is_list_of_float_like(entry)


def numpyfy(df: pd.DataFrame) -> pd.DataFrame:
    with pd.option_context("future.no_silent_downcasting", True):
        df = df.replace(
            {
                "Infinity": np.inf,
                "inf": np.inf,
                "NaN": np.nan,
                "nan": np.nan,
            },
        ).infer_objects()

    for key in df.columns:
        if should_convert_column_to_numpy(df[key]):
            df[key] = df[key].apply(column_to_numpy)
    return df


def flatten_dict(x):
    return pd.io.json._normalize.nested_to_record(x)  # pylint: disable=protected-access


def get_hash_directory(base_directory: Path, object_hash: str, unique_id: str) -> Path:
    """Get a directory for a unique_id in a hash directory.

    Behaves like a dictionary of paths, returns a unique path for unique_id,
    but uses a hash directory structure to avoid having too many files in a single directory.

    This function manages directory structures that look like as follows::

        base_dir/
        ├─ hash1/
        │  ├─ mapping.json
        │  ├─ 0/
        │  ├─ 1/
        ├─ hash2/
        │  ├─ mapping.json
        │  ├─ 0/
        ├─ hash3/
        ...

    The mapping.json contains a dictionary mapping unique_id to the subdirectory.

    Args:
        base_directory (Path): the base directory containing the hash directories.
        object_hash (str): the hash of the object that the unique_id is associated with
        unique_id (str): the unique identifier for the object
    """

    hash_basedir = base_directory / object_hash
    if not hash_basedir.exists():
        hash_basedir.mkdir(parents=True)

    mapping_file = hash_basedir / "mapping.json"
    if not mapping_file.exists():
        mapping_file.touch()
        with mapping_file.open("w") as f:
            json.dump({}, f)

    try:
        mapping = json.loads(mapping_file.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Could not decode mapping file at {mapping_file}. Might be corrupted."
        ) from exc

    if unique_id in mapping:
        hash_dir = hash_basedir / mapping[unique_id]
    else:
        new_id = len(mapping)
        mapping[unique_id] = str(new_id)
        with mapping_file.open("w") as f:
            json.dump(mapping, f)
        hash_dir = hash_basedir / str(new_id)

    if not hash_dir.exists():
        hash_dir.mkdir()
    return hash_dir
