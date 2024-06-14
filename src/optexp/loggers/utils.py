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
