import hashlib
import textwrap
from typing import Union

import attr
from attrs import fields, frozen

BasicType = Union[
    str | bool | int | float | type | None,
    list["BasicType"],
    tuple["BasicType", ...],
    set["BasicType"],
    frozenset["BasicType"],
    dict[str, "BasicType"],
]
ExtendedBasicType = Union[
    "Component",
    str | bool | int | float | type | None,
    list["ExtendedBasicType"],
    dict[str, "ExtendedBasicType"],
]


@frozen
class Component:
    """Base class for all components in the experiments."""

    def loggable_dict(self) -> dict[str, BasicType]:
        """Returns a dictionary representation of the component that can be logged."""

        def _loggable_dict(obj: ExtendedBasicType) -> BasicType:
            if isinstance(obj, dict):
                if not all(isinstance(k, str) for k in obj.keys()):
                    raise ValueError(
                        f"Dict keys must be strings to have be loggable. Got {obj.keys()}"
                    )

            if isinstance(obj, Component) and attr.has(obj.__class__):
                loggable_dict = [
                    (attribute.name, _loggable_dict(getattr(obj, attribute.name)))
                    for attribute in fields(obj.__class__)
                ]
                loggable_dict.append(("__class__", obj.__class__.__qualname__))
                return dict(loggable_dict)

            if isinstance(obj, (list, tuple, set, frozenset)):
                return list(_loggable_dict(v) for v in obj)  # type: ignore

            if isinstance(obj, dict):
                return dict((k, _loggable_dict(v)) for k, v in obj.items())

            if isinstance(obj, (str, int, bool, float, type)) or obj is None:
                return obj

            raise ValueError(
                textwrap.dedent(
                    f"""
                    To be loggable, components must be composed of basic types and components
                    (str, bool, int, float, list, tuple, set, dict, type, and Component). 
                    Received {type(obj)} ({obj}).
                    """
                )
            )

        return _loggable_dict(self)  # type: ignore

    def equivalent_definition(self) -> str:
        return self._repr(show_defaults=False, show_hidden_repr=False)

    def full_definition(self) -> str:
        return self._repr(show_defaults=True, show_hidden_repr=True)

    def _repr(self, show_defaults=False, show_hidden_repr=False) -> str:

        def filter_defaults(attribute):
            if show_defaults:
                return True
            is_default = getattr(self, attribute.name) == attribute.default
            return not is_default

        def filter_hidden(attribute):
            if show_hidden_repr:
                return True
            return attribute.repr

        def inner_repr(obj: dict | set | frozenset | list | tuple) -> str:
            if isinstance(obj, dict):
                return ", ".join(f"{k}: {get_repr(obj[k])}" for k in sorted(obj.keys()))
            if isinstance(obj, (set, frozenset)):
                return ", ".join(get_repr(v) for v in sorted(obj))
            if isinstance(obj, (list, tuple)):
                return ", ".join(get_repr(v) for v in obj)
            raise ValueError(f"Unexpected type {type(obj)}")

        def get_repr(obj) -> str:
            if isinstance(obj, (dict, set, frozenset)):
                return "{" + inner_repr(obj) + "}"
            if isinstance(obj, tuple):
                return "(" + inner_repr(obj) + ")"
            if isinstance(obj, list):
                return "[" + inner_repr(obj) + "]"
            if isinstance(obj, float):
                return f"{obj:.5g}"
            if isinstance(obj, Component):
                return obj._repr(  #  pylint: disable=protected-access
                    show_defaults, show_hidden_repr
                )
            return str(obj)

        filtered_fields = (
            attribute.name
            for attribute in fields(self.__class__)  # type: ignore
            if filter_defaults(attribute) and filter_hidden(attribute)
        )
        args_repr = ", ".join(
            f"{k}={get_repr(getattr(self, k))}" for k in filtered_fields
        )
        return f"{self.__class__.__name__}({args_repr})"

    def equivalent_hash(self) -> str:
        return hashlib.sha1(
            str.encode(self.equivalent_definition()),
            usedforsecurity=False,
        ).hexdigest()

    def short_equivalent_hash(self) -> str:
        return self.equivalent_hash()[:8]

    def __lt__(self, other):
        return self.equivalent_definition() < other.equivalent_definition()
