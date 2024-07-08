import hashlib
import textwrap
from dataclasses import Field, dataclass, fields
from functools import partial
from typing import Union

dataclass_component = partial(dataclass, frozen=True)

BasicType = Union[
    str | bool | int | float | type | None,
    list["BasicType"],
    tuple["BasicType", ...],
    set["BasicType"],
    dict[str, "BasicType"],
]
ExtendedBasicType = Union[
    "Component",
    str | bool | int | float | type | None,
    list["ExtendedBasicType"],
    tuple["ExtendedBasicType", ...],
    set["ExtendedBasicType"],
    dict[str, "ExtendedBasicType"],
]


@dataclass(frozen=True)
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

            if isinstance(obj, Component):
                loggable_dict = [
                    (f.name, _loggable_dict(getattr(obj, f.name))) for f in fields(obj)
                ]
                loggable_dict.append(("__class__", obj.__class__.__qualname__))
                return dict(loggable_dict)

            if isinstance(obj, (list, tuple, set)):
                return type(obj)(_loggable_dict(v) for v in obj)  # type: ignore

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

        def filter_defaults(f: Field):
            if show_defaults:
                return True
            is_default = getattr(self, f.name) == f.default
            return not is_default

        def filter_hidden(f: Field):
            return True if show_hidden_repr else f.repr

        class TypeCustomRepr:
            def __init__(self, x: type):
                self.x = x

            def __repr__(self):
                module = self.x.__module__
                if module == "builtins":
                    return self.x.__qualname__
                return module + "." + self.x.__qualname__

        def get_repr(obj):
            if isinstance(obj, dict):
                return {k: get_repr(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set)):
                return type(obj)(get_repr(v) for v in obj)
            if isinstance(obj, type):
                return TypeCustomRepr(obj)
            if isinstance(obj, Component):
                return obj._repr(  # pylint: disable=protected-access
                    show_defaults, show_hidden_repr
                )
            return obj

        filtered_fields = (
            f.name for f in fields(self) if filter_defaults(f) and filter_hidden(f)
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
