import textwrap
from dataclasses import dataclass, fields
from typing import Union


@dataclass(frozen=True)
class Component:
    """Base class for all components in the experiments."""

    def loggable_dict(self) -> dict:
        def _loggable_dict(
            obj: Union[
                Component, dict, list, bool, int, float, type, set, tuple, type, str
            ],
        ) -> Union[dict, list, bool, int, float, type, set, tuple, type, str]:

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
                return type(obj)(_loggable_dict(v) for v in obj)

            if isinstance(obj, dict):
                return dict((k, _loggable_dict(v)) for k, v in obj.items())

            if isinstance(obj, (str, int, bool, float, type)):
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
