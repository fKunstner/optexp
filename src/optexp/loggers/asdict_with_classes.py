"""Utility function to convert dataclass instances to dictionaries, while preserving the
class name of the dataclass.

The existing implementation of asdict in dataclasses.py creates
dictionaries containing the content of the dataclass instance, but does
not include the class name of the dataclass.

Example usage for the original asdict::

  @dataclass
  class C:
      x: int
      y: int

  c = C(1, 2)
  assert asdict(c) == {'x': 1, 'y': 2}

This makes it impossible to distinguish between different dataclasses,
for example SGD(lr=0.1) and Adam(lr=0.1).

This implementation creates dictionaries with an additional entry
holding the class name

Example usage::

  @dataclass
  class C:
      x: int
      y: int

  c = C(1, 2)
  assert asdict_with_class(c) == {'__class__': 'C', 'x': 1, 'y': 2}
"""

import copy
import inspect
from dataclasses import fields
from fractions import Fraction

_FIELDS = "__dataclass_fields__"


def _is_dataclass_instance(obj):
    """Returns True if obj is an instance of a dataclass."""
    return hasattr(type(obj), _FIELDS)


def _asdict_inner(obj, dict_factory):
    """Dataclass instance as a dictionary.

    Return the fields of a dataclass instance as a new dictionary mapping field names to
    field values.

    Main change from
    https://github.com/python/cpython/blob/26f396a55f8f208f229bdb700f1d7a17ca81493d/Lib/dataclasses.py#L1287-L1325
    is that it includes the class name of the dataclass in the __class__ field.
    """
    if _is_dataclass_instance(obj):
        result = []
        for f in fields(obj):
            value = _asdict_inner(getattr(obj, f.name), dict_factory)
            result.append((f.name, value))

        result.append(("__class__", obj.__class__.__name__))
        return dict_factory(result)

    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*[_asdict_inner(v, dict_factory) for v in obj])
    if isinstance(obj, (list, tuple, set)):
        return type(obj)(_asdict_inner(v, dict_factory) for v in obj)
    if isinstance(obj, dict):
        return type(obj)(
            (_asdict_inner(k, dict_factory), _asdict_inner(v, dict_factory))
            for k, v in obj.items()
        )
    if inspect.isclass(obj) and obj.__class__.__name__ == "type":
        return str(obj)
    if isinstance(obj, Fraction):
        return str(obj)
    if isinstance(obj, type):
        return str(obj.__name__)
    return copy.deepcopy(obj)


def asdict_with_class(obj, dict_factory=dict):
    """Return the fields of a dataclass instance as a new dictionary mapping field names
    to field values.

    Same as dataclasses.asdict, but with one difference:
    The class name of the dataclass is included in the __class__ field.


    Example usage::

      @dataclass
      class C:
          x: int
          y: int

      c = C(1, 2)
      assert asdict_with_class(c) == {'__class__': 'C', 'x': 1, 'y': 2}

    If given, 'dict_factory' will be used instead of built-in dict.
    The function applies recursively to field values that are
    dataclass instances and built-in containers (tuples, lists, and dicts)

    Adapted from
    https://github.com/python/cpython/blob/26f396a55f8f208f229bdb700f1d7a17ca81493d/Lib/dataclasses.py#L1263-L1284
    """
    return _asdict_inner(obj, dict_factory)
