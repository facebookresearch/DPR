from collections import namedtuple
from typing import Any, Dict


def to_namedtuple(d: Dict[str, Any]):
    ARGS_TUPLE = namedtuple("RunArguments", sorted(d))
    return ARGS_TUPLE(**d)
