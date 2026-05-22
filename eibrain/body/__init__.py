"""Compatibility shim for legacy eibrain body/runtime modules.

The canonical runtime owner is ``eihead``. Keep these modules importable while
legacy deployments migrate, but do not add new hot-path consumers here.
"""

COMPATIBILITY_SHIM = True
DEPRECATED_RUNTIME_OWNER = "eihead"
DEPRECATION_REASON = (
    "eibrain.body is retained as an eibrain compatibility shim; device daemon "
    "and real-time hardware runtime ownership has moved to eihead."
)

__all__ = [
    "COMPATIBILITY_SHIM",
    "DEPRECATED_RUNTIME_OWNER",
    "DEPRECATION_REASON",
]
