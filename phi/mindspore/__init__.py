"""
MindSpore integration.

Importing this module registers the MindSpore backend with `phi.math`.
Without this, MindSpore tensors cannot be handled by `phi.math` functions.

To make MindSpore the default backend, import `phi.mindspore.flow`.
"""
from phi import math as _math
from ._mindspore_backend import MindSporeBackend as _MindSporeBackend

MINDSPORE = _MindSporeBackend()
""" Backend for MindSpore operations. """

_math.backend.BACKENDS.append(MINDSPORE)

__all__ = [key for key in globals().keys() if not key.startswith('_')]
