# pylint: disable-msg = wildcard-import, unused-wildcard-import, unused-import
"""
Standard import for MindSpore mode.

Extends the import `from phi.flow import *` by MindSpore-related functions and modules.

The following MindSpore modules are included: `torch`, *torch.nn.functional* as `torchf`, `optim`.

Importing this module registers the MindSpore backend as the default backend unless called within a backend context.
New tensors created via `phi.math` functions will be backed by PyTorch tensors.

See `phi.flow`, `phi.tf.flow`, `phi.jax.flow`.
"""

from phi.flow import *
from . import MINDSPORE

#from .nets import parameter_count, get_parameters, save_state, load_state, dense_net, u_net, update_weights, adam, conv_net, res_net, sgd, sgd as SGD, rmsprop, adagrad, conv_classifier, invertible_net, fno

import mindspore as ms
import mindspore.ops.functional as msf
import mindspore.ops.Optimizer as optim

if not backend.context_backend():
    backend.set_global_default_backend(MINDSPORE)
else:
    from ..math.backend import PHI_LOGGER as _LOGGER
    _LOGGER.warning(f"Importing '{__name__}' within a backend context will not set the default backend.")
