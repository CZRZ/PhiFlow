import numbers
import warnings
from contextlib import contextmanager
from functools import wraps
from typing import List, Callable, Optional, Set, Tuple, Any, Union

import numpy as np
import mindspore as ms
# from mindspore import fft
import mindspore.ops.functional as msf
from packaging import version

from phi.math import DType
from phi.math.backend import Backend, NUMPY, ComputeDevice, PHI_LOGGER
from phi.math.backend._backend import combined_dim, SolveResult, get_functional_derivative_order, TensorType
from phi.math.backend._dtype import DType


class MindSporeBackend(Backend):

    def __init__(self):
        devices = [ComputeDevice(self, ms.get_context(
            "device_target"), ms.get_context("device_target"), -1, 1,
            "id="+str(ms.get_context("device_id")), ms.get_context("device_target"))]

        ms.context.set_context(pynative_synchronize=True)
        #ms.context.set_context(mode=ms.GRAPH_MODE)
        Backend.__init__(self, 'MindSpore', devices,
                         devices[-1])

    def prefers_channels_last(self) -> bool:
        return False

    def is_module(self, obj):
        return isinstance(obj, (ms.nn.Cell))

    def is_tensor(self, x, only_native=False):
        if isinstance(x, ms.Tensor):
            return True
        if isinstance(x, ms.COOTensor):
            return True
        if only_native:
            return False
        if isinstance(x, numbers.Number):
            return True
        if isinstance(x, (tuple, list)) and all(isinstance(c, numbers.Number) for c in x):
            return True
        if isinstance(x, np.ndarray) and x.dtype != np.object:
            return True
        return False

    def is_sparse(self, x):
        return isinstance(x, (ms.COOTensor, ms.CSRTensor))

    def as_tensor(self, x, convert_external=True):
        if isinstance(x, ms.nn.Cell):
            return x
        if self.is_tensor(x, only_native=convert_external):
            tensor = x
        elif isinstance(x, np.ndarray):
            try:
                tensor = ms.Tensor(x)
            except ValueError:  # or TypeError?
                tensor = ms.Tensor(x.copy())
        elif isinstance(x, (tuple, list)):
            try:
                x = np.stack(x)
                tensor = ms.Tensor(x)
            except ValueError:  # there may be Tensors inside the list
                components = [self.as_tensor(c) for c in x]
                tensor = ms.ops.stack(components, dim=0)
        else:
            tensor = ms.Tensor(x)
        # --- Enforce Precision ---
        if self.is_tensor(tensor, only_native=True):
            dtype = self.dtype(tensor)
            if dtype.kind == float:
                tensor = self.to_float(tensor)
            elif dtype.kind == complex:
                tensor = self.to_complex(tensor)
        return tensor

    def recursive_as_tensor(self, obj):
        if isinstance(obj, (tuple, list)):
            return [self.recursive_as_tensor(c) for c in obj]
        elif isinstance(obj, dict):
            return {key: self.recursive_as_tensor(value) for key, value in obj.items()}
        else:
            return self.as_tensor(obj)

    def auto_cast(self, *tensors, **kwargs) -> list:
        tensors = [t if isinstance(t, (numbers.Number, bool)) else self.as_tensor(
            t, True) for t in tensors]
        return Backend.auto_cast(self, *tensors, **kwargs)

    def is_available(self, tensor) -> bool:
        if self.is_tensor(tensor, only_native=True):
            return ms.get_context("mode") == ms.PYNATIVE_MODE
        else:
            return True

    def numpy(self, tensor):
        return tensor.asnumpy()

    def to_dlpack(self, capsule):
        raise NotImplementedError()

    def from_dlpack(self, capsule):
        raise NotImplementedError()

    def copy(self, tensor, only_mutable=False):
        return tensor.copy()

    def get_device(self, tensor: TensorType) -> ComputeDevice:
        return self.get_device_by_ref(str(tensor.device))

    def allocate_on_device(self, tensor: TensorType, device: ComputeDevice) -> TensorType:
        raise NotImplementedError()

    def multi_slice(self, tensor, slices: tuple):
        neg_slices = [i for i, s in enumerate(slices) if isinstance(
            s, slice) and s.step is not None and s.step < 0]
        if neg_slices:
            tensor = tensor.flip(neg_slices)
        pos_slices = [slice(s.start, s.stop, -s.step)
                      if i in neg_slices else s for i, s in enumerate(slices)]
        return tensor[tuple(pos_slices)]

    def sqrt(self, x):
        return ms.ops.sqrt(x)

    def exp(self, x):
        return ms.ops.exp(x)

    def sin(self, x):
        return ms.ops.sin(x)

    def arcsin(self, x):
        return ms.ops.arcsin(x)

    def cos(self, x):
        return ms.ops.cos(x)

    def arccos(self, x):
        return ms.ops.arccos(x)

    def tan(self, x):
        return ms.ops.tan(x)

    def arctan(self, ):
        return ms.ops.arctan()

    def sinh(self, x):
        return ms.ops.sinh(x)

    def arcsinh(self, x):
        return ms.ops.arcsinh(x)

    def arccosh(self, x):
        return ms.ops.arccosh(x)

    def cosh(self, x):
        return ms.ops.cosh(x)

    def tanh(self, x):
        return ms.ops.tanh(x)

    def arctanh(self, x):
        return ms.ops.arctanh(x)

    def log(self, x):
        return ms.ops.log(x)

    def log2(self, x):
        return ms.ops.log2(x)

    def log10(self, x):
        return ms.ops.log10(x)

    def isfinite(self, x):
        return ms.ops.isfinite(x)

    def abs(self, x):
        return ms.ops.abs(x)

    def sign(self, x):
        return ms.ops.sign(x)

    def round(self, x):
        return ms.ops.round(x)

    def ceil(self, x):
        return ms.ops.ceil(x)

    def floor(self, x):
        return ms.ops.floor(x)

    def nonzero(self, values):
        return ms.ops.nonzero(values)

    seed = staticmethod(ms.set_seed)

    def einsum(self, equation, *tensors):
        tensors = self.auto_cast(*tensors, bool_to_int=True, int_to_float=True)
        return ms.ops.einsum(equation, *tensors)

    def jit_compile(self, f: Callable) -> Callable:
        return ms.jit(f)

    def custom_gradient(self, f: Callable, gradient: Callable = None, get_external_cache: Callable = None, on_call_skipped: Callable = None) -> Callable:
        raise NotImplementedError()

    def expand_dims(self, a, axis=0, number=1):
        for _ in range(number):
            a = ms.ops.unsqueeze(a, dim=axis)
        return a

    def transpose(self, tensor, axes):
        return tensor.transpose(axes)

    def equal(self, x, y):
        x, y = self.auto_cast(x, y)
        return x == y

    def random_uniform(self, shape, low, high, dtype: Union[DType, None]):
        dtype = dtype or self.float_type
        if dtype.kind == float:
            return low + (high - low) * ms.ops.rand(size=shape, dtype=to_ms_dtype(dtype), device=self.get_default_device().ref)
        elif dtype.kind == complex:
            real = low.real + (high.real - low.real) * ms.ops.rand(size=shape, dtype=to_ms_dtype(
                DType(float, dtype.precision)), device=self.get_default_device().ref)
            imag = low.imag + (high.imag - low.imag) * ms.ops.rand(size=shape, dtype=to_ms_dtype(
                DType(float, dtype.precision)), device=self.get_default_device().ref)
            return real + 1j * imag
        elif dtype.kind == int:
            return ms.ops.randint(low, high, shape, dtype=to_ms_dtype(dtype))
        else:
            raise ValueError(dtype)

    def random_normal(self, shape, dtype: DType):
        mstype = to_ms_dtype(dtype or self.float_type)
        out = ms.ops.randn(shape, dtype=mstype)
        print(out)
        return out

    def stack(self, values, axis=0):
        values = [self.as_tensor(v) for v in values]
        return ms.ops.stack(values, axis=axis)

    def concat(self, values, axis):
        values = [self.as_tensor(v) for v in values]
        return ms.ops.cat(values, axis=axis)

    def pad(self, value, pad_width, mode='constant', constant_values=0):
        mode = {'constant': 'constant', 'reflect': 'reflect',
                'boundary': 'replicate', 'periodic': 'circular'}.get(mode, None)
        if not mode:
            return NotImplemented
        # for PyTorch, we have to reshape value such that the outer 2 dimensions are not padded.
        ndims = self.ndims(value)
        no_pad_dims = [i for i in range(ndims) if pad_width[i] == (0, 0)]
        pad_dims = [i for i in range(ndims) if pad_width[i] != (0, 0)]
        if not pad_dims:
            return value
        if len(pad_dims) > 3:
            return NotImplemented
        value = ms.ops.transpose(value, tuple(no_pad_dims + pad_dims))
        if len(no_pad_dims) == 0:
            value = ms.ops.unsqueeze(ms.ops.unsqueeze(value, 0), 0)
            def undo_transform(x): return ms.ops.squeeze(
                ms.ops.squeeze(x, 0), 0)
        elif len(no_pad_dims) == 1:
            value = ms.ops.unsqueeze(value, 0)
            def undo_transform(x): return ms.ops.squeeze(x, 0)
        elif len(no_pad_dims) == 2:
            def undo_transform(x): return x
        else:
            old_shape = value.shape
            value = self.reshape(value, (1, np.prod([value.shape[i] for i in range(
                len(no_pad_dims))]), *value.shape[len(no_pad_dims):]))

            def undo_transform(x): return x.view(
                *[old_shape[i] for i in range(len(no_pad_dims))], *x.shape[2:])
        pad_width_reordered = [pad_width[i] for i in pad_dims]
        pad_width_spatial = [item for sublist in reversed(
            pad_width_reordered) for item in sublist]  # flatten
        try:
            constant_values = self.dtype(value).kind(constant_values)
            if mode is 'constant':
                # supports 3D to 5D (batch, channel, 1D to 3D)
                result = ms.ops.pad(value, pad_width_spatial,
                                    mode, value=constant_values)
            else:
                result = ms.ops.pad(value, pad_width_spatial, mode)

        except RuntimeError as err:
            warnings.warn(f"MindSpore error {err}", RuntimeWarning)
            return NotImplemented
        result = undo_transform(result)
        inv_perm = tuple(np.argsort(no_pad_dims + pad_dims))
        result = ms.ops.transpose(result, inv_perm)
        return result

    def grid_sample(self, grid, coordinates, extrapolation: str):
        return super().grid_sample(grid, coordinates, extrapolation)

    def reshape(self, value, shape):
        value = self.as_tensor(value)
        return ms.ops.reshape(value, shape)

    def sum(self, value, axis=None, keepdims=False):
        if axis is None:
            axis = tuple(range(len(value.shape)))
        if axis == () or axis == []:
            return value
        return ms.ops.sum(value, dim=axis, keepdim=keepdims)

    def prod(self, value, axis=None):
        if not self.is_tensor(value, only_native=True):
            return NUMPY.prod(value, axis)
        if axis is None:
            axis = tuple(range(len(value.shape)))
        if isinstance(axis, (tuple, list)):
            for dim in reversed(sorted(axis)):
                value = ms.ops.prod(value, axis=dim)
            return value
        return ms.ops.prod(value, axis=axis)

    def any(self, boolean_tensor, axis=None, keepdims=False):
        boolean_tensor = self.as_tensor(boolean_tensor, convert_external=True)
        if self.dtype(boolean_tensor).kind != bool:
            boolean_tensor = boolean_tensor != 0
        if axis is None:
            return ms.ops.any(boolean_tensor)
        else:
            axes = axis if isinstance(axis, (tuple, list)) else [axis]
            for axis in reversed(sorted(axes)):
                boolean_tensor = ms.ops.any(
                    boolean_tensor, axis=axis, keep_dims=keepdims)
            return boolean_tensor

    def all(self, boolean_tensor, axis=None, keepdims=False):
        boolean_tensor = self.as_tensor(boolean_tensor, convert_external=True)
        if self.dtype(boolean_tensor).kind != bool:
            boolean_tensor = boolean_tensor != 0
        if axis is None:
            return ms.ops.all(boolean_tensor)
        else:
            axes = axis if isinstance(axis, (tuple, list)) else [axis]
            for axis in reversed(sorted(axes)):
                boolean_tensor = ms.ops.all(
                    boolean_tensor, axis=axis, keep_dims=keepdims)
            return boolean_tensor

    def quantile(self, x, quantiles):
        return super().quantile(x, quantiles)

    def argsort(self, x, axis=-1):
        return super().argsort(x, axis)

    def searchsorted(self, sorted_sequence, search_values, side: str, dtype=...):
        return super().searchsorted(sorted_sequence, search_values, side, dtype)

    def divide_no_nan(self, x, y):
        x, y = self.auto_cast(x, y)
        result = x / ms.ops.where(y == 0, ms.ops.ones_like(y), y)
        result = ms.ops.where(y == 0, ms.ops.zeros_like(result), result)
        return result

    def where(self, condition, x=None, y=None):
        condition = self.as_tensor(condition).bool()
        x, y = self.auto_cast(x, y)
        x = self.as_tensor(x)
        y = self.as_tensor(y)
        return ms.ops.where(condition, x, y)

    def mean(self, value, axis=None, keepdims=False):
        if self.dtype(value).kind not in (float, complex):
            value = self.to_float(value)
        return ms.ops.mean(value, axis=axis, keep_dims=keepdims)

    def range(self, start, limit=None, delta=1, dtype: DType = ...):
        return super().range(start, limit, delta, dtype)

    def zeros(self, shape, dtype: DType = None):
        return ms.ops.zeros(shape, dtype=to_ms_dtype(dtype or self.float_type))

    def ones(self, shape, dtype: DType = None):
        return ms.ops.ones(shape, dtype=to_ms_dtype(dtype or self.float_type))

    def ones_like(self, tensor):
        return super().ones_like(tensor)

    def meshgrid(self, *coordinates):
        coordinates = [self.as_tensor(c) for c in coordinates]
        return ms.ops.meshgrid(*coordinates, indexing='ij')

    def linspace(self, start, stop, number):
        if self.is_tensor(stop, only_native=True) or self.is_tensor(start, only_native=True):
            unit = ms.ops.linspace(0, 1, number)
            return unit * (stop - start) + start
        else:
            return ms.ops.linspace(float(start), float(stop), number)

    def mul_matrix_batched_vector(self, A, b):
        A, b = self.auto_cast(A, b)
       # if isinstance(A, ms.Tensor) and (self.is_sparse(A)):
        if self.is_sparse(A):
            result = ms.ops.SparseTensorDenseMatmul()(
                A.indices, A.values, A.shape, ms.ops.swapaxes(b, 0, 1))
            return ms.ops.swapaxes(result, 0, 1)
        else:
            return ms.ops.swapaxes(ms.ops.matmul(A, ms.ops.swapaxes(b, -1, -2)), -1, -2)

    def max(self, x, axis=None, keepdims=False):
        if axis is None:
            result = ms.ops.max(x)
            if keepdims:
                result = self.expand_dims(result, axis=0, number=self.ndims(x))
            return result
        elif isinstance(axis, (tuple, list)):
            for dim in reversed(sorted(axis)):
                x, _ = ms.ops.max(x, axis=dim, keepdims=keepdims)
            return x
        else:
            return ms.ops.max(x, axis=axis, keepdims=keepdims)[0]

    def min(self, x, axis=None, keepdims=False):
        if axis is None:
            result = ms.ops.min(x)
            if keepdims:
                result = self.expand_dims(result, axis=0, number=self.ndims(x))
            return result
        elif isinstance(axis, (tuple, list)):
            for dim in reversed(sorted(axis)):
                x, _ = ms.ops.min(x, axis=dim, keepdims=keepdims)
            return x
        else:
            return ms.ops.min(x, axis=axis, keepdims=keepdims)[0]

    def maximum(self, a, b):
        a_ = self.as_tensor(a)
        b_ = self.as_tensor(b)
        return ms.ops.maximum(a_, b_)

    def minimum(self, a, b):
        a_ = self.as_tensor(a)
        b_ = self.as_tensor(b)
        return ms.ops.minimum(a_, b_)

    def clip(self, x, minimum, maximum):
        if isinstance(minimum, numbers.Number) and isinstance(maximum, numbers.Number):
            return ms.ops.clamp(self.as_tensor(x), minimum, maximum)
        else:
            return self.maximum(minimum, self.minimum(x, maximum))

    def shape(self, tensor):
        if self.is_tensor(tensor, only_native=True):
            return tensor.shape
        else:
            return NUMPY.shape(tensor)

    def staticshape(self, tensor):
        if isinstance(tensor, ms.nn.Cell):
            return ()
        if self.is_tensor(tensor, only_native=True):
            return tuple([int(s) for s in tensor.shape])
        else:
            return NUMPY.staticshape(tensor)

    def batched_gather_nd(self, values, indices):
        values = self.as_tensor(values)
        indices = self.as_tensor(indices).long()
        batch_size = combined_dim(values.shape[0], indices.shape[0])
        result = []
        for b in range(batch_size):
            b_indices = self.unstack(indices[min(b, indices.shape[0] - 1)], -1)
            result.append(values[(min(b, values.shape[0] - 1),) + b_indices])
        return self.stack(result, axis=0)

    def cast(self, x, dtype: DType):
        if isinstance(x, (numbers.Number, bool)):
            # Creating a Tensor here would raise warnings during tracing.
            return dtype.kind(x)
        if not self.is_tensor(x, only_native=True):
            x = self.as_tensor(x)
        if self.dtype(x) == dtype:
            return x
        else:
            return x.to(to_ms_dtype(dtype))

    def dtype(self, array) -> DType:
        if self.is_tensor(array, only_native=True):
            return from_ms_dtype(array.dtype)
        else:
            return NUMPY.dtype(array)

    def tile(self, value, multiples):
        return ms.numpy.tile(value, multiples)
        # if isinstance(multiples, np.ndarray):
        #    multiples = multiples.tolist()
        # return self.as_tensor(value).repeat(multiples)

    def repeat(self, x, repeats, axis: int, new_length=None):
        if isinstance(repeats, (np.ndarray, tuple, list)):
            repeats = self.as_tensor(repeats)
        return ms.ops.repeat_elements(self.as_tensor(x), repeats, axis)

    def sparse_coo_tensor(self, indices, values, shape):
        indices = self.to_int64(indices)
        values = self.to_float(values)

        @ms.jit  # the output of ms.sparse_coo_tensor is considered constant
        def sparse_coo_tensor(values, indices, shape) -> ms.COOTensor:
            return ms.COOTensor(indices, values, shape)

        return sparse_coo_tensor(values, indices, shape)

    def csr_matrix(self, column_indices, row_pointers, values, shape: Tuple[int, int]):
        row_pointers = self.as_tensor(row_pointers)
        column_indices = self.as_tensor(column_indices)
        values = self.as_tensor(values)
        return ms.CSRTensor(row_pointers, column_indices, values, shape)

    def stop_gradient(self, value):
        return ms.ops.stop_gradient(value)


def to_ms_dtype(dtype: DType):
    return _TO_MS[dtype]


def from_ms_dtype(ms_dtype):
    if ms_dtype in _FROM_MS:
        return _FROM_MS[ms_dtype]
    else:
        kind = {'i': int, 'b': bool, 'f': float, 'c': complex}[ms_dtype.kind]
        return DType(kind, ms_dtype.itemsize * 8)


_TO_MS = {
    DType(float, 16): ms.float16,
    DType(float, 32): ms.float32,
    DType(float, 64): ms.float64,
    DType(complex, 64): ms.complex64,
    DType(complex, 128): ms.complex128,
    DType(int, 8): ms.int8,
    DType(int, 16): ms.int16,
    DType(int, 32): ms.int32,
    DType(int, 64): ms.int64,
    DType(bool): ms.bool_,
}
_FROM_MS = {np: dtype for dtype, np in _TO_MS.items()}
