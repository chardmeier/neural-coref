# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

"""This module contains some helper functions that are used by the neural network code."""

import mention_ranking

import collections
import copy
import json
import math
import numbers
import numpy
import torch

from torch.autograd import Variable


# Factory classes to simplify creating the right objects for CPU and GPU calculations.

class CPUFactory:
    """Factory class that creates objects on the CPU."""

    def __init__(self):
        self.is_cuda = False

    @staticmethod
    def to_device(x):
        return x

    @staticmethod
    def zeros(*args):
        return torch.zeros(*args)

    @staticmethod
    def float_tensor(*args):
        return torch.FloatTensor(*args)

    @staticmethod
    def byte_ones(*args):
        return torch.ones(*args).byte()

    @staticmethod
    def byte_eye(*args):
        return torch.eye(*args).byte()

    @staticmethod
    def long_arange(start, end, step=1):
        return torch.arange(start, end, step).long()

    @staticmethod
    def long_tensor(*args):
        return torch.LongTensor(*args)

    @staticmethod
    def long_zeros(*args):
        return torch.LongTensor(*args).zero_()

    @staticmethod
    def get_single(t):
        if isinstance(t, numbers.Number):
            return t

        assert t.numel() == 1
        if isinstance(t, Variable):
            return t.data[0]
        else:
            return t[0]

    @staticmethod
    def cpu_model(model):
        return model


class CudaFactory:
    """Factory class that creates objects on the GPU."""

    def __init__(self):
        self.is_cuda = True

    @staticmethod
    def to_device(x):
        return x.pin_memory().cuda(async=True)

    @staticmethod
    def zeros(*args):
        return torch.cuda.FloatTensor(*args).zero_()

    @staticmethod
    def float_tensor(*args):
        return torch.cuda.FloatTensor(*args)

    @staticmethod
    def byte_ones(*args):
        return torch.cuda.ByteTensor(*args).zero_() + 1

    @staticmethod
    def byte_eye(*args):
        return torch.eye(*args).byte().pin_memory().cuda(async=True)

    @staticmethod
    def long_arange(start, end, step=1):
        return torch.arange(start, end, step).long().pin_memory().cuda(async=True)

    @staticmethod
    def long_tensor(*args):
        return torch.cuda.LongTensor(*args)

    @staticmethod
    def long_zeros(*args):
        return torch.cuda.LongTensor(*args).zero_()

    @staticmethod
    def get_single(t):
        if isinstance(t, numbers.Number):
            return t

        assert t.numel() == 1
        if isinstance(t, Variable):
            return t.cpu().data[0]
        else:
            return t.cpu()[0]

    @staticmethod
    def cpu_model(model):
        return copy.deepcopy(model).cpu()


def to_cpu(inp):
    """Move objects to the CPU if they aren't already there.

    I'm not sure if this is really necessary. If the cpu() method of the CPU tensors was a true no-op,
    we could use that instead of this helper function. During debugging, I sometimes had the impression
    that calling cpu() on CPU tensors slowed down things, but I never investigated this in detail."""
    if isinstance(inp, tuple):
        return tuple(to_cpu(tensor) for tensor in inp)
    elif isinstance(inp, list):
        return list(to_cpu(tensor) for tensor in inp)
    else:
        if inp.is_cuda:
            return inp.cpu()
        else:
            return inp


def save_model(h5, model):
    """Save model weights and configuration to a HDF5 group."""
    model_type = model.__class__.__name__
    grp = h5.require_group(model_type)

    grp.attrs['net-config'] = json.dumps(model.net_config)

    for name, tensor in model.state_dict().items():
        grp[name] = tensor.cpu().numpy()


def load_model(h5, cuda=False):
    """Load model weights and configuration from a HDF5 group."""
    grp = h5['MentionRankingModel']

    net_config = json.loads(grp.attrs['net-config'])

    model = mention_ranking.setup_model(net_config, cuda)
    model.load_state_dict({name: torch.from_numpy(numpy.array(val)) for name, val in grp.items()})
    if cuda:
        model.cuda()

    return model


def recursive_dict_update(d, u):
    """Update a dictionary recursively to allow overwriting default values.

    Borrowed from https://stackoverflow.com/a/3233356"""
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            r = recursive_dict_update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


def sparse(tensor, sparsity, std=0.01):
    """Fills the 2D input Tensor or Variable as a sparse matrix, where the non-zero elements will be drawn from
    the normal distribution :math:`N(0, 0.01)`, as described in "Deep learning via
    Hessian-free optimization" - Martens, J. (2010).

    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        sparsity: The fraction of elements in each column to be set to zero
        std: the standard deviation of the normal distribution used to generate the non-zero values

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> torch.nn.init.sparse(w, sparsity=0.1)

    This is a numpy-based reimplementation that is much faster than the one in PyTorch.
    """
    if isinstance(tensor, Variable):
        sparse(tensor.data, sparsity, std=std)
        return tensor

    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    if isinstance(tensor, torch.FloatTensor) or isinstance(tensor, torch.cuda.FloatTensor):
        dtype = numpy.float32
    elif isinstance(tensor, torch.DoubleTensor) or isinstance(tensor, torch.cuda.DoubleTensor):
        dtype = numpy.float64
    else:
        raise ValueError("Only floating-point tensors are supported")

    rows, cols = tensor.size(0), tensor.size(1)
    num_zeros = int(math.ceil(rows * sparsity))

    n_tensor = numpy.random.normal(0, std, size=(rows, cols)).astype(dtype)
    zero_col_indices = numpy.empty((cols, num_zeros), dtype=numpy.int_, order='C')
    zero_row_indices = numpy.empty((cols, num_zeros), dtype=numpy.int_, order='C')
    for col_idx in range(cols):
        zero_col_indices[col_idx, :] = col_idx
        zero_row_indices[col_idx, :] = numpy.random.choice(rows, size=num_zeros, replace=False)
    n_tensor[zero_row_indices.ravel(order='C'), zero_col_indices.ravel(order='C')] = 0

    tensor.copy_(torch.from_numpy(n_tensor))
    return tensor
