# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import collections
import math
import numpy
import torch

from torch.autograd import Variable


def to_cpu(tensor):
    if tensor.is_cuda:
        return tensor.cpu()
    else:
        return tensor


# from https://stackoverflow.com/a/3233356
def recursive_dict_update(d, u):
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
