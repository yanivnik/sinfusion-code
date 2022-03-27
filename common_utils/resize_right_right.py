# from https://github.com/assafshocher/ResizeRight

from math import ceil
import common_utils.interp_methods_right as interp_methods

import torch


def resize(input,
           scale_factors=None,
           out_shape=None,
           interp_method=interp_methods.cubic,
           support_sz=None,
           antialiasing=True):
    # get properties of the input tensor
    in_shape, n_dims = input.shape, input.ndim

    # determined by the input type
    eps = torch.finfo(input.dtype).eps

    # set missing scale factors or output shapem one according to another,
    # scream if both missing
    scale_factors, out_shape = set_scale_and_out_sz(in_shape, out_shape, scale_factors)

    # sort indices of dimensions according to scale of each dimension.
    # since we are going dim by dim this is efficient
    sorted_filtered_dims_and_scales = [
        (dim, scale_factors[dim])
        for dim in sorted(range(n_dims), key=lambda ind: scale_factors[ind])
        if scale_factors[dim] != 1.
    ]

    # unless support size is specified by the user, it is an attribute
    # of the interpolation method
    if support_sz is None:
        support_sz = interp_method.support_sz

    # output begins identical to input and changes with each iteration
    output = input
    dtype, device = input.dtype, input.device

    # iterate over dims
    for dim, scale_factor in sorted_filtered_dims_and_scales:
        # get 1d set of weights and fields of view for each output location
        # along this dim
        field_of_view, weights = prepare_weights_and_field_of_view_1d(
            dim, scale_factor, in_shape[dim], out_shape[dim], interp_method,
            support_sz, antialiasing, eps, dtype=dtype, device=device)

        # multiply the weights by the values in the field of view and
        # aggreagate
        output = apply_weights(output, field_of_view, weights, dim, n_dims)
    return output


def prepare_weights_and_field_of_view_1d(dim,
                                         scale_factor,
                                         in_sz,
                                         out_sz,
                                         interp_method,
                                         support_sz,
                                         antialiasing,
                                         eps,
                                         dtype=None,
                                         device=None):
    # If antialiasing is taking place, we modify the window size and the
    # interpolation method (see inside function)
    interp_method, cur_support_sz = apply_antialiasing_if_needed(
        interp_method, support_sz, scale_factor, antialiasing)

    # STEP 1- PROJECTED GRID: The non-integer locations of the projection of
    # output pixel locations to the input tensor
    projected_grid = get_projected_grid(in_sz,
                                        out_sz,
                                        scale_factor,
                                        dtype=dtype,
                                        device=device)

    # STEP 2- FIELDS OF VIEW: for each output pixels, map the input pixels
    # that influence it
    field_of_view = get_field_of_view(projected_grid, cur_support_sz, in_sz,
                                      eps)

    # STEP 3- CALCULATE WEIGHTS: Match a set of weights to the pixels in the
    # field of view for each output pixel
    weights = get_weights(interp_method, projected_grid, field_of_view)

    return field_of_view, weights


def apply_weights(input, field_of_view, weights, dim, n_dims):
    # STEP 4- APPLY WEIGHTS: Each output pixel is calculated by multiplying
    # its set of weights with the pixel values in its field of view.
    # We now multiply the fields of view with their matching weights.
    # We do this by tensor multiplication and broadcasting.
    # this step is separated to a different function, so that it can be
    # repeated with the same calculated weights and fields.

    # for this operations we assume the resized dim is the first one.
    # so we transpose and will transpose back after multiplying
    tmp_input = input.transpose(dim, 0)
    # print(input.device, field_of_view.device, weights.device)

    # field_of_view is a tensor of order 2: for each output (1d location
    # along cur dim)- a list of 1d neighbors locations.
    # note that this whole operations is applied to each dim separately,
    # this is why it is all in 1d.
    # neighbors = tmp_input[field_of_view] is a tensor of order image_dims+1:
    # for each output pixel (this time indicated in all dims), these are the
    # values of the neighbors in the 1d field of view. note that we only
    # consider neighbors along the current dim, but such set exists for every
    # multi-dim location, hence the final tensor order is image_dims+1.
    neighbors = tmp_input[field_of_view]

    # weights is an order 2 tensor: for each output location along 1d- a list
    # of weighs matching the field of view. we augment it with ones, for
    # broadcasting, so that when multiplies some tensor the weights affect
    # only its first dim.
    tmp_weights = torch.reshape(weights, (*weights.shape, *[1] * (n_dims - 1)))

    # now we simply multiply the weights with the neighbors, and then sum
    # along the field of view, to get a single value per out pixel
    tmp_output = (neighbors * tmp_weights).sum(1)

    # we transpose back the resized dim to its original position
    return tmp_output.transpose(dim, 0)


def set_scale_and_out_sz(in_shape, out_shape, scale_factors):
    # eventually we must have both scale-factors and out-sizes for all in/out
    # dims. however, we support many possible partial arguments
    if scale_factors is None and out_shape is None:
        raise ValueError("either scale_factors or out_shape should be "
                         "provided")
    if out_shape is not None:
        # if out_shape has less dims than in_shape, we defaultly resize the
        # first dims for numpy and last dims for torch
        out_shape = list(in_shape[:-len(out_shape)]) + list(out_shape)
        if scale_factors is None:
            # if no scale given, we calculate it as the out to in ratio
            # (not recomended)
            scale_factors = [
                out_sz / in_sz for out_sz, in_sz in zip(out_shape, in_shape)
            ]
    if scale_factors is not None:
        # by default, if a single number is given as scale, we assume resizing
        # two dims (most common are images with 2 spatial dims)
        scale_factors = (scale_factors if isinstance(
            scale_factors, (list, tuple)) else [scale_factors, scale_factors])
        # if less scale_factors than in_shape dims, we defaultly resize the
        # first dims for numpy and last dims for torch
        scale_factors = (list(scale_factors) + [1] *
                         (len(in_shape) - len(scale_factors)) +
                         list(scale_factors))
        if out_shape is None:
            # when no out_shape given, it is calculated by multiplying the
            # scale by the in_shape (not recomended)
            out_shape = [
                ceil(scale_factor * in_sz)
                for scale_factor, in_sz in zip(scale_factors, in_shape)
            ]
        # next line intentionally after out_shape determined for stability
        scale_factors = [float(sf) for sf in scale_factors]
    return scale_factors, out_shape


def get_projected_grid(in_sz, out_sz, scale_factor, dtype=None, device=None):
    # we start by having the ouput coordinates which are just integer locations
    to = dict(dtype=dtype, device=device)
    out_coordinates = torch.arange(out_sz, **to)

    # This is projecting the ouput pixel locations in 1d to the input tensor,
    # as non-integer locations.
    # the following fomrula is derived in the paper
    # "From Discrete to Continuous Convolutions" by Shocher et al.
    return (out_coordinates / scale_factor + (in_sz - 1) / 2 - (out_sz - 1) /
            (2 * scale_factor))


def get_field_of_view(projected_grid, cur_support_sz, in_sz, eps):
    # for each output pixel, map which input pixels influence it, in 1d.
    # we start by calculating the leftmost neighbor, using half of the window
    # size (eps is for when boundary is exact int)
    to = dict(dtype=torch.long, device=projected_grid.device)
    left_boundaries = (projected_grid - cur_support_sz / 2 - eps).ceil_().long()

    # then we simply take all the pixel centers in the field by counting
    # window size pixels from the left boundary
    ordinal_numbers = torch.arange(ceil(cur_support_sz - eps), **to)
    # in case using torch we need to match the device
    field_of_view = left_boundaries[:, None] + ordinal_numbers

    # next we do a trick instead of padding, we map the field of view so that
    # it would be like mirror padding, without actually padding
    # (which would require enlarging the input tensor)
    mirror = torch.cat([
        torch.arange(in_sz, **to),
        torch.arange(in_sz - 1, -1, step=-1, **to)
    ])
    field_of_view = mirror[torch.remainder(field_of_view, mirror.shape[0])]
    return field_of_view


def get_weights(interp_method, projected_grid, field_of_view):
    # the set of weights per each output pixels is the result of the chosen
    # interpolation method applied to the distances between projected grid
    # locations and the pixel-centers in the field of view (distances are
    # directed, can be positive or negative)
    weights = interp_method(projected_grid[:, None] - field_of_view)

    # we now carefully normalize the weights to sum to 1 per each output pixel
    sum_weights = weights.sum(1, keepdims=True)
    sum_weights[sum_weights == 0] = 1
    return weights / sum_weights


def apply_antialiasing_if_needed(interp_method, support_sz, scale_factor,
                                 antialiasing):
    # antialiasing is "stretching" the field of view according to the scale
    # factor (only for downscaling). this is low-pass filtering. this
    # requires modifying both the interpolation (stretching the 1d
    # function and multiplying by the scale-factor) and the window size.
    if scale_factor >= 1.0 or not antialiasing:
        return interp_method, support_sz
    cur_interp_method = lambda arg: scale_factor * interp_method(scale_factor *
                                                                 arg)
    cur_support_sz = support_sz / scale_factor
    return cur_interp_method, cur_support_sz
