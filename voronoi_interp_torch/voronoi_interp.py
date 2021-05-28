import math
from torch import nn
from torch.autograd import Function
import torch
import os

from voronoi_interp_torch.cuda.voronoi_interp_cuda import voronoi_interp_forward as voronoi_interp_forward_cuda
from voronoi_interp_torch.cuda.voronoi_interp_cuda import voronoi_interp_backward as voronoi_interp_backward_cuda


class VoronoiInterpFunction(Function):
    @staticmethod
    def forward(ctx, coords, values, H, W):
        interp_image, nearest_idxs = voronoi_interp_forward_cuda(
            coords.contiguous(),
            values.contiguous(),
            H,
            W)
        ctx.save_for_backward(nearest_idxs, torch.tensor(coords.size(0)))

        return interp_image

    @staticmethod
    def backward(ctx, grad_output_image):
        nearest_idxs, n = ctx.saved_variables
        n = n.item()
        d_values = voronoi_interp_backward_cuda(grad_output_image.contiguous(), nearest_idxs.contiguous(), n)
        return None, d_values, None, None


def voronoi_interpolate(coords, values, H, W):
    return VoronoiInterpFunction.apply(coords, values, H, W)
