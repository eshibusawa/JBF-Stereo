# This file is part of JBF-Stereo.
# Copyright (c) 2026, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
import os

import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as ndi

import add_path
from util_cuda import upload_constant

# References
# [1] Hu, Y., Song, R., & Li, Y. (2016). Efficient coarse-to-fine patchmatch for large displacement optical flow.
#     In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 5704-5712).

class SIFT:
    def __init__(self, params):
        self.gpu_module = None
        self.band = None
        self.params = params

    @staticmethod
    def sobel(img, axis):
        # ndi.sobel of every channel on its own (not normalized, a 3-D input would be smoothed along the channels, too)
        if len(img.shape) != 3:
            return None
        img_out = cp.empty_like(img)
        for k in range(3):
            img_out[:,:,k] = ndi.sobel(img[:,:,k], axis=axis, mode='nearest')
        return img_out

    def compile_module(self):
        dn = os.path.dirname(__file__)
        fnl = list()
        fnl.append(os.path.join(dn, 'sift.cu'))

        cuda_source = None
        for fpfn in fnl:
            with open(fpfn, 'r') as f:
                cs = f.read()
            if cuda_source is None:
                cuda_source = cs
            else:
                cuda_source += cs

        cuda_source = cuda_source.replace('SIFT_GRAD_EPS', str(self.params.grad_eps))
        cuda_source = cuda_source.replace('SIFT_MAGNITUDE_EPS', str(self.params.magnitude_eps))
        cuda_source = cuda_source.replace('SIFT_NBINS', str(self.params.sift_bins))
        cuda_source = cuda_source.replace('SIFT_ALPHA', str(self.params.sift_alpha))
        cuda_source = cuda_source.replace('SIFT_DIM', str(16 * self.params.sift_bins))
        assert self.params.sift_step_size == 1, 'computeSIFT outputs one descriptor per pixel'
        cuda_source = cuda_source.replace('SIFT_STEPSIZE', str(self.params.sift_step_size))
        cuda_source = cuda_source.replace('SIFT_CELLSIZE', str(self.params.sift_cell_size))
        if self.params.sift_is_boundary_included:
            cuda_source = cuda_source.replace('SIFT_XSIFT', str(0))
            cuda_source = cuda_source.replace('SIFT_YSIFT', str(0))
        else:
            cuda_source = cuda_source.replace('SIFT_XSIFT', str(2 * self.params.sift_cell_size))
            cuda_source = cuda_source.replace('SIFT_YSIFT', str(2 * self.params.sift_cell_size))

        self.gpu_module = cp.RawModule(code=cuda_source)
        self.gpu_module.compile()

    def setup_module(self):
        if self.gpu_module is None:
            self.compile_module()
        # upload constant etc.
        thetas = np.arange(self.params.sift_bins, dtype=np.float32) * 2 * np.float32(np.pi /self.params.sift_bins)
        upload_constant(self.gpu_module, np.cos(thetas), 'g_SIFTCos')
        upload_constant(self.gpu_module, np.sin(thetas), 'g_SIFTSin')

        # compute filter
        cell_size = self.params.sift_cell_size
        filter = np.zeros(cell_size * 2 + 1, dtype=np.float32)
        filter[0] = filter[cell_size+2] = 0.25
        filter[2:cell_size+2] = 1
        filter[cell_size+2:] = 0
        self.filter = filter
        self.filter_gpu = cp.array(filter[::-1]) # for convolve1d

    def compute_gradient(self, dx, dy):
        if len(dx.shape) != 3 or dx.shape[2] != 3 or dx.dtype != cp.float32:
            return None
        if dy.shape != dx.shape or dx.dtype != dy.dtype != dx.dtype:
            return None

        d = cp.empty_like(dx)
        assert dx.flags.c_contiguous
        assert dy.flags.c_contiguous
        assert d.flags.c_contiguous

        gpu_func = self.gpu_module.get_function('computeGradient')
        sz_block = 1024, 1
        sz_grid = math.ceil(d.shape[0] * d.shape[1] / sz_block[0]), 1
        gpu_func(
            block=sz_block, grid=sz_grid,
            args=(
                d,
                dx,
                dy,
                d.shape[0] * d.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        return d

    def compute_band(self, grad):
        if len(grad.shape) != 3 or grad.shape[2] != 3 or grad.dtype != cp.float32:
            return None

        d = cp.empty((grad.shape[0], grad.shape[1], self.params.sift_bins), dtype=cp.float32)
        assert grad.flags.c_contiguous
        assert d.flags.c_contiguous

        gpu_func = self.gpu_module.get_function('computeBand')
        sz_block = 8, 128
        sz_grid = math.ceil(d.shape[2] / sz_block[0]), math.ceil(d.shape[0] * d.shape[1] / sz_block[1])
        gpu_func(
            block=sz_block, grid=sz_grid,
            args=(
                d,
                grad,
                d.shape[0] * d.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        self.band = d
        return d

    def filter_band(self):
        if self.band is None:
            return None

        band_filtered = ndi.convolve1d(self.band, self.filter_gpu, axis=1, mode="nearest")
        self.band = ndi.convolve1d(band_filtered, self.filter_gpu, axis=0, mode="nearest")
        return self.band

    def compute_SIFT(self):
        if self.band is None:
            return None

        if self.params.sift_is_boundary_included:
            sift_shape = (self.band.shape[0] // self.params.sift_step_size, self.band.shape[1] // self.params.sift_step_size, 16*self.params.sift_bins)
        else:
            sift_shape = ((self.band.shape[0] - 4*self.params.sift_cell_size) // self.params.sift_step_size, (self.band.shape[1] - 4*self.params.sift_cell_size) // self.params.sift_step_size, 16*self.params.sift_bins)

        d = cp.empty(sift_shape, dtype=cp.uint8)
        assert self.band.flags.c_contiguous
        assert d.flags.c_contiguous
        gpu_func = self.gpu_module.get_function('computeSIFT')
        sz_block = 4, 4
        sz_grid = d.shape[1], d.shape[0]
        gpu_func(
            block=sz_block, grid=sz_grid,
            args=(
                d,
                self.band,
                np.int32(self.band.shape[1]),
                np.int32(self.band.shape[0])
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        return d
