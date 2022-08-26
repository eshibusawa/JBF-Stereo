# This file is part of JBF-Stereo.
# Copyright (c) 2022, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
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

import cupy as cp

import add_path
import texture

class patch_match_stereo():
    def __init__(self):
        self.patch_width = 11
        self.patch_height = 11
        self.max_disparity = 128
        self.min_disparity = 0
        self.disparity_range_penalty = 10
        self.weight_gamma = 10
        self.blending_alpha = 0.9
        self.truncate_color = 10
        self.truncate_grad = 2
        self.spatial_delta = 5
        self.enable_consistent_gradient_operator = True
        self.enable_half_pixel_shift = True
        self.enable_red_iteration = True
        self.gpu_module = None

    def compile_module(self):
        assert (self.patch_width % 2) == 1
        assert (self.patch_height % 2) == 1

        dn = os.path.dirname(__file__)
        fnl = list()
        fnl.append(os.path.join(dn, 'rand_mls.cuh'))
        fnl.append(os.path.join(dn, 'patch_match.cu'))

        cuda_source = None
        for fpfn in fnl:
            with open(fpfn, 'r') as f:
                cs = f.read()
            if cuda_source is None:
                cuda_source = cs
            else:
                cuda_source += cs

        cuda_source = cuda_source.replace('PM_PATCH_WIDTH', str(self.patch_width))
        cuda_source = cuda_source.replace('PM_PATCH_HEIGHT', str(self.patch_height))
        cuda_source = cuda_source.replace('PM_PATCH_RADIUS_H', str(self.patch_width//2))
        cuda_source = cuda_source.replace('PM_PATCH_RADIUS_V', str(self.patch_height//2))

        cuda_source = cuda_source.replace('PM_MAX_DISPARITY', str(self.max_disparity))
        cuda_source = cuda_source.replace('PM_MIN_DISPARITY', str(self.min_disparity))
        cuda_source = cuda_source.replace('PM_DISPARITY_RANGE_PENALTY', str(self.disparity_range_penalty))

        cuda_source = cuda_source.replace('PM_WEIGHT_GAMMA', str(self.weight_gamma))
        cuda_source = cuda_source.replace('PM_BLENDING_ALPHA', str(self.blending_alpha))
        cuda_source = cuda_source.replace('PM_TRUNCATE_COLOR', str(self.truncate_color))
        cuda_source = cuda_source.replace('PM_TRUNCATE_GRAD', str(self.truncate_grad))

        cuda_source = cuda_source.replace('PM_SPATIAL_DELTA', str(self.spatial_delta))

        if self.enable_half_pixel_shift:
            options = '-DPM_ENABLE_HALF_PIXEL_SHIFT',
            self.gpu_module = cp.RawModule(code=cuda_source, options=options)
        else:
            self.gpu_module = cp.RawModule(code=cuda_source)
        self.gpu_module.compile()

    def compute_gradient(self):
        self.grad_ref = cp.empty((self.img_ref.shape[0], self.img_ref.shape[1], 2), dtype=cp.float32)
        self.grad_other = cp.empty((self.img_other.shape[0], self.img_other.shape[1], 2), dtype=cp.float32)

        gpu_func = None
        if self.enable_consistent_gradient_operator:
            gpu_func = self.gpu_module.get_function('computeConsistentGradient')
        else:
            gpu_func = self.gpu_module.get_function('computeSobelGradient')

        sz_block = 32, 32
        sz_grid = math.ceil(self.img_ref.shape[1] / sz_block[0]), math.ceil(self.img_ref.shape[0] / sz_block[1])
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                self.grad_ref,
                self.to_ref,
                self.img_ref.shape[0],
                self.img_ref.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()

        sz_grid = math.ceil(self.img_other.shape[1] / sz_block[0]), math.ceil(self.img_other.shape[0] / sz_block[1])
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                self.grad_other,
                self.to_other,
                self.img_other.shape[0],
                self.img_other.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()

        self.to_grad_ref = texture.create_texture_object(self.grad_ref,
            addressMode = cp.cuda.runtime.cudaAddressModeBorder,
            filterMode = cp.cuda.runtime.cudaFilterModeLinear,
            readMode = cp.cuda.runtime.cudaReadModeElementType)
        self.to_grad_other = texture.create_texture_object(self.grad_other,
            addressMode = cp.cuda.runtime.cudaAddressModeBorder,
            filterMode = cp.cuda.runtime.cudaFilterModeLinear,
            readMode = cp.cuda.runtime.cudaReadModeElementType)

    def initialize_planes(self, img_ref, img_other):
        if self.gpu_module is None:
            self.compile_module()

        # upload images and create texture object
        self.img_ref = cp.asarray(img_ref)
        self.img_other = cp.asarray(img_other)
        self.to_ref = texture.create_texture_object(self.img_ref,
            addressMode = cp.cuda.runtime.cudaAddressModeBorder,
            filterMode = cp.cuda.runtime.cudaFilterModeLinear,
            readMode = cp.cuda.runtime.cudaReadModeNormalizedFloat)
        self.to_other = texture.create_texture_object(self.img_other,
            addressMode = cp.cuda.runtime.cudaAddressModeBorder,
            filterMode = cp.cuda.runtime.cudaFilterModeLinear,
            readMode = cp.cuda.runtime.cudaReadModeNormalizedFloat)

        # compute gradient
        self.compute_gradient()

        # initialize planes, cost, random state
        self.random_state = cp.random.randint(0, 2 ** 63, (self.img_ref.shape[0], self.img_ref.shape[1]), dtype=cp.uint64)
        self.planes = cp.empty((self.img_ref.shape[0], self.img_ref.shape[1], 4), dtype=cp.float32)
        self.cost = cp.empty((self.img_ref.shape[0], self.img_ref.shape[1]), dtype=cp.float32)

        assert self.random_state.flags.c_contiguous
        assert self.planes.flags.c_contiguous
        assert self.cost.flags.c_contiguous

        gpu_func = self.gpu_module.get_function('getInitialPlanesAndCosts')
        sz_block = 32, 32
        sz_grid = math.ceil(self.img_ref.shape[1] / sz_block[0]), math.ceil(self.img_ref.shape[0] / sz_block[1])
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                self.planes,
                self.cost,
                self.random_state,
                self.to_ref,
                self.to_other,
                self.to_grad_ref,
                self.to_grad_other,
                self.img_ref.shape[0],
                self.img_ref.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()

    def toggle_red_and_black(self):
        self.enable_red_iteration = not self.enable_red_iteration

    def compute_spatial_propergation(self):
        gpu_func = None
        if (self.enable_red_iteration):
            gpu_func = self.gpu_module.get_function('computeRedSpatialPropagation')
        else:
            gpu_func = self.gpu_module.get_function('computeBlackSpatialPropagation')

        sz_block = 64, 16
        sz_grid = math.ceil(self.img_ref.shape[1] / sz_block[0]), math.ceil(self.img_ref.shape[0] / 2 / sz_block[1])
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                self.planes,
                self.cost,
                self.to_ref,
                self.to_other,
                self.to_grad_ref,
                self.to_grad_other,
                self.img_ref.shape[0],
                self.img_ref.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()

    def compute_random_search(self):
        gpu_func = None
        if (self.enable_red_iteration):
            gpu_func = self.gpu_module.get_function('computeRedRandomSearch')
        else:
            gpu_func = self.gpu_module.get_function('computeBlackRandomSearch')

        sz_block = 64, 16
        sz_grid = math.ceil(self.img_ref.shape[1] / sz_block[0]), math.ceil(self.img_ref.shape[0] / 2 / sz_block[1])
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                self.planes,
                self.cost,
                self.random_state,
                self.to_ref,
                self.to_other,
                self.to_grad_ref,
                self.to_grad_other,
                self.img_ref.shape[0],
                self.img_ref.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()

    def compute_disparity(self):
        disparity_gpu = cp.empty((self.img_ref.shape[0], self.img_ref.shape[1]), dtype=cp.float32)
        gpu_func = self.gpu_module.get_function('computeDisparity')
        sz_block = 32, 32
        sz_grid = math.ceil(self.img_ref.shape[1] / sz_block[0]), math.ceil(self.img_ref.shape[0] / sz_block[1])
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                disparity_gpu,
                self.planes,
                self.img_ref.shape[0],
                self.img_ref.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        return disparity_gpu
