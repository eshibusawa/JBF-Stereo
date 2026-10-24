# This file is part of JBF-Stereo.
# Copyright (c) 2023-2026, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
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

import cv2
import numpy as np

import add_path
import cupy as cp
import cupyx.scipy.ndimage as ndi
import texture
from util_cuda import upload_constant

# References
# [1] Kuschk, G., & Cremers, D. (2013). Fast and accurate large-scale stereo reconstruction using variational methods.
#     In Proceedings of the IEEE International Conference on Computer Vision Workshops (pp. 700-707).
# [2] Yoon, K. J., & Kweon, I. S. (2006). Adaptive support-weight approach for correspondence search.
#     IEEE transactions on pattern analysis and machine intelligence, 28(4), 650-656.
# [3] Werlberger, M., Trobin, W., Pock, T., Wedel, A., Cremers, D., & Bischof, H. (2009).
#     Anisotropic Huber-L1 Optical Flow. In BMVC (Vol. 1, No. 2, p. 3).
# [4] Von Gioi, R. G., Jakubowicz, J., Morel, J. M., & Randall, G. (2008).
#     LSD: A fast line segment detector with a false detection control. IEEE transactions on pattern analysis and machine intelligence, 32(4), 722-732.

class TGV:
    def __init__(self, params):
        self.params = params
        self.gpu_module = None
        self.gpu_module_cub = None
        self.is_setuped = False

    def compile_module(self):
        self.census_length = 64
        self.census_window_size = 2 * self.params.census_radius + 1
        assert self.census_window_size * self.census_window_size <= self.census_length
        self.aggregation_window_size = 2 * self.params.aggregation_radius + 1
        self.range_max = np.iinfo(np.uint8).max
        self.tensor_eps = 1E-7
        self.grad_eps = 1E-7

        dn = os.path.dirname(__file__)
        dn_pm = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'PM')
        fnl = list()
        fnl.append(os.path.join(dn_pm, 'gradient.cu'))
        fnl.append(os.path.join(dn, 'tgv.cu'))

        cuda_source = None
        for fpfn in fnl:
            with open(fpfn, 'r') as f:
                cs = f.read()
            if cuda_source is None:
                cuda_source = cs
            else:
                cuda_source += cs

        cuda_source = cuda_source.replace('GRADIENT_PIXEL_SAMPLER', 'NormalSampler')
        cuda_source = cuda_source.replace('GRADIENT_PIXEL_TYPE', 'unsigned char')
        cuda_source = cuda_source.replace('GRADIENT_SAMPLER_FUNCTION', 'get')

        cuda_source = cuda_source.replace('TGV_CENSUS_RADIUS', str(self.params.census_radius))
        cuda_source = cuda_source.replace('TGV_MAX_DISPARITY', str(self.params.max_disparity))
        cuda_source = cuda_source.replace('TGV_INVALID_HD_COST', str(self.range_max))
        cuda_source = cuda_source.replace('TGV_RANGE_MAX', str(self.range_max))
        cuda_source = cuda_source.replace('TGV_AGGREGATION_RADIUS', str(self.params.aggregation_radius))
        cuda_source = cuda_source.replace('TGV_AGGREGATION_WINDOW_SIZE', str(self.aggregation_window_size))
        cuda_source = cuda_source.replace('TGV_AGGREGATION_K', str(self.params.aggregation_k))
        cuda_source = cuda_source.replace('TGV_TENSOR_A', str(self.params.tensor_a))
        cuda_source = cuda_source.replace('TGV_TENSOR_B', str(self.params.tensor_b))
        cuda_source = cuda_source.replace('TGV_TENSOR_EPS', str(self.tensor_eps))
        cuda_source = cuda_source.replace('TGV_GRAD_EPS', str(self.grad_eps))
        cuda_source = cuda_source.replace('TGV_TAU_U', str(self.params.tau_u))
        cuda_source = cuda_source.replace('TGV_TAU_V', str(self.params.tau_v))
        cuda_source = cuda_source.replace('TGV_TAU_P', str(self.params.tau_p))
        cuda_source = cuda_source.replace('TGV_TAU_Q', str(self.params.tau_q))
        cuda_source = cuda_source.replace('TGV_LAMBDA_S', str(self.params.lambda_s))
        cuda_source = cuda_source.replace('TGV_LAMBDA_A', str(self.params.lambda_a))

        self.gpu_module = cp.RawModule(code=cuda_source)
        self.gpu_module.compile()

    def compile_module_cub(self):
        dn = os.path.dirname(__file__)
        fnl = list()
        fnl.append(os.path.join(dn, 'tgv_cub.cu'))

        cuda_source = None
        for fpfn in fnl:
            with open(fpfn, 'r') as f:
                cs = f.read()
            if cuda_source is None:
                cuda_source = cs
            else:
                cuda_source += cs

        cuda_source = cuda_source.replace('TGV_CENSUS_LENGTH', str(self.census_length))
        cuda_source = cuda_source.replace('TGV_MAX_DISPARITY', str(self.params.max_disparity))
        cuda_source = cuda_source.replace('TGV_LAMBDA_D', str(self.params.lambda_d))

        self.gpu_module_cub = cp.RawModule(code=cuda_source)
        self.gpu_module_cub.compile()

    def setup_module(self):
        if self.gpu_module is None:
            self.compile_module()
        if self.gpu_module_cub is None:
            self.compile_module_cub()
        if self.is_setuped:
            return

        # compute / upload range kernel
        rk = np.exp(-np.arange(0, self.range_max, dtype=np.float32) / self.params.aggregation_gamma_c)
        upload_constant(self.gpu_module, rk, 'g_LASWRange', dtype=cp.float32)

        # compute / upload spatial kernel
        sd = np.arange(-self.params.aggregation_radius, self.params.aggregation_radius + 1, dtype=np.float32)
        sk = np.empty((sd.shape[0], sd.shape[0], 2), dtype=np.float32)
        sk[:,:,0] = sd[:,None]
        sk[:,:,1] = sd[None,:]
        sk = np.exp(-np.linalg.norm(sk, axis=2) / self.params.aggregation_gamma_p)
        upload_constant(self.gpu_module, sk.reshape(-1), 'g_LASWSpace', dtype=cp.float32)

        self.is_setuped = True

    def compute_census_transforms(self, img_ref, img_other):
        self.setup_module()

        # upload images and create texture object
        self.img_ref = cp.asarray(img_ref)
        self.img_other = cp.asarray(img_other)
        if self.params.enable_flipud:
            self.img_ref = cp.flipud(self.img_ref)
            self.img_other = cp.flipud(self.img_other)
        self.to_ref = texture.create_texture_object(self.img_ref,
            addressMode = cp.cuda.runtime.cudaAddressModeBorder,
            filterMode = cp.cuda.runtime.cudaFilterModePoint,
            readMode = cp.cuda.runtime.cudaReadModeElementType)
        self.to_other = texture.create_texture_object(self.img_other,
            addressMode = cp.cuda.runtime.cudaAddressModeBorder,
            filterMode = cp.cuda.runtime.cudaFilterModePoint,
            readMode = cp.cuda.runtime.cudaReadModeElementType)

        self.census_ref = cp.empty(self.img_ref.shape, dtype=np.int64)
        self.census_other = cp.empty(self.img_other.shape, dtype=np.int64)
        assert self.census_ref.flags.c_contiguous
        assert self.census_other.flags.c_contiguous

        gpu_func = self.gpu_module.get_function('computeCensusTransform')
        sz_block = 32, 32
        sz_grid = math.ceil(self.img_ref.shape[1] / sz_block[0]), math.ceil(self.img_ref.shape[0] / sz_block[1])
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                self.census_ref,
                self.to_ref,
                self.img_ref.shape[0],
                self.img_ref.shape[1]
            )
        )

        sz_block = 32, 32
        sz_grid = math.ceil(self.img_other.shape[1] / sz_block[0]), math.ceil(self.img_other.shape[0] / sz_block[1])
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                self.census_other,
                self.to_other,
                self.img_other.shape[0],
                self.img_other.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()

    def compute_hamming_distance(self):
        cost_shape = self.img_ref.shape[0], self.img_ref.shape[1], self.params.max_disparity
        self.hamming_distance = cp.empty(cost_shape, dtype=np.uint8)
        assert self.hamming_distance.flags.c_contiguous
        assert self.hamming_distance.size == cost_shape[0] * cost_shape[1] * cost_shape[2]

        gpu_func = self.gpu_module.get_function('computeHammingDistance')
        sz_block = 1024, 1
        sz_grid = math.ceil(self.hamming_distance.size / sz_block[0]), sz_block[1]
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                self.hamming_distance,
                self.census_ref,
                self.census_other,
                self.hamming_distance.shape[1],
                self.hamming_distance.size
            )
        )
        cp.cuda.runtime.deviceSynchronize()

    def aggregate_cost_LASW(self):
        self.aggregated_cost = cp.empty(self.hamming_distance.shape, dtype=np.float32)
        assert self.aggregated_cost.flags.c_contiguous
        assert self.aggregated_cost.size == self.aggregated_cost.shape[0] * self.aggregated_cost.shape[1] * self.aggregated_cost.shape[2]

        gpu_func = self.gpu_module.get_function('aggregateCostLASW')
        sz_block = 1024, 1
        sz_grid = math.ceil(self.aggregated_cost.size / sz_block[0]), sz_block[1]
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                self.aggregated_cost,
                self.to_ref,
                self.to_other,
                self.hamming_distance,
                self.aggregated_cost.shape[0],
                self.aggregated_cost.shape[1],
                self.aggregated_cost.size
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        self.d_LASW = cp.argmin(self.aggregated_cost, axis=2).astype(cp.int32)

    def compute_tensor(self, to_image):
        """The anisotropic diffusion tensor (xx, xy, yy) of Eq. 2 [1] of the image of the texture object to_image."""
        tensor = cp.empty((*self.img_ref.shape, 3), dtype=np.float32)
        assert tensor.flags.c_contiguous

        gpu_func = self.gpu_module.get_function('computeAnisotropicDiffusionTensor')
        sz_block = 32, 32
        sz_grid = math.ceil(self.img_ref.shape[1] / sz_block[0]), math.ceil(self.img_ref.shape[0] / sz_block[1])
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                tensor,
                to_image,
                self.img_ref.shape[0],
                self.img_ref.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        return tensor

    def compute_anisotropic_diffusion_tensor(self):
        # the gradient of the tensor is taken on a Gaussian smoothed image ([1]: the spatial context by a prior Gaussian convolution)
        to_tensor = self.to_ref
        if self.params.tensor_gradient_sigma > 0:
            smoothed = ndi.gaussian_filter(self.img_ref.astype(cp.float32), self.params.tensor_gradient_sigma, mode='nearest')
            smoothed = cp.ascontiguousarray(cp.clip(cp.rint(smoothed), 0, self.range_max).astype(cp.uint8))
            to_tensor = texture.create_texture_object(smoothed,
                addressMode = cp.cuda.runtime.cudaAddressModeBorder,
                filterMode = cp.cuda.runtime.cudaFilterModePoint,
                readMode = cp.cuda.runtime.cudaReadModeElementType)
        self.var_G = self.compute_tensor(to_tensor)

        if self.params.enable_lsd:
            self.replace_tensor_by_segments(self.compute_segment_mask())

    def compute_segment_mask(self):
        """Binary mask (255, uint8, in the orientation of the tensor) of the line segments of the LSD [4] of the reference image that are
        at least lsd_min_length long. The LSD runs on the CPU (OpenCV)."""
        img = self.img_ref.get()
        if self.params.enable_flipud:
            img = np.flipud(img)
        lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(np.ascontiguousarray(img))[0]
        mask = np.zeros(img.shape, dtype=np.uint8)
        if lines is not None:
            lines = lines.reshape(-1, 4)
            lengths = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
            for x1, y1, x2, y2 in lines[lengths >= self.params.lsd_min_length]:
                cv2.line(mask, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), 255, 1)
        if self.params.enable_flipud:
            mask = np.flipud(mask)
        return mask

    def replace_tensor_by_segments(self, mask):
        """Edge-segment based adaptive regularization [1]: the tensor of the Gaussian smoothed binary mask of the segments (the same
        computation as Eq. 2 with the mask as the image) replaces the tensor of the image where the smoothed mask is above
        lsd_band_threshold. The gradient of the smoothed thin line vanishes on the line, so the replaced region is a band around it."""
        smoothed = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), self.params.lsd_sigma)
        band = smoothed > self.params.lsd_band_threshold
        if self.params.lsd_normalize_mask and smoothed.max() > 0:
            smoothed = smoothed / smoothed.max()
        img = cp.ascontiguousarray(cp.asarray(np.clip(np.rint(self.range_max * smoothed), 0, self.range_max).astype(np.uint8)))
        to_mask = texture.create_texture_object(img,
            addressMode = cp.cuda.runtime.cudaAddressModeBorder,
            filterMode = cp.cuda.runtime.cudaFilterModePoint,
            readMode = cp.cuda.runtime.cudaReadModeElementType)
        tensor = self.compute_tensor(to_mask)
        self.var_G = cp.where(cp.asarray(band)[..., None], tensor, self.var_G)

    def initialize_primal_and_dual_variables(self):
        # primal variable u, auxiliary variable a in Eq. 4 [1]
        self.var_u = self.d_LASW.reshape(-1).astype(cp.float32) / cp.float32(self.params.max_disparity)
        self.var_u_ = cp.copy(self.var_u)
        self.var_a = cp.copy(self.var_u)
        # primal variable v (vector field)
        self.var_v = cp.zeros((self.var_u.size, 2), dtype=cp.float32)
        self.var_v_ = cp.zeros(self.var_v.shape, dtype=cp.float32)
        # dual variable p, q
        self.var_p = cp.zeros((self.var_u.size, 2), dtype=cp.float32)
        self.var_q = cp.zeros((self.var_u.size, 3), dtype=cp.float32)
        # Lagrange multiplier
        self.var_L = cp.zeros((self.var_u.size), dtype=cp.float32)
        # theta
        self.var_theta = 1

    def update_dual_variables(self):
        # update dual variable p, q from fixed u, v, a, L
        # gradient ascents
        assert self.var_p.flags.c_contiguous
        assert self.var_q.flags.c_contiguous
        assert self.var_u_.flags.c_contiguous
        assert self.var_v_.flags.c_contiguous

        gpu_func = self.gpu_module.get_function('updateDualVariables')
        sz_block = 1024, 1
        sz_grid = math.ceil(self.var_u.size / sz_block[0]), sz_block[1]
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                self.var_p,
                self.var_q,
                self.var_u_,
                self.var_v_,
                self.var_G,
                self.img_ref.shape[0],
                self.img_ref.shape[1],
                self.img_ref.size
            )
        )
        cp.cuda.runtime.deviceSynchronize()

    def update_primal_variables(self):
        # update primal variable u, v from fixed p, q, a, L
        # gradient descents
        assert self.var_p.flags.c_contiguous
        assert self.var_q.flags.c_contiguous
        assert self.var_u.flags.c_contiguous
        assert self.var_u_.flags.c_contiguous
        assert self.var_v.flags.c_contiguous
        assert self.var_v_.flags.c_contiguous
        assert self.var_a.flags.c_contiguous
        assert self.var_L.flags.c_contiguous

        gpu_func = self.gpu_module.get_function('updatePrimalVariables')
        sz_block = 1024, 1
        sz_grid = math.ceil(self.var_u.size / sz_block[0]), sz_block[1]
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                self.var_u,
                self.var_u_,
                self.var_v,
                self.var_v_,
                cp.float32(self.var_theta),
                self.var_p,
                self.var_q,
                self.var_a,
                self.var_L,
                self.var_G,
                self.img_ref.shape[0],
                self.img_ref.shape[1],
                self.img_ref.size
            )
        )
        cp.cuda.runtime.deviceSynchronize()

    def update_auxiliary_variable(self):
        # update auxiliary variable a, from fixed p, q, u, v
        # exhaustive search of regularized cost function
        assert self.var_a.flags.c_contiguous
        assert self.var_L.flags.c_contiguous
        assert self.aggregated_cost.flags.c_contiguous
        assert self.var_u.flags.c_contiguous

        gpu_func = self.gpu_module_cub.get_function('updateAuxiliaryVariable')
        sz_block = self.params.max_disparity, 1
        sz_grid = self.img_ref.shape[1], self.img_ref.shape[0]
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                self.var_a,
                self.var_L,
                cp.float32(self.var_theta),
                self.aggregated_cost,
                self.var_u,
                self.img_ref.shape[0],
                self.img_ref.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()

    def process(self, img_ref, img_other):
        # compute initial disparity
        self.compute_census_transforms(img_ref, img_other)
        self.compute_hamming_distance()
        self.aggregate_cost_LASW()
        self.compute_anisotropic_diffusion_tensor()

        # initialize TGV
        self.initialize_primal_and_dual_variables()
        # iteration
        for n in range(self.params.max_iteration):
            # smooth iteration
            for i in range(self.params.max_smooth_iteration):
                self.update_dual_variables()
                self.update_primal_variables()
            self.update_auxiliary_variable()
            # update theta
            self.var_theta *= (1 - self.params.beta * n)

        # compute disparity
        self.d_TGV = self.var_u.reshape(self.img_ref.shape) * cp.float32(self.params.max_disparity)

    def get_disparity(self, method):
        if method == 'LASW':
            d = self.d_LASW
        elif method == 'TGV':
            d = self.d_TGV
        else:
            return None

        if self.params.enable_flipud:
            d = cp.flipud(d)

        return d.get()
