# This file is part of JBF-Stereo.
# Copyright (c) 2023, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
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
import numpy as np
from scipy.spatial import Delaunay

import add_path
import texture
from util_cuda import upload_constant
import ccl

class ELAS():
    def __init__(self, param):
        assert param.disp_min >= 0
        assert not param.subsampling, 'not implemented yet'

        self.param = param
        self.gpu_module = None
        self.descriptor_length = 16
        self.descriptor_padding = 3
        self.support_matches = None
        self.disparity_ref = None
        self.disparity_other = None

    def compile_module(self):
        dn = os.path.dirname(__file__)
        fnl = list()
        fnl.append(os.path.join(dn, 'elas.cu'))

        cuda_source = None
        for fpfn in fnl:
            with open(fpfn, 'r') as f:
                cs = f.read()
            if cuda_source is None:
                cuda_source = cs
            else:
                cuda_source += cs

        # precompute parameters
        self.plane_radius = int(max(math.ceil(self.param.sigma * self.param.sradius), 2.0))
        self.invalidated_disparity = -10

        cuda_source = cuda_source.replace('ELAS_DISPARITY_MAX', str(self.param.disp_max))
        cuda_source = cuda_source.replace('ELAS_DISPARITY_MIN', str(self.param.disp_min))
        cuda_source = cuda_source.replace('ELAS_INVALID_DISPARITY', str(-1))
        cuda_source = cuda_source.replace('ELAS_INVALIDATED_DISPARITY', str(self.invalidated_disparity))
        cuda_source = cuda_source.replace('ELAS_CANDIDATE_STEPSIZE', str(self.param.candidate_stepsize))
        cuda_source = cuda_source.replace('ELAS_SUPPORT_TEXTURE', str(self.param.support_texture))
        cuda_source = cuda_source.replace('ELAS_SUPPORT_THRESHOLD', str(self.param.support_threshold))
        cuda_source = cuda_source.replace('ELAS_LR_THRESHOLD', str(self.param.lr_threshold))
        cuda_source = cuda_source.replace('ELAS_INCON_WINDOW_SIZE', str(self.param.incon_window_size))
        cuda_source = cuda_source.replace('ELAS_INCON_THRESHOLD', str(self.param.incon_threshold))
        cuda_source = cuda_source.replace('ELAS_INCON_MIN_SUPPORT', str(self.param.incon_min_support))
        cuda_source = cuda_source.replace('ELAS_MAX_REDUN_DIST', str(5))
        cuda_source = cuda_source.replace('ELAS_MAX_REDUN_THRESHOLD', str(1))
        cuda_source = cuda_source.replace('ELAS_DESCRIPTOR_LENGTH', str(self.descriptor_length))
        cuda_source = cuda_source.replace('ELAS_DESCRIPTOR_PADDING', str(self.descriptor_padding))
        cuda_source = cuda_source.replace('ELAS_U_STEP', str(2))
        cuda_source = cuda_source.replace('ELAS_V_STEP', str(2))
        cuda_source = cuda_source.replace('ELAS_WINDOW_SIZE', str(3))
        cuda_source = cuda_source.replace('ELAS_GRID_SIZE', str(self.param.grid_size))
        cuda_source = cuda_source.replace('ELAS_PLANE_RADIUS', str(self.plane_radius))
        cuda_source = cuda_source.replace('ELAS_MATCH_TEXTURE', str(self.param.match_texture))
        cuda_source = cuda_source.replace('ELAS_LR_THRESHOLD', str(self.param.lr_threshold))
        cuda_source = cuda_source.replace('ELAS_IPOL_GAP_WIDTH', str(self.param.ipol_gap_width))
        cuda_source = cuda_source.replace('ELAS_DISCON_THRESHOLD', str(3))
        cuda_source = cuda_source.replace('ELAS_HALF_MEAN_WINDOW_SIZE', str(8//2))
        cuda_source = cuda_source.replace('ELAS_MEDIAN_WINDOW_SIZE', str(7))

        options = list()
        if self.param.add_corners:
            options.append('-DELAS_ADD_CORNERS')

        if len(options) == 0:
            self.gpu_module = cp.RawModule(code=cuda_source)
        else:
            self.gpu_module = cp.RawModule(code=cuda_source, options=tuple(options))
        self.gpu_module.compile()

        self.ccl = ccl.CCL()
        self.ccl.compile_module()

    def setup_module(self):
        if self.gpu_module is None:
            self.compile_module()

        # precompute parameters
        d = np.arange(self.param.disp_min, self.param.disp_max + 1)
        s, g, b = self.param.sigma, self.param.gamma, self.param.beta
        self.prior = ((-np.log(g + np.exp(-d * d / 2 / s / s)) + np.log(g))/b).astype(np.int32)
        # upload prior
        upload_constant(self.gpu_module, self.prior, 'g_prior', dtype=cp.int32)

    def compute_gradient(self):
        self.grad_ref = cp.empty((self.img_ref.shape[0], self.img_ref.shape[1], 2), dtype=cp.uint8)
        self.grad_other = cp.empty((self.img_other.shape[0], self.img_other.shape[1], 2), dtype=cp.uint8)

        assert self.grad_ref.flags.c_contiguous
        assert self.grad_other.flags.c_contiguous

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

    def compute_descriptor(self):
        self.compute_gradient()
        self.descriptor_ref = cp.empty((self.img_ref.shape[0], self.img_ref.shape[1], self.descriptor_length), dtype=cp.uint8)
        self.descriptor_other = cp.empty((self.img_other.shape[0], self.img_other.shape[1], self.descriptor_length), dtype=cp.uint8)

        assert self.descriptor_ref.flags.c_contiguous
        assert self.descriptor_other.flags.c_contiguous

        gpu_func = self.gpu_module.get_function('computeDescriptor')
        sz_block = 32, 32
        wp = self.descriptor_ref.shape[1] - (self.descriptor_padding * 2)
        hp = self.descriptor_ref.shape[0] - (self.descriptor_padding * 2)
        sz_grid = math.ceil(wp / sz_block[0]), math.ceil(hp / sz_block[1])
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                self.descriptor_ref,
                self.grad_ref,
                self.img_ref.shape[0],
                self.img_ref.shape[1]
            )
        )

        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                self.descriptor_other,
                self.grad_other,
                self.img_other.shape[0],
                self.img_other.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()

    def compute_disparity_candidate(self):
        self.compute_gradient()
        wc = self.img_ref.shape[1] // self.param.candidate_stepsize
        hc = self.img_ref.shape[0] // self.param.candidate_stepsize

        disparity_candidate = cp.zeros((hc, wc), dtype=cp.int16)

        assert disparity_candidate.flags.c_contiguous

        gpu_func = self.gpu_module.get_function('computeSupportDisparityLR')
        sz_block = 32, 32
        wcp = disparity_candidate.shape[1] - 1
        hcp = disparity_candidate.shape[0] - 1
        sz_grid = math.ceil(wcp / sz_block[0]), math.ceil(hcp / sz_block[1])
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                disparity_candidate,
                self.descriptor_ref,
                self.descriptor_other,
                disparity_candidate.shape[0],
                disparity_candidate.shape[1],
                self.img_ref.shape[0],
                self.img_ref.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()

        disparity_candidate2 = cp.zeros_like(disparity_candidate)
        gpu_func = self.gpu_module.get_function('removeInconsistentSupportPoints')
        sz_block = 32, 32
        wc = disparity_candidate.shape[1]
        hc = disparity_candidate.shape[0]
        sz_grid = math.ceil(wc / sz_block[0]), math.ceil(hc / sz_block[1])
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                disparity_candidate2,
                disparity_candidate,
                disparity_candidate.shape[0],
                disparity_candidate.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        disparity_candidate = disparity_candidate2

        gpu_func = self.gpu_module.get_function('removeRedundantSupportPointsV')
        sz_block = 32, 1
        wc = disparity_candidate.shape[1]
        sz_grid = math.ceil(wc / sz_block[0]), 1
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                disparity_candidate,
                disparity_candidate.shape[0],
                disparity_candidate.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()

        gpu_func = self.gpu_module.get_function('removeRedundantSupportPointsH')
        sz_block = 32, 1
        hc = disparity_candidate.shape[0]
        sz_grid = math.ceil(hc / sz_block[0]), 1
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                disparity_candidate,
                disparity_candidate.shape[0],
                disparity_candidate.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        self.support_matches = disparity_candidate

    def compute_support_matches(self):
        if self.support_matches is None:
            self.compute_disparity_candidate()

        support_candidate = cp.zeros((self.support_matches.size + 6, 3), dtype=cp.int32)
        assert support_candidate.flags.c_contiguous
        support_candidate_non_corner = support_candidate[6:,:]
        assert support_candidate_non_corner.flags.c_contiguous

        gpu_func = self.gpu_module.get_function('computeSupportFromCandidate')
        sz_block = 32, 32
        wc, hc = self.support_matches.shape[1] - 1, self.support_matches.shape[0] - 1
        sz_grid = math.ceil(wc / sz_block[0]), math.ceil(hc / sz_block[1])
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                support_candidate_non_corner,
                self.support_matches,
                self.support_matches.shape[0],
                self.support_matches.shape[1],
                support_candidate_non_corner.shape[0]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        num_valid = int(support_candidate_non_corner[-1,0].get())
        self.support_matches = support_candidate_non_corner[:num_valid, :]

        if self.param.add_corners:
            h, w = self.img_ref.shape[0], self.img_ref.shape[1]
            work = cp.full((4,), -256, dtype=cp.uint64)
            support_candidate[:4,:] = cp.array([[0, 0, -1], [0, h - 1, -1], [w - 1, 0, -1], [w - 1, h - 1, -1]], dtype=cp.int32)
            gpu_func = self.gpu_module.get_function('addCornerToSupport')
            sz_block = 1024, 1
            sz_grid = math.ceil(num_valid / sz_block[0]), 1
            gpu_func(
                block=sz_block,
                grid=sz_grid,
                args=(
                    support_candidate,
                    work,
                    support_candidate_non_corner,
                    num_valid
                )
            )
            cp.cuda.runtime.deviceSynchronize()
            self.support_matches = support_candidate[:num_valid + 6, :]

    def compute_delaunay_triangulation(self):
        if self.support_matches is None:
            self.compute_support_matches()
        else:
            if self.support_matches.shape[1] != 3:
                self.compute_support_matches()
        support_matches = self.support_matches.get()
        self.triangulation_ref = Delaunay(support_matches[:,:2]).simplices
        support_matches[:,0] -= support_matches[:,2]
        self.triangulation_other = Delaunay(support_matches[:,:2]).simplices

    def compute_disparity_planes(self):
        if self.triangulation_ref is None:
            self.compute_delaunay_triangulation()

        assert self.triangulation_ref.shape == self.triangulation_other.shape
        self.triangulation_ref = cp.array(self.triangulation_ref)
        self.triangulation_other = cp.array(self.triangulation_other)
        triangulation_f = cp.empty((2, self.triangulation_ref.shape[0], 6), dtype=cp.float32)
        triangulation_f_ref = triangulation_f[0]
        triangulation_f_other = triangulation_f[1]
        assert triangulation_f_ref.flags.c_contiguous
        assert triangulation_f_other.flags.c_contiguous

        gpu_func = self.gpu_module.get_function('computeDisparityPlaneLR')
        sz_block = 1024, 1
        sz_grid = math.ceil(self.triangulation_ref.shape[0] / sz_block[0]), 1
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                triangulation_f_ref,
                triangulation_f_other,
                self.triangulation_ref,
                self.triangulation_other,
                self.support_matches,
                self.triangulation_ref.shape[0]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        self.triangulation_f_ref = triangulation_f_ref
        self.triangulation_f_other = triangulation_f_other

    def create_grid(self):
        if self.triangulation_ref is None:
            self.compute_delaunay_triangulation()

        grid_height = math.ceil(self.img_ref.shape[0] / self.param.grid_size)
        grid_width = math.ceil(self.img_ref.shape[1] / self.param.grid_size)
        grid_depth = self.param.disp_max + 1
        work = cp.zeros((2, grid_height, grid_width, grid_depth), dtype=cp.uint8)
        assert work.flags.c_contiguous

        gpu_func = self.gpu_module.get_function('createGridMaskLR')
        sz_block = 1024, 1
        sz_grid = math.ceil(self.support_matches.shape[0] / sz_block[0]), 1
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                work,
                self.support_matches,
                self.support_matches.shape[0],
                work.shape[1],
                work.shape[2],
                work.shape[3]
            )
        )
        cp.cuda.runtime.deviceSynchronize()

        grid = cp.zeros((2, grid_height, grid_width, grid_depth + 1), dtype=cp.int16)
        assert grid.flags.c_contiguous
        gpu_func = self.gpu_module.get_function('createGridLR')
        sz_block = 32, 32
        sz_grid = math.ceil(work.shape[2] / sz_block[0]), math.ceil(work.shape[1] / sz_block[1])
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                grid,
                work,
                work.shape[1],
                work.shape[2],
                work.shape[3]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        self.grid_ref = grid[0]
        self.grid_other = grid[1]

    def compute_target_pixels(self):
        if self.triangulation_f_ref is None:
            self.compute_disparity_planes()

        uvs = cp.empty((2, self.img_ref.size, 3), dtype=cp.int32)
        uvs_ref = uvs[0]
        uvs_other = uvs[1]
        assert uvs_ref.flags.c_contiguous
        assert uvs_other.flags.c_contiguous

        uv_counts = cp.zeros((2, self.triangulation_ref.shape[0] + 1), dtype=cp.int32)
        uv_counts_ref = uv_counts[0]
        uv_counts_other = uv_counts[1]
        assert uv_counts_ref.flags.c_contiguous
        assert uv_counts_other.flags.c_contiguous

        gpu_func = self.gpu_module.get_function('computeTargetPixelsLR')
        sz_block = 1024, 1
        sz_grid = math.ceil(self.triangulation_ref.shape[0] / sz_block[0]), 1
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                uvs_ref,
                uvs_other,
                uv_counts_ref,
                uv_counts_other,
                self.triangulation_ref,
                self.triangulation_other,
                self.support_matches,
                self.triangulation_ref.shape[0],
                self.img_ref.shape[0],
                self.img_ref.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        count = uv_counts[:,-1].get()
        assert count[0] < self.img_ref.size
        assert count[1] < self.img_other.size
        self.uvs_ref = uvs[0,:count[0], :]
        self.uvs_other = uvs[1,:count[1], :]
        assert uvs_ref.flags.c_contiguous
        assert uvs_other.flags.c_contiguous
        self.uv_counts_ref = uv_counts[0]
        self.uv_counts_other = uv_counts[1]

    def compute_disparity(self):
        if self.uvs_ref is None:
            self.compute_target_pixels()

        d_ref = cp.full(self.img_ref.shape, self.invalidated_disparity, dtype=cp.int16)
        d_other = cp.full(self.img_other.shape, self.invalidated_disparity, dtype=cp.int16)
        assert d_ref.flags.c_contiguous
        assert d_other.flags.c_contiguous

        gpu_func = self.gpu_module.get_function('findMatchL')
        sz_block = 1024, 1
        sz_grid = math.ceil(self.uvs_ref.shape[0] / sz_block[0]), 1
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                d_ref,
                self.descriptor_ref,
                self.descriptor_other,
                self.uvs_ref,
                self.grid_ref,
                self.triangulation_f_ref,
                self.uvs_ref.shape[0],
                d_ref.shape[1],
                self.grid_ref.shape[1],
                self.grid_other.shape[2]
            )
        )

        gpu_func = self.gpu_module.get_function('findMatchR')
        sz_block = 1024, 1
        sz_grid = math.ceil(self.uvs_other.shape[0] / sz_block[0]), 1
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                d_other,
                self.descriptor_ref,
                self.descriptor_other,
                self.uvs_other,
                self.grid_other,
                self.triangulation_f_other,
                self.uvs_other.shape[0],
                d_other.shape[1],
                self.grid_other.shape[1],
                self.grid_other.shape[2]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        self.disparity_ref = d_ref
        self.disparity_other = d_other

    def left_right_consistency_check(self):
        if self.disparity_ref is None:
            return None

        d_ref = cp.copy(self.disparity_ref)
        d_other = cp.copy(self.disparity_other)
        assert d_ref.flags.c_contiguous
        assert d_other.flags.c_contiguous

        gpu_func = self.gpu_module.get_function('leftRightConsistencyCheck')
        sz_block = 1024, 1
        sz_grid = math.ceil(self.disparity_ref.size / sz_block[0]), 1
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                d_ref,
                d_other,
                self.disparity_ref,
                self.disparity_other,
                self.disparity_ref.shape[0],
                self.disparity_ref.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        self.disparity_ref = d_ref
        self.disparity_other = d_other

    def remove_small_segments(self):
        if self.param.postprocess_only_left:
            l = self.disparity_ref,
        else:
            l = self.disparity_ref, self.disparity_other

        for d in l:
            counts = cp.zeros(d.size, dtype=cp.int32)
            assert counts.flags.c_contiguous

            label = self.ccl.labeling(d, self.param.speckle_sim_threshold)
            gpu_func = self.gpu_module.get_function('countLabels')
            sz_block = 1024, 1
            sz_grid = math.ceil(d.size / sz_block[0]), 1
            gpu_func(
                block=sz_block,
                grid=sz_grid,
                args=(
                    counts,
                    label,
                    d.shape[0],
                    d.shape[1]
                )
            )
            cp.cuda.runtime.deviceSynchronize()
            label = self.ccl.labeling(d, self.param.speckle_sim_threshold)
            gpu_func = self.gpu_module.get_function('removeSmallSegments')
            sz_block = 1024, 1
            sz_grid = math.ceil(d.size / sz_block[0]), 1
            gpu_func(
                block=sz_block,
                grid=sz_grid,
                args=(
                    d,
                    counts,
                    label,
                    self.param.speckle_size,
                    d.shape[0],
                    d.shape[1]
                )
            )
            cp.cuda.runtime.deviceSynchronize()


    def gap_interpolation(self):
        if self.param.postprocess_only_left:
            l = self.disparity_ref,
        else:
            l = self.disparity_ref, self.disparity_other

        dfs = list()
        for d in l:
            df = cp.array(d, dtype=cp.float32)
            assert df.flags.c_contiguous
            gpu_func = self.gpu_module.get_function('gapInterpolationH')
            sz_block = 1024, 1
            sz_grid = math.ceil(df.shape[0] / sz_block[0]), 1
            gpu_func(
                block=sz_block,
                grid=sz_grid,
                args=(
                    df,
                    df.shape[0],
                    df.shape[1]
                )
            )
            cp.cuda.runtime.deviceSynchronize()
            gpu_func = self.gpu_module.get_function('gapInterpolationV')
            sz_block = 1024, 1
            sz_grid = math.ceil(df.shape[1] / sz_block[0]), 1
            gpu_func(
                block=sz_block,
                grid=sz_grid,
                args=(
                    df,
                    df.shape[0],
                    df.shape[1]
                )
            )
            cp.cuda.runtime.deviceSynchronize()
            dfs.append(df)
        self.disparity_ref = dfs[0]
        if not self.param.postprocess_only_left:
            self.disparity_other = dfs[1]

    def adaptive_mean(self):
        if self.param.postprocess_only_left:
            l = self.disparity_ref,
        else:
            l = self.disparity_ref, self.disparity_other

        dfs = list()
        for d in l:
            df = cp.empty_like(d)
            assert df.flags.c_contiguous
            gpu_func = self.gpu_module.get_function('adaptiveMeanH')
            sz_block = 32, 32
            sz_grid = math.ceil(d.shape[1] / sz_block[0]), math.ceil(d.shape[0] / sz_block[1])
            gpu_func(
                block=sz_block,
                grid=sz_grid,
                args=(
                    df,
                    d,
                    d.shape[0],
                    d.shape[1]
                )
            )
            cp.cuda.runtime.deviceSynchronize()

            df2 = cp.empty_like(df)
            assert df2.flags.c_contiguous
            gpu_func = self.gpu_module.get_function('adaptiveMeanV')
            sz_block = 32, 32
            sz_grid = math.ceil(df.shape[1] / sz_block[0]), math.ceil(df.shape[0] / sz_block[1])
            gpu_func(
                block=sz_block,
                grid=sz_grid,
                args=(
                    df2,
                    df,
                    df.shape[0],
                    df.shape[1]
                )
            )
            cp.cuda.runtime.deviceSynchronize()
            dfs.append(df2)

        self.disparity_ref = dfs[0]
        if not self.param.postprocess_only_left:
            self.disparity_other = dfs[1]

    def median(self):
        if self.param.postprocess_only_left:
            l = self.disparity_ref,
        else:
            l = self.disparity_ref, self.disparity_other

        dfs = list()
        for d in l:
            df = cp.empty_like(d)
            assert df.flags.c_contiguous
            gpu_func = self.gpu_module.get_function('medianH')
            sz_block = 32, 32
            sz_grid = math.ceil(d.shape[1] / sz_block[0]), math.ceil(d.shape[0] / sz_block[1])
            gpu_func(
                block=sz_block,
                grid=sz_grid,
                args=(
                    df,
                    d,
                    d.shape[0],
                    d.shape[1]
                )
            )
            cp.cuda.runtime.deviceSynchronize()

            df2 = cp.empty_like(df)
            assert df2.flags.c_contiguous
            gpu_func = self.gpu_module.get_function('medianV')
            sz_block = 32, 32
            sz_grid = math.ceil(df.shape[1] / sz_block[0]), math.ceil(df.shape[0] / sz_block[1])
            gpu_func(
                block=sz_block,
                grid=sz_grid,
                args=(
                    df2,
                    df,
                    df.shape[0],
                    df.shape[1]
                )
            )
            cp.cuda.runtime.deviceSynchronize()
            dfs.append(df2)

        self.disparity_ref = dfs[0]
        if not self.param.postprocess_only_left:
            self.disparity_other = dfs[1]

    def initialize(self, img_ref, img_other):
        if self.gpu_module is None:
            self.setup_module()

        # upload images and create texture object
        self.img_ref = cp.asarray(img_ref)
        self.img_other = cp.asarray(img_other)
        self.to_ref = texture.create_texture_object(self.img_ref,
            addressMode = cp.cuda.runtime.cudaAddressModeBorder,
            filterMode = cp.cuda.runtime.cudaFilterModePoint,
            readMode = cp.cuda.runtime.cudaReadModeElementType)
        self.to_other = texture.create_texture_object(self.img_other,
            addressMode = cp.cuda.runtime.cudaAddressModeBorder,
            filterMode = cp.cuda.runtime.cudaFilterModePoint,
            readMode = cp.cuda.runtime.cudaReadModeElementType)

    def process(self, img_ref, img_other):
        self.initialize(img_ref, img_other)
        self.compute_descriptor()
        self.compute_disparity_candidate()
        self.compute_support_matches()
        self.compute_delaunay_triangulation()
        self.compute_disparity_planes()
        self.create_grid()
        self.compute_target_pixels()
        self.compute_disparity()
        self.left_right_consistency_check()
        self.remove_small_segments()
        self.gap_interpolation()
        if self.param.filter_adaptive_mean:
            self.adaptive_mean()
        if self.param.filter_median:
            self.median()

    def get_disparity(self):
        if (self.disparity_ref is None) or (self.disparity_other is None):
            return None, None
        return self.disparity_ref.get(), self.disparity_other.get()
