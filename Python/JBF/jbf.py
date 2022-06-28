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

import numpy as np
import cupy as cp

class joint_bilateral_filter_parameters():
    def __init__(self):
        self.radius_h = 5
        self.radius_v = 5
        self.sigma_space = 10
        self.range_min = 0
        self.range_max = 256
        self.sigma_range = 10


class joint_bilateral_filter():
    @staticmethod
    def create_texture_object(img_gpu):
        if img_gpu.dtype == cp.uint8:
            channel_format_descriptor = cp.cuda.texture.ChannelFormatDescriptor(8, 0, 0, 0, cp.cuda.runtime.cudaChannelFormatKindUnsigned)
        elif img_gpu.dtype == cp.int16:
            channel_format_descriptor = cp.cuda.texture.ChannelFormatDescriptor(16, 0, 0, 0, cp.cuda.runtime.cudaChannelFormatKindSigned)
        else:
            return None
        img_gpu_2d = cp.cuda.texture.CUDAarray(channel_format_descriptor, img_gpu.shape[1], img_gpu.shape[0])
        img_gpu_2d.copy_from(img_gpu)

        img_rd = cp.cuda.texture.ResourceDescriptor(cp.cuda.runtime.cudaResourceTypeArray,
            cuArr = img_gpu_2d)
        img_td = cp.cuda.texture.TextureDescriptor(addressModes = (cp.cuda.runtime.cudaAddressModeBorder, cp.cuda.runtime.cudaAddressModeBorder),
            filterMode=cp.cuda.runtime.cudaFilterModePoint,
            readMode=cp.cuda.runtime.cudaReadModeElementType,
            normalizedCoords = 0)
        img_to = cp.cuda.texture.TextureObject(img_rd, img_td)
        return img_to

    def __init__(self, param):
        self.compute_kernel(param)
        self.compile_module()
        self.upload_kernel()
        self.radius_h = param.radius_h
        self.radius_v = param.radius_v

    def compute_kernel(self, param):
        r = np.arange(param.range_min, param.range_max + 1)
        d = r - param.range_min
        self.range_kernel = np.exp(-0.5 * d * d / param.sigma_range / param.sigma_range).astype(np.float32)

        r_h = np.arange(-param.radius_h, param.radius_h + 1)
        r_v = np.arange(-param.radius_v, param.radius_v + 1)
        r_vh = np.empty((2, r_v.shape[0], r_h.shape[0]), dtype=np.float32)
        r_vh[0] = r_h[np.newaxis, :]
        r_vh[1] = r_v[:, np.newaxis]
        d_vh = np.sum(r_vh * r_vh, axis = 0)
        self.spatial_kernel = np.exp(-0.5 * d_vh / param.sigma_space / param.sigma_space).astype(np.float32)

    def compile_module(self):
        fpfn = os.path.join(os.path.dirname(__file__), 'jbf_cuda.cu')
        # load raw kernel
        with open(fpfn, 'r') as f:
            cuda_source = f.read()
        cuda_source = cuda_source.replace('BILATERAL_FILTER_COEFFICIENTS_SIZE_RANGE', str(self.range_kernel.size))
        cuda_source = cuda_source.replace('BILATERAL_FILTER_COEFFICIENTS_SIZE_SPACE', str(self.spatial_kernel.size))
        self.module = cp.RawModule(code=cuda_source)
        self.jbf = self.module.get_function("applyJointBilateralFilter")

    def upload_kernel(self):
        # upload kernel coefficients
        range_kernel_gpu_ptr = self.module.get_global("g_BFRange")
        range_kernel_gpu = cp.ndarray(self.range_kernel.shape, cp.float32, range_kernel_gpu_ptr)
        range_kernel_gpu[:] = cp.array(self.range_kernel)
        spatial_kernel_gpu_ptr = self.module.get_global("g_BFSpace")
        spatial_kernel_gpu = cp.ndarray(self.spatial_kernel.shape, cp.float32, spatial_kernel_gpu_ptr)
        spatial_kernel_gpu[:] = cp.array(self.spatial_kernel)

    def apply(self, img, guide_img):
        assert img.dtype == np.int16
        assert guide_img.dtype == np.uint8

        # upload images
        img_gpu = cp.array(img, dtype=cp.int16)
        guide_img_gpu = cp.array(guide_img, dtype=cp.uint8)
        img_filtered_gpu = cp.zeros(img.shape, dtype=img_gpu.dtype)

        assert img_gpu.flags.c_contiguous
        assert guide_img_gpu.flags.c_contiguous
        assert img_filtered_gpu.flags.c_contiguous

        img_to = self.create_texture_object(img_gpu)
        guide_img_to = self.create_texture_object(guide_img_gpu)

        sz_block = 32, 32
        sz_grid = math.ceil(img_filtered_gpu.shape[1] / sz_block[1]), math.ceil(img_filtered_gpu.shape[0] / sz_block[0])
        # call the kernel
        self.jbf(
            block=sz_block,
            grid=sz_grid,
            args=(
                img_filtered_gpu,
                img_to,
                guide_img_to,
                self.radius_h,
                self.radius_v,
                img_filtered_gpu.shape[1],
                img_filtered_gpu.shape[0]
            )
        )

        return cp.asnumpy(img_filtered_gpu)
