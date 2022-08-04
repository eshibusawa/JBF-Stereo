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

from texture import create_texture_object

def create_Hann_window(length):
    s = 2 * np.pi / (length - 1)
    return (1 - np.cos(s * np.arange(0, length, dtype=np.float32)))/2

def create_DFT_rotator(length):
    s = -2j * np.pi / length
    return np.exp(s * np.arange(0, length, dtype=np.float32))

def check_two_power(val):
    p = int(math.log2(val))
    return val == 2**p

class phase_only_correlation_parameters():
    def __init__(self):
        self.window_width = 16
        self.averaging_window_height = 7
        self.is_use_hann_window = False
        self.is_spectrum_weighting = False
        self.is_use_dc_suppression = False

class phase_only_correlation():
    def __init__(self, param, pixel_type=np.uint8):
        self.window_width = param.window_width
        self.averaging_window_height = param.averaging_window_height
        self.is_use_hann_window = param.is_use_hann_window
        self.is_spectrum_weighting = param.is_spectrum_weighting
        self.is_use_dc_suppression = param.is_use_dc_suppression
        self.pixel_type = pixel_type

        self.compute_constants()
        self.compile_module()
        self.upload_constants()

    def compute_constants(self):
        self.hann_window = create_Hann_window(self.window_width)
        self.rotator = create_DFT_rotator(self.window_width)

    def compile_module(self):
        assert check_two_power(self.hann_window.shape[0])
        assert (self.pixel_type == np.uint8) or (self.pixel_type == np.float32)

        fpfn = os.path.join(os.path.dirname(__file__), 'poc_cuda.cu')
        # load raw kernel
        with open(fpfn, 'r') as f:
            cuda_source = f.read()
        fpfn = os.path.join(os.path.dirname(__file__), 'poc_cuda_test.cu')
        # load raw kernel
        with open(fpfn, 'r') as f:
            cuda_source += f.read()
        cuda_source = cuda_source.replace('POC_WINDOW_WIDTH', str(self.hann_window.shape[0]))
        cuda_source = cuda_source.replace('POC_AVERAGING_WINDOW_HEIGHT', str(self.averaging_window_height))
        cuda_source = cuda_source.replace('POC_USE_HANN_WINDOW', str(1 if self.is_use_hann_window else 0))
        cuda_source = cuda_source.replace('POC_USE_SPECTRUM_WEIGHTING', str(1 if self.is_spectrum_weighting else 0))
        cuda_source = cuda_source.replace('POC_USE_DC_SUPPRESSION', str(1 if self.is_use_dc_suppression else 0))
        cuda_source = cuda_source.replace('POC_PIXEL_TYPE', 'float' if self.pixel_type == np.float32 else 'unsigned char')
        self.module = cp.RawModule(code=cuda_source)
        self.module.compile()

    def upload_symbol(self, arr, key, dtype=cp.float32):
        arr_ptr = self.module.get_global(key)
        arr_gpu = cp.ndarray(arr.shape, dtype, arr_ptr)
        arr_gpu[:] = cp.array(arr, dtype=dtype)

    def upload_constants(self):
        # upload Hann window
        self.upload_symbol(self.hann_window, 'g_HannWindow')
        self.upload_symbol(self.rotator, 'g_rotator', cp.complex64)

    def get_transformed_image(self, img):
        assert img.dtype == self.pixel_type

        # upload images
        img_gpu = cp.array(img)
        img_transformed_gpu = cp.zeros((img_gpu.shape[0], img_gpu.shape[1], self.window_width), dtype=cp.complex64)

        assert img_gpu.flags.c_contiguous
        assert img_transformed_gpu.flags.c_contiguous

        img_to = create_texture_object(img_gpu, addressMode = cp.cuda.runtime.cudaAddressModeBorder,
            filterMode = cp.cuda.runtime.cudaFilterModePoint,
            readMode = cp.cuda.runtime.cudaReadModeElementType)

        if self.pixel_type == np.uint8:
            sz_block = 32, 32
        else:
            sz_block = 32, 16
        sz_grid = math.ceil(img_gpu.shape[1] / sz_block[1]), math.ceil(img_gpu.shape[0] / sz_block[0])
        # call the kernel
        pocfunc = self.module.get_function("applyTransformation")
        pocfunc(
            block=sz_block,
            grid=sz_grid,
            args=(
                img_transformed_gpu.data,
                img_to,
                img_transformed_gpu.shape[1],
                img_transformed_gpu.shape[0]
            )
        )

        return img_transformed_gpu

    def get_phase_only_correlation(self, img_ref, img_other, disparity):
        img_ref_F = self.get_transformed_image(img_ref)
        img_other_F = self.get_transformed_image(img_other)
        disparity_gpu = cp.array(disparity, dtype=cp.int32)
        poc_gpu = cp.empty(img_ref_F.shape, dtype=cp.float32)

        assert img_ref_F.flags.c_contiguous
        assert img_other_F.flags.c_contiguous
        assert disparity_gpu.flags.c_contiguous
        assert poc_gpu.flags.c_contiguous

        sz_block = 16, 16
        sz_grid = math.ceil(img_ref_F.shape[1] / sz_block[1]), math.ceil(img_ref_F.shape[0] / sz_block[0])
        # call the kernel
        pocfunc = self.module.get_function("getPhaseOnlyCorrelation")
        pocfunc(
            block=sz_block,
            grid=sz_grid,
            args=(
                poc_gpu.data,
                img_ref_F.data,
                img_other_F.data,
                disparity_gpu.data,
                img_ref_F.shape[1],
                img_ref_F.shape[0]
            )
        )

        return poc_gpu

    def get_disparity(self, poc):
        disparity_gpu = cp.zeros(poc.shape[:2], dtype=cp.float32)
        correlation_value_gpu = cp.zeros(poc.shape[:2], dtype=cp.float32)
        assert poc.flags.c_contiguous
        assert disparity_gpu.flags.c_contiguous

        sz_block = 32, 32
        sz_grid = math.ceil(poc.shape[1] / sz_block[1]), math.ceil(poc.shape[0] / sz_block[0])
        # call the kernel
        pocfunc = self.module.get_function("getDisparity")
        pocfunc(
            block=sz_block,
            grid=sz_grid,
            args=(
                disparity_gpu.data,
                correlation_value_gpu.data,
                poc.data,
                disparity_gpu.shape[1],
                disparity_gpu.shape[0]
            )
        )

        return disparity_gpu, correlation_value_gpu
