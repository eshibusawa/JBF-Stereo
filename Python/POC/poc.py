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

import os

import numpy as np
import cupy as cp

def create_Hann_window(length):
    s = 2 * np.pi / (length - 1)
    return (1 - np.cos(s * np.arange(0, length, dtype=np.float32)))/2

def create_DFT_rotator(length):
    s = -2j * np.pi / length
    return np.exp(s * np.arange(0, length, dtype=np.float32))

class phase_only_correlation_parameters():
    def __init__(self):
        self.window_width = 16
        self.is_use_hann_window = False

class phase_only_correlation():
    def __init__(self, param):
        self.window_width = param.window_width
        self.is_use_hann_window = param.is_use_hann_window

        self.compute_constants()
        self.compile_module()
        self.upload_constants()

    def compute_constants(self):
        self.hann_window = create_Hann_window(self.window_width)
        self.rotator = create_DFT_rotator(self.window_width)

    def compile_module(self):
        fpfn = os.path.join(os.path.dirname(__file__), 'poc_cuda.cu')
        # load raw kernel
        with open(fpfn, 'r') as f:
            cuda_source = f.read()
        fpfn = os.path.join(os.path.dirname(__file__), 'poc_cuda_test.cu')
        # load raw kernel
        with open(fpfn, 'r') as f:
            cuda_source += f.read()
        cuda_source = cuda_source.replace('POC_WINDOW_WIDTH', str(self.hann_window.shape[0]))
        cuda_source = cuda_source.replace('POC_USE_HANN_WINDOW', str(1 if self.is_use_hann_window else 0))
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
