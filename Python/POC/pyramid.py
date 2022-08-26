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

import add_path
from texture import create_texture_object
from poc import phase_only_correlation_parameters, phase_only_correlation

class image_pyramid():
    def __init__(self, param):
        # determine max pyramid level
        max_level = 0
        for l in range(2, param.max_disparity):
            if math.floor(param.window_width/4) * (2 << (l - 1)) >= param.max_disparity:
                max_level = l
                break
        self.max_pyramid_level = max_level
        self.compile_module()

    def compile_module(self):
        fpfn = os.path.join(os.path.dirname(__file__), 'pyramid_cuda.cu')
        # load raw kernel
        with open(fpfn, 'r') as f:
            cuda_source = f.read()
        self.module = cp.RawModule(code=cuda_source)
        self.module.compile()

    def get_image_pyramid(self, img):
        img_gpu = cp.array(img.astype(np.float32))
        assert img_gpu.flags.c_contiguous

        pyramid = list()
        pyramid.append(img_gpu.get())
        for l in range(self.max_pyramid_level):
            img_to = create_texture_object(img_gpu, addressMode = cp.cuda.runtime.cudaAddressModeBorder,
                filterMode = cp.cuda.runtime.cudaFilterModePoint,
                readMode = cp.cuda.runtime.cudaReadModeElementType)

            img_gpu_transformed = cp.empty((img_gpu.shape[0]//2, img_gpu.shape[1]//2), dtype=img_gpu.dtype)

            sz_block = 32, 32
            sz_grid = math.ceil(img_gpu.shape[1] / sz_block[0]), math.ceil(img_gpu.shape[0] / sz_block[1])
            # call the kernel
            pocfunc = self.module.get_function("downSampling")
            pocfunc(
                block=sz_block,
                grid=sz_grid,
                args=(
                    img_gpu_transformed.data,
                    img_to,
                    img_gpu_transformed.shape[1],
                    img_gpu_transformed.shape[0]
                )
            )
            img_gpu = img_gpu_transformed
            pyramid.append(img_gpu.get())

        return pyramid

    def upsampling(self, disprity, shape):
        d_gpu = cp.array(disprity.astype(np.float32))
        assert d_gpu.flags.c_contiguous

        d_to = create_texture_object(d_gpu, addressMode = cp.cuda.runtime.cudaAddressModeBorder,
            filterMode = cp.cuda.runtime.cudaFilterModePoint,
            readMode = cp.cuda.runtime.cudaReadModeElementType)
        du_gpu = cp.zeros(shape, dtype=d_gpu.dtype)

        sz_block = 32, 32
        sz_grid = math.ceil(du_gpu.shape[1] / sz_block[0]), math.ceil(du_gpu.shape[0] / sz_block[1])
        # call the kernel
        pocfunc = self.module.get_function("upSampling")
        pocfunc(
            block=sz_block,
            grid=sz_grid,
            args=(
                du_gpu.data,
                d_to,
                du_gpu.shape[1],
                du_gpu.shape[0]
            )
        )

        return du_gpu.get()
