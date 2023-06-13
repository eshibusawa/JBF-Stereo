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

class CCL():
    def __init__(self):
        self.gpu_module = None
        self.block = 1024

    def compile_module(self):
        dn = os.path.dirname(__file__)
        fnl = list()
        fnl.append(os.path.join(dn, os.path.join('CCL', 'ccl_le.cu')))

        cuda_source = None
        for fpfn in fnl:
            with open(fpfn, 'r') as f:
                cs = f.read()
            if cuda_source is None:
                cuda_source = cs
            else:
                cuda_source += cs

        self.gpu_module = cp.RawModule(code=cuda_source)
        self.gpu_module.compile()

    def labeling(self, d, threshold):
        if self.gpu_module is None:
            self.compile_module()

        sz = int(math.sqrt(float(d.size) / self.block)) + 1
        l_gpu = cp.empty(d.shape, dtype=cp.int32)
        r_gpu = cp.empty(d.shape, dtype=cp.int32)
        assert l_gpu.flags.c_contiguous
        assert r_gpu.flags.c_contiguous

        gpu_func = self.gpu_module.get_function('init_CCL')
        sz_block = self.block, 1
        sz_grid = sz, sz
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                l_gpu,
                r_gpu,
                l_gpu.size
            )
        )
        cp.cuda.runtime.deviceSynchronize()

        converged_gpu = cp.empty((1), dtype=cp.int32)
        while True:
            converged_gpu[:] = 0
            gpu_func = self.gpu_module.get_function('scanning')
            gpu_func(
                    block=sz_block,
                    grid=sz_grid,
                    args=(
                        d,
                        l_gpu,
                        r_gpu,
                        converged_gpu,
                        l_gpu.size,
                        d.shape[1],
                        threshold
                    )
                )
            if converged_gpu.get() != 0:
                gpu_func = self.gpu_module.get_function('analysis')
                gpu_func(
                        block=sz_block,
                        grid=sz_grid,
                        args=(
                            l_gpu,
                            r_gpu,
                            l_gpu.size
                        )
                    )
                cp.cuda.runtime.deviceSynchronize()
                gpu_func = self.gpu_module.get_function('labeling')
                gpu_func(
                        block=sz_block,
                        grid=sz_grid,
                        args=(
                            l_gpu,
                            r_gpu,
                            l_gpu.size
                        )
                    )
                cp.cuda.runtime.deviceSynchronize()
            else:
                break
        return l_gpu
