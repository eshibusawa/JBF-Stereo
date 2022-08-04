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

from unittest import TestCase
from nose.tools import ok_
import numpy as np
import cupy as cp
import scipy.signal
from scipy.signal.windows import hann

import add_path
from poc import create_Hann_window
from poc import phase_only_correlation_parameters
from poc import phase_only_correlation

class POCTestCase(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def create_hann_window_test(self):
        NN = 16, 32, 64
        for N in NN:
            h = hann(N).astype(np.float32)
            h2 = create_Hann_window(N)
            err = np.abs(h - h2)
            ok_(np.max(err) < 1e-6)

    def fft_test(self):
        NN = 4, 8, 16, 32, 64
        for N in NN:
            param = phase_only_correlation_parameters()
            param.window_width = N
            param.is_use_hann_window = False
            poc = phase_only_correlation(param)

            input = np.random.rand(param.window_width).astype(np.float32)
            input_gpu = cp.array(input, dtype=cp.float32)
            output_gpu = cp.zeros(param.window_width, dtype=cp.complex64)

            fftfunc = poc.module.get_function("applyFFTTest")
            sz_block = 1, 1
            sz_grid = 1, 1
            # call the kernel
            fftfunc(
                block=sz_block,
                grid=sz_grid,
                args=(
                    output_gpu.data,
                    input_gpu.data,
                )
            )
            output_ref = np.fft.fft(input)
            err = np.abs(output_ref - output_gpu.get())
            ok_(np.max(err) < 1e-5)

            fftfunc = poc.module.get_function("applyInverseFFTTest")
            # call the kernel
            input_gpu = cp.zeros(param.window_width, dtype=cp.complex64)
            fftfunc(
                block=sz_block,
                grid=sz_grid,
                args=(
                    input_gpu.data,
                    output_gpu.data,
                )
            )
            err = np.abs(input - np.real(input_gpu.get()))
            ok_(np.max(err) < 1e-5)

    def stft_test(self):
        length = 1024
        NN = 4, 8, 16, 32, 64
        is_use_window = False, True
        for N in NN:
            for f in is_use_window:
                param = phase_only_correlation_parameters()
                param.window_width = N
                param.is_use_hann_window = f
                poc = phase_only_correlation(param)

                input = (255 * np.random.rand(length)).astype(np.uint8).reshape(1, -1)
                output_gpu = poc.get_transformed_image(input)

                if param.is_use_hann_window:
                    window = scipy.signal.get_window('hann', param.window_width, False)
                else:
                    window = scipy.signal.get_window('boxcar', param.window_width, False)
                scale = np.sum(window)

                _, _, output_ref = scipy.signal.stft(input[0], window=window, nperseg = param.window_width, \
                                noverlap = param.window_width - 1, return_onesided = False)
                output_ref = output_ref[:,0:input.shape[1]].T

                err = np.abs(output_ref - output_gpu[0].get()/scale)
                ok_(np.max(err) < 5e-5)

    def phase_only_correlation_test(self):
        pixel_types = np.uint8, np.float32
        NN = 16, 32, 64
        for p in pixel_types:
            for N in NN:
                length = 2 * N
                shift_ref = N//8
                param = phase_only_correlation_parameters()
                param.window_width = N
                param.averaging_window_height = 1
                param.is_use_hann_window = False
                param.is_spectrum_weighting = False

                poc = phase_only_correlation(param, p)

                input1 = (255 * np.random.rand(length)).astype(p).reshape(1, -1)
                input2 = np.roll(input1, -shift_ref)
                disparity = np.zeros(input1.shape, dtype=np.int32)

                pocf_gpu = poc.get_phase_only_correlation(input1, input2, disparity)
                x, y = pocf_gpu.shape[1]//2, pocf_gpu.shape[0]//2
                pocf = pocf_gpu[y, x, :].get()

                input1_F = np.fft.fft(input1[y, x - N//2: x + N//2])
                input2_F = np.fft.fft(input2[y, x - N//2: x + N//2])
                pocf_ref = input1_F * np.conj(input2_F)
                pocf_ref = pocf_ref / (np.abs(pocf_ref) + 1E-7)
                pocf_ref = np.fft.ifft(pocf_ref)
                pocf_ref = np.fft.fftshift(pocf_ref)

                err = np.abs(pocf_ref - pocf)
                ok_(np.max(err) < 2e-7)

                shift = int(np.argmax(pocf) - (N//2))
                ok_(shift_ref == shift)
