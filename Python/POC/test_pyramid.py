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

import numpy as np

import add_path
from poc import phase_only_correlation_parameters
from pyramid import image_pyramid


class TestPyramid:
    def make_parameters(self, window_width=16, max_disparity=128):
        param = phase_only_correlation_parameters()
        param.window_width = window_width
        param.max_disparity = max_disparity
        param.is_use_hann_window = True
        param.is_spectrum_weighting = True
        return param

    def test_max_pyramid_level(self):
        # the search range of a level is window_width / 4, the disparity doubles at every level
        for window_width in 16, 32, 64:
            for max_disparity in 32, 64, 128, 256:
                level = image_pyramid(self.make_parameters(window_width, max_disparity)).max_pyramid_level
                assert (window_width // 4) * 2**level >= max_disparity
                assert level == 2 or (window_width // 4) * 2**(level - 1) < max_disparity

    def test_down_sampling(self):
        pyramid = image_pyramid(self.make_parameters())
        img = (255 * np.random.rand(48, 64)).astype(np.float32)
        levels = pyramid.get_image_pyramid(img)
        assert len(levels) == pyramid.max_pyramid_level + 1
        assert levels[0].shape == img.shape
        ref = img
        for level in levels[1:]:
            ref = ref[0:2 * (ref.shape[0] // 2), 0:2 * (ref.shape[1] // 2)]
            ref = ref.reshape(ref.shape[0] // 2, 2, ref.shape[1] // 2, 2).mean(axis=(1, 3))
            assert level.shape == ref.shape
            assert np.max(np.abs(level - ref)) < 1e-4

    def test_address_mode(self):
        # the up-sampling reads one texel outside of the coarse image when the size of the fine level is odd:
        # zero (border), mirror (the last texel again) or wrap (the first texel); the 2x2 averaging never leaves the image
        for address_mode in 'border', 'mirror', 'wrap':
            param = self.make_parameters()
            param.address_mode = address_mode
            pyramid = image_pyramid(param)
            d = (10 * np.random.rand(6, 10)).astype(np.float32)
            shape = (13, 21)
            du = pyramid.upsampling(d, shape)
            y = np.arange(shape[0]) // 2
            x = np.arange(shape[1]) // 2
            if address_mode == 'border':
                ref = 2 * np.where((y[:, None] < 6) & (x[None, :] < 10), d[np.minimum(y, 5)][:, np.minimum(x, 9)], 0)
            elif address_mode == 'mirror':
                ref = 2 * d[np.minimum(y, 5)][:, np.minimum(x, 9)]
            else:
                ref = 2 * d[y % 6][:, x % 10]
            assert np.max(np.abs(du - ref)) < 1e-6, address_mode

            img = (255 * np.random.rand(48, 64)).astype(np.float32)
            levels = pyramid.get_image_pyramid(img)
            ref = img
            for level in levels[1:]:
                ref = ref[0:2 * (ref.shape[0] // 2), 0:2 * (ref.shape[1] // 2)]
                ref = ref.reshape(ref.shape[0] // 2, 2, ref.shape[1] // 2, 2).mean(axis=(1, 3))
                assert np.max(np.abs(level - ref)) < 1e-4, address_mode

    def test_up_sampling(self):
        pyramid = image_pyramid(self.make_parameters())
        d = (10 * np.random.rand(12, 20)).astype(np.float32)
        du = pyramid.upsampling(d, (24, 40))
        ref = 2 * np.repeat(np.repeat(d, 2, axis=0), 2, axis=1)
        assert np.max(np.abs(du - ref)) < 1e-6
