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

import numpy as np

colormap_raw = np.array([
[0.0, 0.0, 0.5156],
[0.0, 0.0, 0.5312],
[0.0, 0.0, 0.5469],
[0.0, 0.0, 0.5625],
[0.0, 0.0, 0.5781],
[0.0, 0.0, 0.5938],
[0.0, 0.0, 0.6094],
[0.0, 0.0, 0.6250],
[0.0, 0.0, 0.6406],
[0.0, 0.0, 0.6562],
[0.0, 0.0, 0.6719],
[0.0, 0.0, 0.6875],
[0.0, 0.0, 0.7031],
[0.0, 0.0, 0.7188],
[0.0, 0.0, 0.7344],
[0.0, 0.0, 0.7500],
[0.0, 0.0, 0.7656],
[0.0, 0.0, 0.7812],
[0.0, 0.0, 0.7969],
[0.0, 0.0, 0.8125],
[0.0, 0.0, 0.8281],
[0.0, 0.0, 0.8438],
[0.0, 0.0, 0.8594],
[0.0, 0.0, 0.8750],
[0.0, 0.0, 0.8906],
[0.0, 0.0, 0.9062],
[0.0, 0.0, 0.9219],
[0.0, 0.0, 0.9375],
[0.0, 0.0, 0.9531],
[0.0, 0.0, 0.9688],
[0.0, 0.0, 0.9844],
[0.0, 0.0, 1.0000],
[0.0, 0.0156, 1.0000],
[0.0, 0.0312, 1.0000],
[0.0, 0.0469, 1.0000],
[0.0, 0.0625, 1.0000],
[0.0, 0.0781, 1.0000],
[0.0, 0.0938, 1.0000],
[0.0, 0.1094, 1.0000],
[0.0, 0.1250, 1.0000],
[0.0, 0.1406, 1.0000],
[0.0, 0.1562, 1.0000],
[0.0, 0.1719, 1.0000],
[0.0, 0.1875, 1.0000],
[0.0, 0.2031, 1.0000],
[0.0, 0.2188, 1.0000],
[0.0, 0.2344, 1.0000],
[0.0, 0.2500, 1.0000],
[0.0, 0.2656, 1.0000],
[0.0, 0.2812, 1.0000],
[0.0, 0.2969, 1.0000],
[0.0, 0.3125, 1.0000],
[0.0, 0.3281, 1.0000],
[0.0, 0.3438, 1.0000],
[0.0, 0.3594, 1.0000],
[0.0, 0.3750, 1.0000],
[0.0, 0.3906, 1.0000],
[0.0, 0.4062, 1.0000],
[0.0, 0.4219, 1.0000],
[0.0, 0.4375, 1.0000],
[0.0, 0.4531, 1.0000],
[0.0, 0.4688, 1.0000],
[0.0, 0.4844, 1.0000],
[0.0, 0.5000, 1.0000],
[0.0, 0.5156, 1.0000],
[0.0, 0.5312, 1.0],
[0.0, 0.5469, 1.0],
[0.0, 0.5625, 1.0],
[0.0, 0.5781, 1.0],
[0.0, 0.5938, 1.0],
[0.0, 0.6094, 1.0],
[0.0, 0.6250, 1.0],
[0.0, 0.6406, 1.0],
[0.0, 0.6562, 1.0],
[0.0, 0.6719, 1.0],
[0.0, 0.6875, 1.0],
[0.0, 0.7031, 1.0],
[0.0, 0.7188, 1.0],
[0.0, 0.7344, 1.0],
[0.0, 0.7500, 1.0],
[0.0, 0.7656, 1.0],
[0.0, 0.7812, 1.0],
[0.0, 0.7969, 1.0],
[0.0, 0.8125, 1.0],
[0.0, 0.8281, 1.0],
[0.0, 0.8438, 1.0],
[0.0, 0.8594, 1.0],
[0.0, 0.8750, 1.0],
[0.0, 0.8906, 1.0],
[0.0, 0.9062, 1.0],
[0.0, 0.9219, 1.0],
[0.0, 0.9375, 1.0],
[0.0, 0.9531, 1.0],
[0.0, 0.9688, 1.0],
[0.0, 0.9844, 1.0],
[0.0, 1.0, 1.0],
[0.0156, 1.0, 0.9844],
[0.0312, 1.0, 0.9688],
[0.0469, 1.0, 0.9531],
[0.0625, 1.0, 0.9375],
[0.0781, 1.0, 0.9219],
[0.0938, 1.0, 0.9062],
[0.1094, 1.0, 0.8906],
[0.1250, 1.0, 0.8750],
[0.1406, 1.0, 0.8594],
[0.1562, 1.0, 0.8438],
[0.1719, 1.0, 0.8281],
[0.1875, 1.0, 0.8125],
[0.2031, 1.0, 0.7969],
[0.2188, 1.0, 0.7812],
[0.2344, 1.0, 0.7656],
[0.2500, 1.0, 0.7500],
[0.2656, 1.0, 0.7344],
[0.2812, 1.0, 0.7188],
[0.2969, 1.0, 0.7031],
[0.3125, 1.0, 0.6875],
[0.3281, 1.0, 0.6719],
[0.3438, 1.0, 0.6562],
[0.3594, 1.0, 0.6406],
[0.3750, 1.0, 0.6250],
[0.3906, 1.0, 0.6094],
[0.4062, 1.0, 0.5938],
[0.4219, 1.0, 0.5781],
[0.4375, 1.0, 0.5625],
[0.4531, 1.0, 0.5469],
[0.4688, 1.0, 0.5312],
[0.4844, 1.0, 0.5156],
[0.5000, 1.0, 0.5000],
[0.5156, 1.0, 0.4844],
[0.5312, 1.0, 0.4688],
[0.5469, 1.0, 0.4531],
[0.5625, 1.0, 0.4375],
[0.5781, 1.0, 0.4219],
[0.5938, 1.0, 0.4062],
[0.6094, 1.0, 0.3906],
[0.6250, 1.0, 0.3750],
[0.6406, 1.0, 0.3594],
[0.6562, 1.0, 0.3438],
[0.6719, 1.0, 0.3281],
[0.6875, 1.0, 0.3125],
[0.7031, 1.0, 0.2969],
[0.7188, 1.0, 0.2812],
[0.7344, 1.0, 0.2656],
[0.7500, 1.0, 0.2500],
[0.7656, 1.0, 0.2344],
[0.7812, 1.0, 0.2188],
[0.7969, 1.0, 0.2031],
[0.8125, 1.0, 0.1875],
[0.8281, 1.0, 0.1719],
[0.8438, 1.0, 0.1562],
[0.8594, 1.0, 0.1406],
[0.8750, 1.0, 0.1250],
[0.8906, 1.0, 0.1094],
[0.9062, 1.0, 0.0938],
[0.9219, 1.0, 0.0781],
[0.9375, 1.0, 0.0625],
[0.9531, 1.0, 0.0469],
[0.9688, 1.0, 0.0312],
[0.9844, 1.0, 0.0156],
[1.0, 1.0, 0.0],
[1.0, 0.9844, 0.0],
[1.0, 0.9688, 0.0],
[1.0, 0.9531, 0.0],
[1.0, 0.9375, 0.0],
[1.0, 0.9219, 0.0],
[1.0, 0.9062, 0.0],
[1.0, 0.8906, 0.0],
[1.0, 0.875, 0.0],
[1.0, 0.8594, 0.0],
[1.0, 0.8438, 0.0],
[1.0, 0.8281, 0.0],
[1.0, 0.8125, 0.0],
[1.0, 0.7969, 0.0],
[1.0, 0.7812, 0.0],
[1.0, 0.7656, 0.0],
[1.0, 0.75, 0.0],
[1.0, 0.7344, 0.0],
[1.0, 0.7188, 0.0],
[1.0, 0.7031, 0.0],
[1.0, 0.6875, 0.0],
[1.0, 0.6719, 0.0],
[1.0, 0.6562, 0.0],
[1.0, 0.6406, 0.0],
[1.0, 0.625, 0.0],
[1.0, 0.6094, 0.0],
[1.0, 0.5938, 0.0],
[1.0, 0.5781, 0.0],
[1.0, 0.5625, 0.0],
[1.0, 0.5469, 0.0],
[1.0, 0.5312, 0.0],
[1.0, 0.5156, 0.0],
[1.0, 0.5, 0.0],
[1.0, 0.4844, 0.0],
[1.0, 0.4688, 0.0],
[1.0, 0.4531, 0.0],
[1.0, 0.4375, 0.0],
[1.0, 0.4219, 0.0],
[1.0, 0.4062, 0.0],
[1.0, 0.3906, 0.0],
[1.0, 0.375, 0.0],
[1.0, 0.3594, 0.0],
[1.0, 0.3438, 0.0],
[1.0, 0.3281, 0.0],
[1.0, 0.3125, 0.0],
[1.0, 0.2969, 0.0],
[1.0, 0.2812, 0.0],
[1.0, 0.2656, 0.0],
[1.0, 0.25, 0.0],
[1.0, 0.2344, 0.0],
[1.0, 0.2188, 0.0],
[1.0, 0.2031, 0.0],
[1.0, 0.1875, 0.0],
[1.0, 0.1719, 0.0],
[1.0, 0.1562, 0.0],
[1.0, 0.1406, 0.0],
[1.0, 0.125, 0.0],
[1.0, 0.1094, 0.0],
[1.0, 0.0938, 0.0],
[1.0, 0.0781, 0.0],
[1.0, 0.0625, 0.0],
[1.0, 0.0469, 0.0],
[1.0, 0.0312, 0.0],
[1.0, 0.0156, 0.0],
[1.0, 0.0, 0.0],
[0.9844, 0.0, 0.0],
[0.9688, 0.0, 0.0],
[0.9531, 0.0, 0.0],
[0.9375, 0.0, 0.0],
[0.9219, 0.0, 0.0],
[0.9062, 0.0, 0.0],
[0.8906, 0.0, 0.0],
[0.875, 0.0, 0.0],
[0.8594, 0.0, 0.0],
[0.8438, 0.0, 0.0],
[0.8281, 0.0, 0.0],
[0.8125, 0.0, 0.0],
[0.7969, 0.0, 0.0],
[0.7812, 0.0, 0.0],
[0.7656, 0.0, 0.0],
[0.75, 0.0, 0.0],
[0.7344, 0.0, 0.0],
[0.7188, 0.0, 0.0],
[0.7031, 0.0, 0.0],
[0.6875, 0.0, 0.0],
[0.6719, 0.0, 0.0],
[0.6562, 0.0, 0.0],
[0.6406, 0.0, 0.0],
[0.625, 0.0, 0.0],
[0.6094, 0.0, 0.0],
[0.5938, 0.0, 0.0],
[0.5781, 0.0, 0.0],
[0.5625, 0.0, 0.0],
[0.5469, 0.0, 0.0],
[0.5312, 0.0, 0.0],
[0.5156, 0.0, 0.0],
[0.5, 0.0, 0.0]])
colormap = np.fliplr(colormap_raw)

def depth_to_colormap(depth, min_z, max_z):
    mask = depth > 0
    delta = 1/min_z - 1/max_z
    index = np.round(np.maximum(0, np.minimum(1/(depth + 1E-6) - 1/max_z, delta) / delta) * 255).astype(np.int32)
    img = (colormap[index] * 255).astype(np.uint8)
    img[~mask] = (0,0,0)
    return img

def disparity_to_colormap(disparity, min_d, max_d):
    mask = disparity > 0
    delta = max_d - min_d
    index = np.round(np.maximum(0, np.minimum(disparity - min_d, delta) / delta) * 255).astype(np.int32)
    img = (colormap[index] * 255).astype(np.uint8)
    img[~mask] = (0,0,0)
    return img
