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
import cv2

import add_path
from colormap import colormap as jet
from jbf import joint_bilateral_filter as jbf
from jbf import joint_bilateral_filter_parameters as jbf_param

def disparity_to_colormap(disparity, min_d, max_d):
    mask = disparity > 0
    delta = max_d - min_d
    index = np.round(np.maximum(0, np.minimum(disparity - min_d, delta) / delta) * 255).astype(np.int)
    img = (jet[index] * 255).astype(np.uint8)
    img[~mask] = (0,0,0)
    return img

if __name__ == '__main__':
    l = cv2.imread('../data/teddy/im2.png', cv2.IMREAD_GRAYSCALE)
    r = cv2.imread('../data/teddy/im6.png', cv2.IMREAD_GRAYSCALE)
    disparity_gt = (cv2.imread('../data/teddy/disp2.png', cv2.IMREAD_GRAYSCALE).astype(np.int16) * 4)
    numDisparities = 64
    max_disparity_visualization = numDisparities

    di = disparity_to_colormap(disparity_gt/16, 1, max_disparity_visualization)
    cv2.imwrite('disparity_gt.png', di)

    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=7)
    disparity = stereo.compute(l, r)
    di = disparity_to_colormap(disparity/16, 1, max_disparity_visualization)
    cv2.imwrite('disparity_block_matching.png', di)

    cv2.filterSpeckles(disparity, -16, 32, 16)
    di = disparity_to_colormap(disparity/16, 1, max_disparity_visualization)
    cv2.imwrite('disparity_speckle.png', di)

    mask = disparity < 0
    p = jbf_param()
    p.radius_h = 5
    p.radius_v = 5
    p.sigma_space = 3
    p.range_min = 0
    p.range_max = 255
    p.sigma_range = 3
    filter = jbf(p)
    disparity = filter.apply(disparity, l)
    di = disparity_to_colormap(disparity/16, 1, max_disparity_visualization)
    cv2.imwrite('disparity_refined.png', di)
