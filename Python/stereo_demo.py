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

import cv2

import add_path
from colormap import disparity_to_colormap
from jbf import joint_bilateral_filter as jbf
from jbf import joint_bilateral_filter_parameters as jbf_param
from patch_match import patch_match_stereo
from elas import ELAS as elas
from elas_params import elas_params as elas_param

def jbf_demo(l, r, max_disparity, max_disparity_visualization):
    stereo = cv2.StereoBM_create(numDisparities=max_disparity, blockSize=7)
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

    return disparity

def patch_match_demo(l, r, max_disparity):
    pm = patch_match_stereo()
    pm.max_disparity = max_disparity
    pm.min_disparity = 0
    pm.patch_width = 11
    pm.patch_height = 11
    pm.truncate_color = 50
    pm.truncate_grad = 20
    pm.weight_gamma = 20
    pm.blending_alpha = 0.8
    pm.spatial_delta = 5
    pm.enable_consistent_gradient_operator = True
    pm.compile_module()

    pm.initialize_planes(l, r)
    for _ in range(0, 5):
        pm.compute_spatial_propergation()
        pm.compute_random_search()
        pm.toggle_red_and_black()

        pm.compute_spatial_propergation()
        pm.compute_random_search()
        pm.toggle_red_and_black()

    return pm.compute_disparity().get()

def elas_demo(l, r):
    p = elas_param()
    e = elas(p)
    e.process(l, r)

    return e.get_disparity()