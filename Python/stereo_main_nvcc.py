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

#
# This demo requires CUDAToolkit and executable nvcc!
#
import numpy as np
import cv2

import add_path
import add_environment
from tgv import TGV as tgv_impl
from tgv_params import tgv_params

from colormap import disparity_to_colormap

if __name__ == '__main__':
    print('TGV Demo')
    l = cv2.imread('../data/teddy/im2.png', cv2.IMREAD_GRAYSCALE)
    r = cv2.imread('../data/teddy/im6.png', cv2.IMREAD_GRAYSCALE)
    disparity_gt = (cv2.imread('../data/teddy/disp2.png', cv2.IMREAD_GRAYSCALE).astype(np.int16) * 4)
    numDisparities = 64
    max_disparity_visualization = numDisparities
    di = disparity_to_colormap(disparity_gt/16, 0, max_disparity_visualization)
    cv2.imwrite('disparity_gt.png', di)

    param = tgv_params()
    param.max_disparity = numDisparities
    tgv = tgv_impl(param)
    tgv.setup_module()
    tgv.process(l, r)
    disparity_lasw = tgv.get_disparity('LASW')
    di = disparity_to_colormap(disparity_lasw, 0, max_disparity_visualization)
    cv2.imwrite('disparity_lasw.png', di)
    disparity_tgv = tgv.get_disparity('TGV')
    di = disparity_to_colormap(disparity_tgv, 0, max_disparity_visualization)
    cv2.imwrite('disparity_tgv.png', di)
