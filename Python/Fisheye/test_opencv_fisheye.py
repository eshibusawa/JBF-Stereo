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
import cv2

from opencv_fisheye import opencv_fisheye as kbcm

def create_grid():
    x_range = np.arange(0, 120*5, 10, dtype=np.float32) # [mm]
    y_range = np.arange(0, 100*5, 10, dtype=np.float32) # [mm]
    x_range -= np.average(x_range)
    y_range -= np.average(y_range)
    xyz = np.empty((3, y_range.shape[0], x_range.shape[0]), dtype=np.float32)
    xyz[0, :, :] = x_range[np.newaxis, :]
    xyz[1, :, :] = y_range[:, np.newaxis]
    xyz[2, :, :] = 100
    xyz = xyz.reshape(3, -1)
    return xyz

class OpenCVFisheyeTestCase(TestCase):
    def setUp(self):
        self.eps_mm = 1E-1
        self.kb_camera = kbcm.get_default_camera()

    def tearDown(self):
        pass

    def project_unproject_test(self):
        xyz = create_grid()
        xy = self.kb_camera.project(xyz)
        if False:
            img = np.zeros((self.kb_camera.sz[0], self.kb_camera.sz[1], 3), dtype=np.uint8)
            for pt in xy.T:
                cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), 1)
            cv2.imwrite('fisheye_opencv.png', img)

        xyz2 = self.kb_camera.unproject(xy)
        s = xyz[2,:] / xyz2[2,:]
        xyz2 = s * xyz2

        err = np.abs(xyz - xyz2)
        ok_(np.max(err) < self.eps_mm)
