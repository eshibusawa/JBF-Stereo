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

class opencv_fisheye():
    def __init__(self, ks, ds, sz):
        self.K = np.array([[ks[0], 0, ks[2]], [0, ks[1], ks[3]], [0, 0, 1]], dtype=np.float32)
        self.ds = np.copy(ds)
        self.sz = sz

    def project(self, X):
        dummy_rotation = np.zeros(3, dtype=np.float32)
        dummy_translation = dummy_rotation
        x, _ = cv2.fisheye.projectPoints(X.T[:,np.newaxis,:], dummy_rotation, dummy_translation, self.K, self.ds)

        return x[:,0,:].T

    def unproject(self, xy):
        xy2 = cv2.fisheye.undistortPoints(xy.T[:,np.newaxis,:], self.K, self.ds)
        xyz = np.vstack((xy2.T[:,0,:], np.ones(xy2.shape[0], dtype=xy2.dtype)))
        s = np.linalg.norm(xyz, axis=0)

        return xyz/s

    def unproject_rays(self, sz = None):
        if sz is None:
            sz = self.sz
        xy = np.empty((2, sz[0], sz[1]), dtype=np.float32)
        xy[0,:,:] = np.arange(0, sz[1])[np.newaxis,:]
        xy[1,:,:] = np.arange(0, sz[0])[:,np.newaxis]
        xy = xy.reshape(2, -1)

        return self.unproject(xy)

    @staticmethod
    def get_default_camera():
        # the following parameters are obtaind from:
        # https://github.com/menandro/vfs/blob/master/test_vfs/main.cpp
        ks = np.array([285.722, 286.759, 420.135, 403.394], dtype=np.float32)
        ds = np.array([-0.00659769, 0.0473251, -0.0458264, 0.00897725], dtype=np.float32)
        return opencv_fisheye(ks, ds, (800, 848))
