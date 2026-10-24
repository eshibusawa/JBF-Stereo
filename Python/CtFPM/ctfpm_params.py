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

# References
# [1] Hu, Y., Song, R., & Li, Y. (2016). Efficient coarse-to-fine patchmatch for large displacement optical flow.
#     In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 5704-5712).

class ctfpm_params:
    def __init__(self):
        self.ratio = 0.5
        self.min_width = 30
        self.step = 3
        self.max_iters = 8
        self.stop_iter_ratio = 0.05
        # the search is symmetric around 0 (initial range and search radius from max_disparity); the matches whose disparity
        # is out of [min_disparity, max_disparity] are removed by the cross check
        self.min_disparity = 0
        self.max_disparity = 400
        self.check_threshold = 3
        self.border_width = 5
        self.rand_seed = 0
        self.grad_eps = 1E-7
        self.magnitude_eps = 1E-2
        self.sift_bins = 8
        self.sift_alpha = 9
        self.sift_step_size = 1
        self.sift_cell_size = 2
        self.sift_is_boundary_included = True
