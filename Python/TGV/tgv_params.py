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

# References
# [1] Kuschk, G., & Cremers, D. (2013). Fast and accurate large-scale stereo reconstruction using variational methods.
#     In Proceedings of the IEEE International Conference on Computer Vision Workshops (pp. 700-707).
# [2] Yoon, K. J., & Kweon, I. S. (2006). Adaptive support-weight approach for correspondence search.
#     IEEE transactions on pattern analysis and machine intelligence, 28(4), 650-656.
# [3] Werlberger, M., Trobin, W., Pock, T., Wedel, A., Cremers, D., & Bischof, H. (2009).
#     Anisotropic Huber-L1 Optical Flow. In BMVC (Vol. 1, No. 2, p. 3).

class tgv_params:
    def __init__(self):
        self.max_disparity = 64
        self.census_radius = 3
        self.aggregation_radius = 7
        self.aggregation_gamma_c = 14
        self.aggregation_gamma_p = 8
        self.aggregation_k = 5
        # settings of [1]
        self.max_iteration = 80
        self.max_smooth_iteration = 150
        self.enable_flipud = True
        # settings of [3]
        self.tensor_a = 5.0
        self.tensor_b = 0.5
        # Middlebury settings of [1]
        self.tau_u = 1/math.sqrt(12)
        self.tau_v = 1/math.sqrt(8)
        self.tau_p = 1/math.sqrt(12)
        self.tau_q = 1/math.sqrt(8)
        self.lambda_d = 1.0
        self.lambda_s = 0.2
        self.lambda_a = 8 * self.lambda_s
        self.beta = 1E-3
