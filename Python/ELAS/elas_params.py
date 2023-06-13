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

class elas_params:
    def __init__(self):
        self.disp_min              = 0
        self.disp_max              = 255
        self.support_threshold     = 0.95
        self.support_texture       = 10
        self.candidate_stepsize    = 5
        self.incon_window_size     = 5
        self.incon_threshold       = 5
        self.incon_min_support     = 5
        self.add_corners           = True
        self.grid_size             = 20
        self.beta                  = 0.02
        self.gamma                 = 5
        self.sigma                 = 1
        self.sradius               = 3
        self.match_texture         = 0
        self.lr_threshold          = 2
        self.speckle_sim_threshold = 1
        self.speckle_size          = 200
        self.ipol_gap_width        = 5000
        self.filter_median         = True
        self.filter_adaptive_mean  = False
        self.postprocess_only_left = False
        self.subsampling           = False
