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

import cupy as cp

def create_texture_object(img_gpu,
    addressMode = cp.cuda.runtime.cudaAddressModeBorder,
    filterMode = cp.cuda.runtime.cudaFilterModePoint,
    readMode = cp.cuda.runtime.cudaReadModeElementType):
    if img_gpu.dtype == cp.uint8:
        channel_format_descriptor = cp.cuda.texture.ChannelFormatDescriptor(8, 0, 0, 0, cp.cuda.runtime.cudaChannelFormatKindUnsigned)
    elif img_gpu.dtype == cp.int16:
        channel_format_descriptor = cp.cuda.texture.ChannelFormatDescriptor(16, 0, 0, 0, cp.cuda.runtime.cudaChannelFormatKindSigned)
    else:
        return None
    img_gpu_2d = cp.cuda.texture.CUDAarray(channel_format_descriptor, img_gpu.shape[1], img_gpu.shape[0])
    img_gpu_2d.copy_from(img_gpu)

    img_rd = cp.cuda.texture.ResourceDescriptor(cp.cuda.runtime.cudaResourceTypeArray,
        cuArr = img_gpu_2d)
    img_td = cp.cuda.texture.TextureDescriptor(addressModes = (addressMode, addressMode),
        filterMode=filterMode,
        readMode=readMode,
        normalizedCoords = 0)
    img_to = cp.cuda.texture.TextureObject(img_rd, img_td)
    return img_to