// This file is part of JBF-Stereo.
// Copyright (c) 2022-2026, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Reads the texel (x, y) of a texture of srcWidth x srcHeight; x and y may be outside of the image (address mode of the texture).
__device__ inline float fetch(cudaTextureObject_t tex, int x, int y, int srcWidth, int srcHeight)
{
#if POC_NORMALIZED_COORDS
	return tex2D<float>(tex, (x + 0.5f) / srcWidth, (y + 0.5f) / srcHeight);
#else
	return tex2D<float>(tex, x, y);
#endif
}

extern "C" __global__ void downSampling(
	float* output,
	cudaTextureObject_t texImage,
	int width,
	int height,
	int srcWidth,
	int srcHeight)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}
	const int index = indexX + indexY * width;
    float tmp;
    tmp = fmaf(0.25f, fetch(texImage, 2 * indexX,     2 * indexY,     srcWidth, srcHeight), 0);
    tmp = fmaf(0.25f, fetch(texImage, 2 * indexX + 1, 2 * indexY,     srcWidth, srcHeight), tmp);
    tmp = fmaf(0.25f, fetch(texImage, 2 * indexX,     2 * indexY + 1, srcWidth, srcHeight), tmp);
    tmp = fmaf(0.25f, fetch(texImage, 2 * indexX + 1, 2 * indexY + 1, srcWidth, srcHeight), tmp);
    output[index] = tmp;
}

extern "C" __global__ void upSampling(
	float* output,
	cudaTextureObject_t texDisparity,
	int width,
	int height,
	int srcWidth,
	int srcHeight)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}
	const int index = indexX + indexY * width;
    output[index] = 2 * fetch(texDisparity, indexX/2, indexY/2, srcWidth, srcHeight);
}
