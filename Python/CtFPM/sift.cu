// This file is part of JBF-Stereo.
// Copyright (c) 2026, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
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

// References
// [1] Hu, Y., Song, R., & Li, Y. (2016). Efficient coarse-to-fine patchmatch for large displacement optical flow.
//     In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 5704-5712).

__constant__ float g_SIFTSin[(SIFT_NBINS)];
__constant__ float g_SIFTCos[(SIFT_NBINS)];

extern "C" __global__ void computeGradient(float3 *output,
    const float3 * __restrict__ inputX,
    const float3 * __restrict__ inputY,
	int length)
{
	const int indexL = blockIdx.x * blockDim.x + threadIdx.x;
	if (indexL >= length)
	{
		return;
	}
    float3 mag = make_float3(
        hypotf(inputX[indexL].x, inputY[indexL].x),
        hypotf(inputX[indexL].y, inputY[indexL].y),
        hypotf(inputX[indexL].z, inputY[indexL].z));
    float magMax = (mag.x < (SIFT_GRAD_EPS)) ? (SIFT_GRAD_EPS) : mag.x;
    float2 grad = make_float2(inputX[indexL].x / magMax, inputY[indexL].x / magMax);
    if (mag.y > magMax)
    {
        magMax = mag.y;
        grad.x = inputX[indexL].y / magMax;
        grad.y = inputY[indexL].y / magMax;
    }
    if (mag.z > magMax)
    {
        magMax = mag.z;
        grad.x = inputX[indexL].z / magMax;
        grad.y = inputY[indexL].z / magMax;
    }

    output[indexL] = make_float3(magMax, grad.x, grad.y);
}

extern "C" __global__ void computeBand(float *output,
    const float3 * __restrict__ input,
	int length)
{
	const int indexB = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexL = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexL >= length) || (indexB >= (SIFT_NBINS)))
	{
		return;
	}
    const int index = indexL * (SIFT_NBINS) + indexB;

    float3 grad = input[indexL];
    float tmp = fmaxf(fmaf(grad.y, g_SIFTCos[indexB], grad.z * g_SIFTSin[indexB]), 0);
    {
        tmp = powf(tmp, (SIFT_ALPHA));
    }
    output[index] = tmp * grad.x;
}

// grid: (SIFT width, SIFT height), block: (4, 4). One block builds one descriptor,
// one thread handles one of its 4x4 cells (16 cells x SIFT_NBINS bins).
extern "C" __global__ void computeSIFT(unsigned char *output,
    const float * __restrict__ input,
    int width,
    int height)
{
    const int cell = threadIdx.y * 4 + threadIdx.x;
    const int y = min(max((SIFT_YSIFT) + (int)blockIdx.y * (SIFT_STEPSIZE) + ((int)threadIdx.y - 1) * (SIFT_CELLSIZE), 0), height - 1);
    const int x = min(max((SIFT_XSIFT) + (int)blockIdx.x * (SIFT_STEPSIZE) + ((int)threadIdx.x - 1) * (SIFT_CELLSIZE), 0), width - 1);
    const float *src = input + ((size_t)y * width + x) * (SIFT_NBINS);

    float v[(SIFT_NBINS)];
    float sum = 0;
    for (int k = 0; k < (SIFT_NBINS); k++)
    {
        v[k] = src[k];
        sum += v[k] * v[k];
    }
    // the 16 threads of a block are lanes 0-15 of one warp
    for (int offset = 8; offset > 0; offset >>= 1)
    {
        sum += __shfl_xor_sync(0xffff, sum, offset);
    }

    const float magnitudeEPS = (SIFT_MAGNITUDE_EPS);
    const float denom = sqrtf(sum) + magnitudeEPS;
    unsigned char *dst = output + ((size_t)blockIdx.y * gridDim.x + blockIdx.x) * (SIFT_DIM) + cell * (SIFT_NBINS);
    for (int k = 0; k < (SIFT_NBINS); k++)
    {
        dst[k] = (unsigned char)fminf(v[k] / denom * 255.0f, 255.0f);
    }
}
