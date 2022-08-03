// This file is part of JBF-Stereo.
// Copyright (c) 2022, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
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

__constant__ float g_HannWindow[POC_WINDOW_WIDTH];
__constant__ float2 g_rotator[POC_WINDOW_WIDTH];

__device__ void _FFT(int sign, float2 *arr)
{
	for (int i = 0, j = 1; j < POC_WINDOW_WIDTH - 1; j++)
	{
		for (int k = POC_WINDOW_WIDTH >> 1; k > (i ^= k); k >>= 1);
		if (j < i)
		{
			float2 tmp;
			tmp.x = arr[j].x;
			tmp.y = arr[j].y;
			arr[j].x = arr[i].x;
			arr[j].y = arr[i].y;
			arr[i].x = tmp.x;
			arr[i].y = tmp.y;
		}
	}

	for (int mh = 1, m = 0; (m = mh << 1) <= POC_WINDOW_WIDTH; mh = m)
	{
		int indexReversed = 0;
		for (int i = 0; i < POC_WINDOW_WIDTH; i += m)
		{
			float2 w = g_rotator[indexReversed];
			w.y = sign * w.y;
			for (int k = POC_WINDOW_WIDTH >> 2; k > (indexReversed ^= k); k >>= 1);
			for (int j = i; j < mh + i; j++)
			{
				int k = j + mh;
				float2 tmp;
				tmp.x = arr[j].x - arr[k].x;
				tmp.y = arr[j].y - arr[k].y;
				arr[j].x += arr[k].x;
				arr[j].y += arr[k].y;
				arr[k].x = fmaf(w.x, tmp.x, -(w.y * tmp.y));
				arr[k].y = fmaf(w.x, tmp.y, w.y * tmp.x);
			}
		}
	}
}

__device__ void FFT(const float *input, float2 *output)
{
	# pragma unroll
	for(int i = 0; i < POC_WINDOW_WIDTH; i++)
	{
#if POC_USE_HANN_WINDOW
		output[i].x = g_HannWindow[i] * input[i];
#else
		output[i].x = input[i];
#endif
		output[i].y = 0;
	}
	_FFT(1, output);
}

__device__ void iFFT(const float2 *input, float2 *output)
{
	# pragma unroll
	for(int i = 0; i < POC_WINDOW_WIDTH; i++)
	{
		output[i].x = input[i].x / POC_WINDOW_WIDTH;
		output[i].y = input[i].y / POC_WINDOW_WIDTH;
	}
	_FFT(-1, output);
}

extern "C" __global__ void applyTransformation(
	float2* output,
	cudaTextureObject_t texImage,
	int width,
	int height)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}

	const int index = indexX * POC_WINDOW_WIDTH + indexY * (width * POC_WINDOW_WIDTH);
	# pragma unroll
	for(int i = 0; i < POC_WINDOW_WIDTH; i++)
	{
		int indexT = indexX + i - (POC_WINDOW_WIDTH / 2);
		// border mirror
		// indexT = (indexT < 0) ? -indexT : indexT;
		// indexT = (indexT >= width) ? (width - (indexT % width) - 2) : indexT;
#if POC_USE_HANN_WINDOW
		output[index + i].x = g_HannWindow[i] * tex2D<unsigned char>(texImage, indexT, indexY);
#else
		output[index + i].x = tex2D<unsigned char>(texImage, indexT, indexY);
#endif
		output[index + i].y = 0;
	}
	_FFT(1, output + index);
}
