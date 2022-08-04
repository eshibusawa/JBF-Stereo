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
		output[index + i].x = g_HannWindow[i] * tex2D<POC_PIXEL_TYPE>(texImage, indexT, indexY);
#else
		output[index + i].x = tex2D<POC_PIXEL_TYPE>(texImage, indexT, indexY);
#endif
		output[index + i].y = 0;
	}
	_FFT(1, output + index);
}

extern "C" __global__ void getPhaseOnlyCorrelation(
	float* output,
	const float2* __restrict__ inputRef,
	const float2* __restrict__ inputOther,
	const int* __restrict__ disparity,
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
	int indexRef = 0, indexOther = 0;
	// compute complex correlation
	float2 tmp[POC_WINDOW_WIDTH] = {};
	# pragma unroll
	for(int j = -POC_AVERAGING_WINDOW_HEIGHT/2; j <= POC_AVERAGING_WINDOW_HEIGHT/2; j++)
	{
		int indexYRef = indexY + j;
		indexYRef = (indexYRef < 0) ? 0 : indexYRef;
		indexYRef = (indexYRef >= height) ? height - 1 : indexYRef;
		int indexXOther = indexX - disparity[indexX + indexY * width];
		indexXOther = (indexXOther < 0) ? 0 : indexXOther;

		indexRef = indexX * POC_WINDOW_WIDTH + indexYRef * (width * POC_WINDOW_WIDTH);
		indexOther = indexXOther * POC_WINDOW_WIDTH + indexYRef * (width * POC_WINDOW_WIDTH);

#if POC_USE_SPECTRUM_WEIGHTING
		# pragma unroll
		for(int i = 0, ii = POC_WINDOW_WIDTH - 1; i < POC_WINDOW_WIDTH/4; i++, ii--)
		{
			tmp[i].x = fmaf(inputRef[indexRef + i].x, inputOther[indexOther + i].x, tmp[i].x);
			tmp[i].x = fmaf(inputRef[indexRef + i].y, inputOther[indexOther + i].y, tmp[i].x);
			tmp[i].y = fmaf(inputRef[indexRef + i].y, inputOther[indexOther + i].x, tmp[i].y);
			tmp[i].y = fmaf(inputRef[indexRef + i].x, -inputOther[indexOther + i].y, tmp[i].y);
			tmp[ii].x = fmaf(inputRef[indexRef + ii].x, inputOther[indexOther + ii].x, tmp[ii].x);
			tmp[ii].x = fmaf(inputRef[indexRef + ii].y, inputOther[indexOther + ii].y, tmp[ii].x);
			tmp[ii].y = fmaf(inputRef[indexRef + ii].y, inputOther[indexOther + ii].x, tmp[ii].y);
			tmp[ii].y = fmaf(inputRef[indexRef + ii].x, -inputOther[indexOther + ii].y, tmp[ii].y);
		}
#else
		# pragma unroll
		for(int i = 0; i < POC_WINDOW_WIDTH; i++)
		{
			tmp[i].x = fmaf(inputRef[indexRef + i].x, inputOther[indexOther + i].x, tmp[i].x);
			tmp[i].x = fmaf(inputRef[indexRef + i].y, inputOther[indexOther + i].y, tmp[i].x);
			tmp[i].y = fmaf(inputRef[indexRef + i].y, inputOther[indexOther + i].x, tmp[i].y);
			tmp[i].y = fmaf(inputRef[indexRef + i].x, -inputOther[indexOther + i].y, tmp[i].y);
		}
#endif
	}
	float scale = 0;
	# pragma unroll
	for(int i = 0; i < POC_WINDOW_WIDTH; i++)
	{
		scale = 1 / (hypotf(tmp[i].x, tmp[i].y) + 1E-7) / POC_WINDOW_WIDTH;
		tmp[i].x *= scale;
		tmp[i].y *= scale;
	}

	// iFFT / FFTShift
	_FFT(-1, tmp);
	# pragma unroll
	for(int i = 0; i < POC_WINDOW_WIDTH/2; i++)
	{
		output[index + i] = tmp[i + POC_WINDOW_WIDTH/2].x;
		output[index + i + POC_WINDOW_WIDTH/2] = tmp[i].x;
	}
}

extern "C" __global__ void getDisparity(
	float* output,
	float* outputValue,
	const float* __restrict__ input,
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
	const int indexD = indexX + indexY * width;
	float maxValue = -(1 << 30);
	int maxIndex = 0;
	# pragma unroll
	for(int i = 0; i < POC_WINDOW_WIDTH; i++)
	{
		if (input[index + i] > maxValue)
		{
			maxValue = input[index + i];
			maxIndex = i;
		}
	}
	output[indexD] = maxIndex - POC_WINDOW_WIDTH/2;
	outputValue[indexD] = maxValue;

	if ((maxIndex != 0) && (maxIndex != POC_WINDOW_WIDTH -1))
	{
		float v = input[index + maxIndex - 1] - input[index + maxIndex + 1];
#if POC_USE_SPECTRUM_WEIGHTING
		float u = input[index + maxIndex - 1] + input[index + maxIndex + 1];
#else
		float u = input[index + maxIndex - 1] + input[index + maxIndex + 1] + 2 * maxValue;
#endif
		float delta = v/u;
		output[indexD] -= delta;
		float pd = -static_cast<float>(3.141592653589793) * delta;
#if POC_USE_SPECTRUM_WEIGHTING
		float flac = sinf(pd/2);
#else
		float flac = sinf(pd);
#endif
		if (fabsf(flac) < 1E-7)
		{
#if POC_USE_SPECTRUM_WEIGHTING
			outputValue[indexD] = 2 * maxValue;
#else
			outputValue[indexD] = maxValue;
#endif
		}
		else
		{
			outputValue[indexD] = maxValue * pd / flac;
		}
	}
}
