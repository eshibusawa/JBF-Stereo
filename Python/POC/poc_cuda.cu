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

// References
// [1] Shibahara, T., Numa, N., Nagashima, S., Aoki, T., Nakajima, H., & Kobayashi, K. (2008).
//     一次元位相限定相関法に基づくステレオ画像の高精度サブピクセル対応付け手法
//     [A high-accuracy sub-pixel correspondence technique using 1D phase-only correlation].
//     電子情報通信学会論文誌 D [IEICE Transactions on Information and Systems (Japanese edition)], J91-D(9), 2343–2356. (In Japanese)

// The window width POC_WINDOW_WIDTH is a power of two between 8 and 64.
// A window is transformed by POC_LANES (= min(width, 32)) lanes of a warp, every lane keeps POC_REGS (= width / POC_LANES)
// points and the radix-2 butterflies exchange the points between the lanes with __shfl_xor_sync.
// A warp handles POC_WINDOWS_PER_WARP (= 32 / POC_LANES) windows, i.e. neighbouring pixels along x (blockDim.x is 32).
#define POC_FULL_MASK 0xffffffffu

__constant__ float g_HannWindow[POC_WINDOW_WIDTH];
__constant__ float2 g_rotator[POC_WINDOW_WIDTH];

__device__ inline float2 complexMultiply(float2 a, float2 b)
{
	return make_float2(fmaf(a.x, b.x, -(a.y * b.y)), fmaf(a.x, b.y, a.y * b.x));
}

__device__ inline int bitReverse(int i)
{
	return static_cast<int>(__brev(static_cast<unsigned int>(i)) >> (32 - POC_LOG2_WINDOW_WIDTH));
}

// In-place radix-2 decimation-in-time FFT of a window distributed over the lanes (sign = 1: forward, -1: inverse without 1/N).
// The point i = laneInWindow + POC_LANES * r is held by v[r]. The input must be loaded in the bit-reversed order
// (point i holds the element bitReverse(i)), the output is in the natural order.
__device__ void warpFFT(int sign, float2 (&v)[POC_REGS])
{
	const int laneInWindow = threadIdx.x & (POC_LANES - 1);
	# pragma unroll
	for (int h = 1; h < POC_LANES; h <<= 1)
	{
		const bool upper = (laneInWindow & h) != 0;
		float2 w = g_rotator[(laneInWindow & (h - 1)) * (POC_WINDOW_WIDTH / (2 * h))];
		w.y = sign * w.y;
		# pragma unroll
		for (int r = 0; r < POC_REGS; r++)
		{
			float2 other;
			other.x = __shfl_xor_sync(POC_FULL_MASK, v[r].x, h, POC_LANES);
			other.y = __shfl_xor_sync(POC_FULL_MASK, v[r].y, h, POC_LANES);
			const float2 a = upper ? other : v[r];
			const float2 t = complexMultiply(w, upper ? v[r] : other);
			v[r] = upper ? make_float2(a.x - t.x, a.y - t.y) : make_float2(a.x + t.x, a.y + t.y);
		}
	}
#if POC_REGS == 2
	// the last stage (h = 32) combines the two points of a lane
	{
		float2 w = g_rotator[laneInWindow];
		w.y = sign * w.y;
		const float2 t = complexMultiply(w, v[1]);
		v[1] = make_float2(v[0].x - t.x, v[0].y - t.y);
		v[0] = make_float2(v[0].x + t.x, v[0].y + t.y);
	}
#endif
}

extern "C" __global__ void applyTransformation(
	float2* output,
	cudaTextureObject_t texImage,
	int width,
	int height)
{
	const int laneInWindow = threadIdx.x & (POC_LANES - 1);
	const int indexX = blockIdx.x * POC_WINDOWS_PER_WARP + (threadIdx.x / POC_LANES);
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	// the lanes of the out of range windows take part in the shuffles
	const bool valid = (indexX < width) && (indexY < height);

	float2 v[POC_REGS];
	# pragma unroll
	for (int r = 0; r < POC_REGS; r++)
	{
		const int n = bitReverse(laneInWindow + POC_LANES * r);
		float value = 0;
		if (valid)
		{
#if POC_NORMALIZED_COORDS
			value = tex2D<POC_PIXEL_TYPE>(texImage, (indexX + n - (POC_WINDOW_WIDTH / 2) + 0.5f) / width, (indexY + 0.5f) / height);
#else
			value = tex2D<POC_PIXEL_TYPE>(texImage, indexX + n - (POC_WINDOW_WIDTH / 2), indexY);
#endif
#if POC_USE_HANN_WINDOW
			value = g_HannWindow[n] * value;
#endif
		}
		v[r] = make_float2(value, 0);
	}
#if POC_USE_DC_CUT
	// subtract the mean of the window (weighted by the Hann window): the windowed DC is then 0 and does not leak into k = +-1.
	// v holds w * x here, so the mean is (sum of v) / (sum of w) and w * mean is subtracted; the sum is a reduction over the lanes of the window
	float sum = 0;
	# pragma unroll
	for (int r = 0; r < POC_REGS; r++)
	{
		sum += v[r].x;
	}
	# pragma unroll
	for (int offset = POC_LANES / 2; offset > 0; offset >>= 1)
	{
		sum += __shfl_xor_sync(POC_FULL_MASK, sum, offset, POC_LANES);
	}
	const float mean = sum / POC_WINDOW_SUM;
	# pragma unroll
	for (int r = 0; r < POC_REGS; r++)
	{
#if POC_USE_HANN_WINDOW
		v[r].x -= g_HannWindow[bitReverse(laneInWindow + POC_LANES * r)] * mean;
#else
		v[r].x -= mean;
#endif
	}
#endif
	warpFFT(1, v);

	if (valid)
	{
		const int index = indexX * POC_WINDOW_WIDTH + indexY * (width * POC_WINDOW_WIDTH);
		# pragma unroll
		for (int r = 0; r < POC_REGS; r++)
		{
			output[index + laneInWindow + POC_LANES * r] = v[r];
		}
	}
}

extern "C" __global__ void getPhaseOnlyCorrelation(
	float* output,
	const float2* __restrict__ inputRef,
	const float2* __restrict__ inputOther,
	const int* __restrict__ disparity,
	int width,
	int height)
{
	const int laneInWindow = threadIdx.x & (POC_LANES - 1);
	const int indexX = blockIdx.x * POC_WINDOWS_PER_WARP + (threadIdx.x / POC_LANES);
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	const bool valid = (indexX < width) && (indexY < height);

	// every lane accumulates the complex correlation of its frequency bin(s) over the averaging window height;
	// the point i holds the bin bitReverse(i) because the inverse FFT below takes its input in the bit-reversed order
	float2 v[POC_REGS];
	# pragma unroll
	for (int r = 0; r < POC_REGS; r++)
	{
		v[r] = make_float2(0, 0);
		const int k = bitReverse(laneInWindow + POC_LANES * r);
#if POC_USE_SPECTRUM_WEIGHTING
		if (valid && ((k < POC_WINDOW_WIDTH / 4) || (k >= POC_WINDOW_WIDTH - POC_WINDOW_WIDTH / 4)))
#else
		if (valid)
#endif
		{
			# pragma unroll
			for(int j = -POC_AVERAGING_WINDOW_HEIGHT/2; j <= POC_AVERAGING_WINDOW_HEIGHT/2; j++)
			{
				int indexYRef = indexY + j;
				indexYRef = (indexYRef < 0) ? 0 : indexYRef;
				indexYRef = (indexYRef >= height) ? height - 1 : indexYRef;
				int indexXOther = indexX - disparity[indexX + indexY * width];
				indexXOther = (indexXOther < 0) ? 0 : indexXOther;
				indexXOther = (indexXOther >= width) ? width - 1 : indexXOther;

				const float2 a = inputRef[indexX * POC_WINDOW_WIDTH + indexYRef * (width * POC_WINDOW_WIDTH) + k];
				const float2 b = inputOther[indexXOther * POC_WINDOW_WIDTH + indexYRef * (width * POC_WINDOW_WIDTH) + k];
				v[r].x = fmaf(a.x, b.x, v[r].x);
				v[r].x = fmaf(a.y, b.y, v[r].x);
				v[r].y = fmaf(a.y, b.x, v[r].y);
				v[r].y = fmaf(a.x, -b.y, v[r].y);
			}
		}
		float scale = 1 / (hypotf(v[r].x, v[r].y) + 1E-7) / POC_WINDOW_WIDTH;
		v[r].x *= scale;
		v[r].y *= scale;
#if POC_USE_DC_CUT
		if (k == 0)
		{
			// the DC was cut from the signal; the peak model (and the peak evaluation formula) assumes the bin of a window
			// with an offset, whose normalized value is 1 (phase 0), so restore it (1 / N after the scale above)
			v[r].x = 1.0f / POC_WINDOW_WIDTH;
			v[r].y = 0;
		}
#endif
	}

	// iFFT / FFTShift
	warpFFT(-1, v);
	if (valid)
	{
		const int index = indexX * POC_WINDOW_WIDTH + indexY * (width * POC_WINDOW_WIDTH);
		# pragma unroll
		for (int r = 0; r < POC_REGS; r++)
		{
			output[index + ((laneInWindow + POC_LANES * r + POC_WINDOW_WIDTH / 2) & (POC_WINDOW_WIDTH - 1))] = v[r].x;
		}
	}
}

extern "C" __global__ void getDisparity(
	float* output,
	float* outputValue,
	const float* __restrict__ input,
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
	output[indexD] = disparity[indexD] + maxIndex - POC_WINDOW_WIDTH/2;
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
		float pd = -POC_MATH_PI * delta;
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
