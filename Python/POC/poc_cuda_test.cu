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

// Launch with one warp (block = (32, 1), grid = (1, 1)); the first window of the warp is stored.
extern "C" __global__ void applyFFTTest(
	float2* output,
	const float* __restrict__ input
)
{
	const int laneInWindow = threadIdx.x & (POC_LANES - 1);
	float2 v[POC_REGS];
	# pragma unroll
	for (int r = 0; r < POC_REGS; r++)
	{
		v[r] = make_float2(input[bitReverse(laneInWindow + POC_LANES * r)], 0);
	}
	warpFFT(1, v);
	if (threadIdx.x < POC_LANES)
	{
		# pragma unroll
		for (int r = 0; r < POC_REGS; r++)
		{
			output[laneInWindow + POC_LANES * r] = v[r];
		}
	}
}

extern "C" __global__ void applyInverseFFTTest(
	float2* output,
	const float2* __restrict__ input
)
{
	const int laneInWindow = threadIdx.x & (POC_LANES - 1);
	float2 v[POC_REGS];
	# pragma unroll
	for (int r = 0; r < POC_REGS; r++)
	{
		const float2 x = input[bitReverse(laneInWindow + POC_LANES * r)];
		v[r] = make_float2(x.x / POC_WINDOW_WIDTH, x.y / POC_WINDOW_WIDTH);
	}
	warpFFT(-1, v);
	if (threadIdx.x < POC_LANES)
	{
		# pragma unroll
		for (int r = 0; r < POC_REGS; r++)
		{
			output[laneInWindow + POC_LANES * r] = v[r];
		}
	}
}
