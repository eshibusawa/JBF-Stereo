// This file is part of JBF-Stereo.
// Copyright (c) 2023, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
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
// [1] Kuschk, G., & Cremers, D. (2013). Fast and accurate large-scale stereo reconstruction using variational methods.
//     In Proceedings of the IEEE International Conference on Computer Vision Workshops (pp. 700-707).
// [2] Yoon, K. J., & Kweon, I. S. (2006). Adaptive support-weight approach for correspondence search.
//     IEEE transactions on pattern analysis and machine intelligence, 28(4), 650-656.
// [3] Werlberger, M., Trobin, W., Pock, T., Wedel, A., Cremers, D., & Bischof, H. (2009).
//     Anisotropic Huber-L1 Optical Flow. In BMVC (Vol. 1, No. 2, p. 3).

#include <cub/block/block_reduce.cuh>

extern "C" __global__ void updateAuxiliaryVariable(
	float *a,
	float *L,
	float theta_n,
	const float * __restrict__ cost,
	const float * __restrict__ u,
	int height,
	int width)
{
	const int indexX = blockIdx.x;
	const int indexY = blockIdx.y;
	const int indexD = threadIdx.x;
	const int indexXY = indexX + indexY * width;
	const int indexXYD = indexXY * (TGV_MAX_DISPARITY) + indexD;

	float dataTerm = static_cast<float>(TGV_LAMBDA_D) * cost[indexXYD] / static_cast<float>(TGV_CENSUS_LENGTH);
	const float u_a = (u[indexXY] - indexD / static_cast<float>(TGV_MAX_DISPARITY));
	const float smoothnessTerm = (L[indexXY] + u_a / (2 * theta_n)) * u_a;
	const float energy = dataTerm + smoothnessTerm;

	__shared__ float energyShared[(TGV_MAX_DISPARITY)];
	energyShared[indexD] = energy;
	__syncthreads();
	typedef cub::KeyValuePair<int, float> KeyValuePairT;
	typedef cub::BlockReduce<KeyValuePairT, TGV_MAX_DISPARITY> BlockReduceT;
	__shared__ typename BlockReduceT::TempStorage tempStorage;

	KeyValuePairT threadData(indexD, energy); // threadData.key, threadData.value
	KeyValuePairT minPair = BlockReduceT(tempStorage).Reduce(threadData, cub::ArgMin());

	if (indexD == 0)
	{
		const float m = static_cast<float>(TGV_MAX_DISPARITY);
		a[indexXY] = minPair.key / m;
		if ((minPair.key > 0) && ((minPair.key + 1 < (TGV_MAX_DISPARITY))))
		{
			// subdisparity refinemt Eq. 10 in [1]
			const float lambda = static_cast<float>(TGV_LAMBDA_D);
			const float &t_minus = energyShared[minPair.key - 1], &t = energyShared[minPair.key], &t_plus = energyShared[minPair.key + 1];
			const float aParabola = fmaf(0.5f, t_plus + t_minus, -t), bParabola = 0.5f * (t_plus - t_minus);
			const float t_nume = (u[indexXY] - a[indexXY]) / (theta_n * m) - fmaf(lambda, bParabola, L[indexXY] / m);
			const float t_denom = fmaf(2 * lambda, m * aParabola, 1.f / (theta_n * m));
			a[indexXY] += (t_nume / t_denom);
		}
		L[indexXY] = L[indexXY] + (u[indexXY] - a[indexXY]) / (2 * theta_n);
	}
}
