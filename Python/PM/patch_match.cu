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
// [1] Bleyer, M., Rhemann, C., & Rother, C. (2011, August).
//     Patchmatch stereo-stereo matching with slanted support windows. In Bmvc (Vol. 11, pp. 1-11).
// [2] Galliani, S., Lasinger, K., & Schindler, K. (2015).
//     Massively parallel multiview stereopsis by surface normal diffusion. In ICCV (pp. 873-881).

__constant__ int g_patchWidth(PM_PATCH_WIDTH);
__constant__ int g_patchHeight(PM_PATCH_HEIGHT);
__constant__ int g_radiusH(PM_PATCH_RADIUS_H);
__constant__ int g_radiusV(PM_PATCH_RADIUS_V);

__constant__ float g_maxDisparity(PM_MAX_DISPARITY);
__constant__ float g_minDisparity(PM_MIN_DISPARITY);
__constant__ float g_disparityRangePenalty(PM_DISPARITY_RANGE_PENALTY);

__constant__ float g_weightGamma(PM_WEIGHT_GAMMA); // gamma of Eq. (4) in [1]
__constant__ float g_blendingAlpha(PM_BLENDING_ALPHA); // alpha of Eq. (5) in [1]
__constant__ float g_truncateColor(PM_TRUNCATE_COLOR); // tau_col of Eq. (5) in [1]
__constant__ float g_truncateGrad(PM_TRUNCATE_GRAD); // tau_grad of Eq. (5) in [1]

__constant__ int g_propagationOffsetNum(8);
__constant__ int g_propagationOffsetX[] = { 0,  0, -1, -(PM_SPATIAL_DELTA), 0, 0, 1, (PM_SPATIAL_DELTA)}; // Fig. 2 (c) in [2]
__constant__ int g_propagationOffsetY[] = {-1, -(PM_SPATIAL_DELTA),  0,  0, 1, (PM_SPATIAL_DELTA), 0, 0}; // Fig. 2 (c) in [2]

inline __device__ float4 getRandomPlane(float minDisparity, float maxDisparity, unsigned long int &rs)
{
	float4 p;
	float s, c;
	sincosf(lsfr(rs), &s, &c);
	p.x = s;
	p.y = s;
	p.z = c;
	sincosf(lsfr(rs), &s, &c);
	p.x *= c;
	p.y *= s;
	p.w = unifBetween(minDisparity, maxDisparity, rs);
	return p;
}

inline __device__ float computePixelWeight(float pixelRef, float pixelOther)
{
	return expf(-fabsf(pixelRef - pixelOther) / g_weightGamma);
}

__device__ float computePixelCost(
	float2 pointRef,
	float2 pointOther,
	cudaTextureObject_t texRef,
	cudaTextureObject_t texOther,
	cudaTextureObject_t texGradRef,
	cudaTextureObject_t texGradOther)
{
	// Eq. (5)
	// color
	const float colorCost = fmin(fabsf(PM_PIXEL_SAMPLER<float>::get<255>(texRef, pointRef.x, pointRef.y) - PM_PIXEL_SAMPLER<float>::get<255>(texOther, pointOther.x, pointOther.y)), g_truncateColor);

	float2 gradRef = PM_PIXEL_SAMPLER<float2>::get(texGradRef, pointRef.x, pointRef.y);
	float2 gradOther = PM_PIXEL_SAMPLER<float2>::get(texGradOther, pointOther.x, pointOther.y);

	const float gradCost = fmin(fabsf(gradRef.x - gradOther.x) + fabsf(gradRef.y - gradOther.y), g_truncateGrad);

	return fmaf(1 - g_blendingAlpha, colorCost, g_blendingAlpha * gradCost);
}

__device__ float getPatchCost(
	const float2 pointRef,
	float4 plane,
	cudaTextureObject_t texRef,
	cudaTextureObject_t texOther,
	cudaTextureObject_t texGradRef,
	cudaTextureObject_t texGradOther)
{
	const float a = -plane.x / plane.z;
	const float b = -plane.y / plane.z;
	const float c = fmaf(plane.x, pointRef.x, fmaf(plane.y, pointRef.y, plane.z * plane.w)) / plane.z;
	float costSum = 0.f;
	for (int j = -g_radiusV; j <= g_radiusV; j++)
	{
		for (int i = -g_radiusH; i <= g_radiusH; i++)
		{
			float dp = fmaf(a, (pointRef.x + i), fmaf(b, (pointRef.y + j), c));
			if ((dp < g_minDisparity) || (dp > g_maxDisparity))
			{
				costSum += g_disparityRangePenalty;
				continue;
			}

			float2 pointOther;
			pointOther.x = pointRef.x + i - dp;
			pointOther.y = pointRef.y + j;

			// Eq. (4)
			const float weight = computePixelWeight(PM_PIXEL_SAMPLER<float>::get<255>(texRef, pointRef.x, pointRef.y),
				PM_PIXEL_SAMPLER<float>::get<255>(texRef, pointRef.x + i, pointRef.y + j));

			const float cost = computePixelCost(
				make_float2(pointRef.x + i, pointRef.y + j),
				pointOther,
				texRef, texOther, texGradRef, texGradOther);
			costSum = fmaf(weight, cost, costSum);
		}
	}

	return costSum;
}

extern "C" __global__ void getInitialPlanesAndCosts(
	float4 *planes,
	float *costs,
	unsigned long int *randomState,
	cudaTextureObject_t texRef,
	cudaTextureObject_t texOther,
	cudaTextureObject_t texGradRef,
	cudaTextureObject_t texGradOther,
	int height,
	int width)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;

	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}

	const int index = indexX + indexY * width;
	unsigned long int rs = randomState[index];
	float4 p = getRandomPlane(g_minDisparity, g_maxDisparity, rs);
	randomState[index] = rs;
	planes[index] = p;
	costs[index] = getPatchCost(make_float2(indexX, indexY), p, texRef, texOther, texGradRef, texGradOther);
}

__device__ void computeSpatialPropagation(
	float4 *planes,
	float *costs,
	int indexX,
	int indexY,
	cudaTextureObject_t texRef,
	cudaTextureObject_t texOther,
	cudaTextureObject_t texGradRef,
	cudaTextureObject_t texGradOther,
	int height,
	int width)
{
	const int index = indexX + indexY * width;
	for (int k = 0; k < g_propagationOffsetNum; k++)
	{
		const int indexOX = indexX + g_propagationOffsetX[k];
		const int indexOY = indexY + g_propagationOffsetY[k];
		if ((indexOX < 0) || (indexOY < 0) || (indexOX >= width) || (indexOY >= height))
		{
			continue;
		}

		const int indexO = indexOX + indexOY * width;
		const float updatedCost = getPatchCost(make_float2(indexX, indexY), planes[indexO], texRef, texOther, texGradRef, texGradOther);
		if (updatedCost < costs[index])
		{
			costs[index] = updatedCost;
			planes[index] = planes[indexO];
		}
	}
}

extern "C" __global__ void computeRedSpatialPropagation(
	float4 *planes,
	float *costs,
	cudaTextureObject_t texRef,
	cudaTextureObject_t texOther,
	cudaTextureObject_t texGradRef,
	cudaTextureObject_t texGradOther,
	int height,
	int width)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexHY = blockIdx.y * blockDim.y + threadIdx.y;
	int indexY = 2 * indexHY;
	if (indexX % 2 == 0)
	{
		indexY += 1;
	}
	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}
	computeSpatialPropagation(planes, costs, indexX, indexY, texRef, texOther, texGradRef, texGradOther, height, width);
}

extern "C" __global__ void computeBlackSpatialPropagation(
	float4 *planes,
	float *costs,
	cudaTextureObject_t texRef,
	cudaTextureObject_t texOther,
	cudaTextureObject_t texGradRef,
	cudaTextureObject_t texGradOther,
	int height,
	int width)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexHY = blockIdx.y * blockDim.y + threadIdx.y;
	int indexY = 2 * indexHY;
	if (indexX % 2 == 1)
	{
		indexY += 1;
	}
	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}
	computeSpatialPropagation(planes, costs, indexX, indexY, texRef, texOther, texGradRef, texGradOther, height, width);
}

__device__ void computeRandomSearch(
	float4 *planes,
	float *costs,
	unsigned long int *randomState,
	int indexX,
	int indexY,
	cudaTextureObject_t texRef,
	cudaTextureObject_t texOther,
	cudaTextureObject_t texGradRef,
	cudaTextureObject_t texGradOther,
	int height,
	int width)
{
	// p. 6, Plane Refinement of [1]
	const int index = indexX + indexY * width;
	const float2 p = make_float2(indexX, indexY);
	unsigned long int rs = randomState[index];

	float maxDeltaz0 = g_maxDisparity * 0.5f;
	float maxDeltan = 1.f;
	float norm = 0.f;
	while (maxDeltaz0 > .1f)
	{
		float4 planeCandidate = planes[index];
		// apply perturbation
		planeCandidate.x += unifBetween(-maxDeltan, maxDeltan, rs);
		planeCandidate.y += unifBetween(-maxDeltan, maxDeltan, rs);
		planeCandidate.z += unifBetween(-maxDeltan, maxDeltan, rs);
		planeCandidate.w += unifBetween(-maxDeltaz0, maxDeltaz0, rs);
		// normalization
		norm = norm3df(planeCandidate.x, planeCandidate.y, planeCandidate.z);
		planeCandidate.x /= norm;
		planeCandidate.y /= norm;
		planeCandidate.z /= norm;
		float updatedCost = getPatchCost(p, planeCandidate, texRef, texOther, texGradRef, texGradOther);
		if (updatedCost < costs[index])
		{
			costs[index] = updatedCost;
			planes[index] = planeCandidate;
		}

		maxDeltaz0 *= .5f;
		maxDeltan *= .5f;
	}

	randomState[index] = rs;
}

extern "C" __global__ void computeRedRandomSearch(
	float4 *planes,
	float *costs,
	unsigned long int *randomState,
	cudaTextureObject_t texRef,
	cudaTextureObject_t texOther,
	cudaTextureObject_t texGradRef,
	cudaTextureObject_t texGradOther,
	int height,
	int width)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexHY = blockIdx.y * blockDim.y + threadIdx.y;
	int indexY = 2 * indexHY;
	if (indexX % 2 == 0)
	{
		indexY += 1;
	}
	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}
	computeRandomSearch(planes, costs, randomState, indexX, indexY, texRef, texOther, texGradRef, texGradOther, height, width);
}

extern "C" __global__ void computeBlackRandomSearch(
	float4 *planes,
	float *costs,
	unsigned long int *randomState,
	cudaTextureObject_t texRef,
	cudaTextureObject_t texOther,
	cudaTextureObject_t texGradRef,
	cudaTextureObject_t texGradOther,
	int height,
	int width)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexHY = blockIdx.y * blockDim.y + threadIdx.y;
	int indexY = 2 * indexHY;
	if (indexX % 2 == 1)
	{
		indexY += 1;
	}
	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}
	computeRandomSearch(planes, costs, randomState, indexX, indexY, texRef, texOther, texGradRef, texGradOther, height, width);
}

extern "C" __global__ void computeDisparity(
	float *disparity,
	const float4* __restrict__ planes,
	int height,
	int width)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}

	const int index = indexX + indexY * width;
	const float a = -planes[index].x / planes[index].z;
	const float b = -planes[index].y / planes[index].z;
	const float c = fmaf(planes[index].x, indexX, fmaf(planes[index].y, indexY, planes[index].z * planes[index].w)) / planes[index].z;
	disparity[index] = fmaf(a, indexX, fmaf(b, indexY, c));
}
