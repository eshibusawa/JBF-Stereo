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

__constant__ int g_prior[(ELAS_DISPARITY_MAX) - (ELAS_DISPARITY_MIN) + 1];

namespace
{
struct Left
{
	__device__ static inline void point(int3 p, float &x, float &y)
	{
		x = p.x;
		y = p.y;
	}

	__device__ static inline void pointRef(int3 p, float &x, float &y)
	{
		point(p, x, y);
	}

	__device__ static inline void pointOther(int3 p, float &x, float &y)
	{
		x = p.x - p.z;
		y = p.y;
	}

	__device__ static inline int warp(int u, int d)
	{
		return u - d;
	}

	__device__ static inline short disparityMaxValid(int u, int width)
	{
		return min((ELAS_DISPARITY_MAX), u - (ELAS_WINDOW_SIZE) - (ELAS_U_STEP));
	}

	__device__ static inline void descriptor(
			const unsigned char *d1,
			const unsigned char *d2,
			const unsigned char **dRef,
			const unsigned char **dOther
		)
	{
		*dRef = d1;
		*dOther = d2;
	}

	__device__ static inline void plane(
			const float *triangle,
			float *plane
		)
	{
		plane[0] = triangle[0];
		plane[1] = triangle[1];
		plane[2] = triangle[2];
		plane[3] = triangle[3];
	}

	__device__ static inline void triangleRef(
			const float *plane,
			float *triangle
		)
	{
		triangle[0] = plane[0];
		triangle[1] = plane[1];
		triangle[2] = plane[2];
	}

	__device__ static inline void triangleOther(
			const float *plane,
			float *triangle
		)
	{
		triangle[3] = plane[0];
		triangle[4] = plane[1];
		triangle[5] = plane[2];
	}
};

struct Right
{
	__device__ static inline void point(int3 p, float &x, float &y)
	{
		x = p.x - p.z;
		y = p.y;
	}

	__device__ static inline void pointRef(int3 p, float &x, float &y)
	{
		x = p.x;
		y = p.y;
	}

	__device__ static inline void pointOther(int3 p, float &x, float &y)
	{
		x = p.x + p.z;
		y = p.y;
	}

	__device__ static inline int warp(int u, int d)
	{
		return u + d;
	}

	__device__ static inline short disparityMaxValid(int u, int width)
	{
		return min((ELAS_DISPARITY_MAX), width - u - (ELAS_WINDOW_SIZE) - (ELAS_U_STEP));
	}

	__device__ static inline void descriptor(
			const unsigned char *d1,
			const unsigned char *d2,
			const unsigned char **dRef,
			const unsigned char **dOther
		)
	{
		*dRef = d2;
		*dOther = d1;
	}

	__device__ static inline void plane(
			const float *triangle,
			float *plane
		)
	{
		plane[0] = triangle[3];
		plane[1] = triangle[4];
		plane[2] = triangle[5];
		plane[3] = triangle[0];
	}

	__device__ static inline void triangleRef(
			const float *plane,
			float *triangle
		)
	{
		triangle[3] = plane[0];
		triangle[4] = plane[1];
		triangle[5] = plane[2];
	}

	__device__ static inline void triangleOther(
			const float *plane,
			float *triangle
		)
	{
		triangle[0] = plane[0];
		triangle[1] = plane[1];
		triangle[2] = plane[2];
	}
};

template<typename DisparityType>
struct Horizontal
{
	__device__ static inline int range(
			const DisparityType *p,
			int q0,
			int height,
			int width,
			const DisparityType **pMax,
			const DisparityType **pMin
		)
	{
		// q0 = v
		*pMax = p + (q0 + 1) * width;
		*pMin = p + q0 * width;
		return width;
	}

	__device__ static inline int index(int q0, int q1, int width)
	{
		return width * q0 + q1;
	}

	__device__ static inline int stride(int d, int width)
	{
		return d;
	}

	__device__ static inline int length(int height, int width)
	{
		return width;
	}
};

template<typename DisparityType>
struct Vertical
{
	__device__ static inline int range(
			const DisparityType *p,
			int q0,
			int height,
			int width,
			const DisparityType **pMax,
			const DisparityType **pMin
		)
	{
		// q0 = u
		*pMax = p + width * height;
		*pMin = p;
		return height;
	}

	__device__ static inline int index(int q0, int q1, int width)
	{
		return width * q1 + q0;
	}

	__device__ static inline int stride(int d, int width)
	{
		return d * width;
	}

	__device__ static inline int length(int height, int width)
	{
		return height;
	}
};
};

extern "C" __global__ void computeSobelGradient(
	uchar2 *gradient,
	cudaTextureObject_t tex,
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
	short du =
			tex2D<short>(tex, indexX - 1, indexY - 1)
			+ (tex2D<short>(tex, indexX - 1, indexY) << 1)
			+ tex2D<short>(tex, indexX - 1, indexY + 1)
			- tex2D<short>(tex, indexX + 1, indexY - 1)
			- (tex2D<short>(tex, indexX + 1, indexY) << 1)
			- tex2D<short>(tex, indexX + 1, indexY + 1);
	short dv =
			tex2D<short>(tex, indexX - 1, indexY - 1)
			+ (tex2D<short>(tex, indexX, indexY - 1) << 1)
			+ tex2D<short>(tex, indexX + 1, indexY - 1)
			- tex2D<short>(tex, indexX - 1, indexY + 1)
			- (tex2D<short>(tex, indexX, indexY + 1) << 1)
			- tex2D<short>(tex, indexX + 1, indexY + 1);

	du = max(du >> 2, -128);
	du = min(du, 127) + 128;
	dv = max(dv >> 2, -128);
	dv = min(dv, 127) + 128;
	gradient[index] = make_uchar2(static_cast<unsigned char>(du), static_cast<unsigned char>(dv));
}

extern "C" __global__ void computeDescriptor(
	unsigned char *output,
	const uchar2 * __restrict__ gradient,
	int height,
	int width)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	const int indexXp = indexX + (ELAS_DESCRIPTOR_PADDING); // padded
	const int indexYp = indexY + (ELAS_DESCRIPTOR_PADDING); // padded
	if ((indexXp >= width) || (indexYp >= height))
	{
		return;
	}

	const int indexG = width * indexYp + indexXp;
	const int indexD = indexG * (ELAS_DESCRIPTOR_LENGTH);
	unsigned char *outputP = output + indexD;
	const uchar2 *g2 = gradient + indexG;
	const uchar2 *g1 = g2 - width;
	const uchar2 *g0 = g1 - width;
	const uchar2 *g3 = g2 + width;
	const uchar2 *g4 = g3 + width;
	outputP[0]  = (*(g0 + 0)).x;
	outputP[1]  = (*(g1 - 2)).x;
	outputP[2]  = (*(g1 + 0)).x;
	outputP[3]  = (*(g1 + 2)).x;
	outputP[4]  = (*(g2 - 1)).x;
	outputP[5]  = (*(g2 + 0)).x;
	outputP[6]  = (*(g2 + 0)).x;
	outputP[7]  = (*(g2 + 1)).x;
	outputP[8]  = (*(g3 - 2)).x;
	outputP[9]  = (*(g3 + 0)).x;
	outputP[10] = (*(g3 + 2)).x;
	outputP[11] = (*(g4 + 0)).x;
	outputP[12] = (*(g1 + 0)).y;
	outputP[13] = (*(g2 - 1)).y;
	outputP[14] = (*(g2 + 1)).y;
	outputP[15] = (*(g3 + 0)).y;
}

__device__ inline int computeDescriptorSum(const unsigned char * __restrict__ d)
{
	// ideally this function should be implemented by using warp shuffle!
	int ret = 0;
#pragma unroll
for (int k = 0; k < (ELAS_DESCRIPTOR_LENGTH); k++)
	{
		ret += abs(static_cast<int>(d[k]) - 128);
	}
	return ret;
}

template<typename Side> __device__ inline short computeMatchingDisparity_(
	int u,
	int v,
	const unsigned char * __restrict__ descriptor1,
	const unsigned char * __restrict__ descriptor2,
	int height,
	int width
	)
{
	const unsigned char *dRef, *dOther;
	Side::descriptor(descriptor1, descriptor2, &dRef, &dOther);

	if ((u < (ELAS_WINDOW_SIZE) + (ELAS_U_STEP)) ||
		(u > width - 1 - (ELAS_WINDOW_SIZE) - (ELAS_U_STEP)) ||
		(v < (ELAS_WINDOW_SIZE) + (ELAS_V_STEP)) ||
		(v > height - 1 - (ELAS_WINDOW_SIZE) - (ELAS_V_STEP)))
	{
		return (ELAS_INVALID_DISPARITY);
	}

	int sum = computeDescriptorSum(dRef + (v * width + u) * (ELAS_DESCRIPTOR_LENGTH));
	if (sum < (ELAS_SUPPORT_TEXTURE))
	{
		return (ELAS_INVALID_DISPARITY);
	}

	const int offsetNW = (ELAS_DESCRIPTOR_LENGTH) * (-(ELAS_U_STEP) - width * (ELAS_V_STEP));
	const int offsetNE = (ELAS_DESCRIPTOR_LENGTH) * ( (ELAS_U_STEP) - width * (ELAS_V_STEP));
	const int offsetSW = (ELAS_DESCRIPTOR_LENGTH) * (-(ELAS_U_STEP) + width * (ELAS_V_STEP));
	const int offsetSE = (ELAS_DESCRIPTOR_LENGTH) * ( (ELAS_U_STEP) + width * (ELAS_V_STEP));
	const unsigned char *dRefL = dRef + (width * v + u) * (ELAS_DESCRIPTOR_LENGTH);
	const unsigned char *dRefNW = dRefL + offsetNW;
	const unsigned char *dRefNE = dRefL + offsetNE;
	const unsigned char *dRefSW = dRefL + offsetSW;
	const unsigned char *dRefSE = dRefL + offsetSE;
	short bestDisparity1 = (ELAS_INVALID_DISPARITY), bestDisparity2 = (ELAS_INVALID_DISPARITY);
	short bestCost1 = 32767, bestCost2 = 32767;
	short disparityMinValid = (ELAS_DISPARITY_MIN);
	short disparityMaxValid = Side::disparityMaxValid(u, width);
	const float supportThreshold = static_cast<float>((ELAS_SUPPORT_THRESHOLD));

	if (disparityMaxValid - disparityMinValid < 10)
	{
		return (ELAS_INVALID_DISPARITY);
	}

	for (short d = disparityMinValid; d <= disparityMaxValid; d++)
	{
		const int deltaDisparity = Side::warp(u, d);
		const unsigned char *dOtherL = dOther + (v * width + deltaDisparity) * (ELAS_DESCRIPTOR_LENGTH);
		const unsigned char *dOtherNW = dOtherL + offsetNW;
		const unsigned char *dOtherNE = dOtherL + offsetNE;
		const unsigned char *dOtherSW = dOtherL + offsetSW;
		const unsigned char *dOtherSE = dOtherL + offsetSE;
		int sad = 0;
		for (int k = 0; k < (ELAS_DESCRIPTOR_LENGTH); k++)
		{
			// ideally this function should be implemented by using warp shuffle!
			sad += abs(static_cast<short>(dRefNW[k]) - dOtherNW[k]);
			sad += abs(static_cast<short>(dRefNE[k]) - dOtherNE[k]);
			sad += abs(static_cast<short>(dRefSW[k]) - dOtherSW[k]);
			sad += abs(static_cast<short>(dRefSE[k]) - dOtherSE[k]);
		}

		if (sad < bestCost1)
		{
			bestCost2 = bestCost1;
			bestDisparity2 = bestDisparity1;
			bestCost1 = sad;
			bestDisparity1 = d;
		}
		else if (sad < bestCost2)
		{
			bestCost2 = sad;
			bestDisparity2 = d;
		}
	}
	short ret = (ELAS_INVALID_DISPARITY);
	if ((bestDisparity1 >= 0) && (bestDisparity2 >= 0) && (static_cast<float>(bestCost1) < (supportThreshold * bestCost2)))
	{
		ret = bestDisparity1;
	}

	return ret;
}

extern "C" __global__ void computeSupportDisparityLR(
	short *output,
	const unsigned char * __restrict__ descriptor1,
	const unsigned char * __restrict__ descriptor2,
	int heightCandidate,
	int widthCandidate,
	int height,
	int width)
{
	const int indexXc = blockIdx.x * blockDim.x + threadIdx.x + 1;
	const int indexYc = blockIdx.y * blockDim.y + threadIdx.y + 1;
	if ((indexXc >= widthCandidate) || (indexYc >= heightCandidate))
	{
		return;
	}

	short ret = (ELAS_INVALID_DISPARITY);
	const int indexC = indexYc * widthCandidate + indexXc;

	const int indexX = indexXc * (ELAS_CANDIDATE_STEPSIZE);
	const int indexY = indexYc * (ELAS_CANDIDATE_STEPSIZE);
	if ((indexX >= width) || (indexY >= height))
	{
		output[indexC] = ret;
		return;
	}

	short d = computeMatchingDisparity_<Left>(indexX, indexY, descriptor1, descriptor2, height, width);
	if (d >= 0)
	{
		short d2 = computeMatchingDisparity_<Right>(indexX - d, indexY, descriptor1, descriptor2, height, width);
		if ((d2 >= 0) && abs(d - d2) <= (ELAS_LR_THRESHOLD))
		{
			ret = d;
		}
	}
	output[indexC] = ret;
}

extern "C" __global__ void removeInconsistentSupportPoints(
	short *output,
	const short * __restrict__ input,
	int height,
	int width)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}

	const int index = indexY * width + indexX;
	const short d = input[index];
	if (d < 0)
	{
		output[index] = (ELAS_INVALID_DISPARITY);
		return;
	}

	int support = 0;
	const int uMinValid = max(0, indexX - (ELAS_INCON_WINDOW_SIZE));
	const int uMaxValid = min(width - 1, indexX + (ELAS_INCON_WINDOW_SIZE));
	const int vMinValid = max(0, indexY - (ELAS_INCON_WINDOW_SIZE));
	const int vMaxValid = min(height - 1, indexY + (ELAS_INCON_WINDOW_SIZE));
	for (int v = vMinValid; v <= vMaxValid; v++)
	{
		for (int u = uMinValid; u <= uMaxValid; u++)
		{
			const int index2 = v * width + u;
			short d2 = input[index2];
			support = (((d2 >= 0) && (abs(d - d2) <= (ELAS_INCON_THRESHOLD)) ) ? (support + 1) : support);
		}
	}
	short ret = d;
	if (support < (ELAS_INCON_MIN_SUPPORT))
	{
		ret = (ELAS_INVALID_DISPARITY);
	}
	output[index] = ret;
}

template<typename Direction> __device__ inline void removeRedundantSupportPoints_(
	short *output,
	int q0,
	int height,
	int width)
{
	const short *pMax = NULL, *pMin = NULL;
	const int length = Direction::range(output, q0, height, width, &pMax, &pMin);
	for (int q1 = 0; q1 < length; q1++)
	{
		const int index = Direction::index(q0, q1, width);
		const short *pd = output + index;
		if ((*pd) < 0)
		{
			continue;
		}

		bool isRedundant = true;
		const int directions[] = {-1, 1};
		for (int k = 0; k < 2; k++)
		{
			bool isSupported = false;
			const short *pd2 = pd;
			const int stride = Direction::stride(directions[k], width);
			for (int l = 0; l < (ELAS_MAX_REDUN_DIST); l++)
			{
				pd2 += stride;
				if ((pd2 < pMin) || (pd2 >= pMax))
				{
					break;
				}
				if (((*pd2) >= 0) && (abs((*pd) - (*pd2)) <= (ELAS_MAX_REDUN_THRESHOLD)))
				{
					isSupported = true;
					break;
				}
			}

			if (!isSupported)
			{
				isRedundant = false;
				break;
			}
		}

		if (isRedundant)
		{
			output[index] = (ELAS_INVALID_DISPARITY);
		}
	}
}

extern "C" __global__ void removeRedundantSupportPointsV(
	short *output,
	int height,
	int width)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	if (indexX >= width)
	{
		return;
	}
	removeRedundantSupportPoints_<Vertical<short> >(output, indexX, height, width);
}

extern "C" __global__ void removeRedundantSupportPointsH(
	short *output,
	int height,
	int width)
{
	const int indexY = blockIdx.x * blockDim.x + threadIdx.x;
	if (indexY >= height)
	{
		return;
	}
	removeRedundantSupportPoints_<Horizontal<short> >(output, indexY, height, width);
}

extern "C" __global__ void computeSupportFromCandidate(
	int3 *output,
	const short * __restrict__ input,
	int height,
	int width,
	int maxSupport)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x + 1;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y + 1;
	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}
	const int index = indexY * width + indexX;
	if (input[index] >= 0)
	{
		const int indexO = atomicAdd(&(output[maxSupport - 1].x), 1);
		output[indexO].x = indexX * (ELAS_CANDIDATE_STEPSIZE);
		output[indexO].y = indexY * (ELAS_CANDIDATE_STEPSIZE);
		output[indexO].z = input[index];
	}
}

extern "C" __global__ void addCornerToSupport(
	int3 *output,
	unsigned long long *work,
	const int3 * __restrict__ input,
	int length)
{
	const int indexL = blockIdx.x * blockDim.x + threadIdx.x;
	if ((indexL >= length))
	{
		return;
	}

	for (int k = 0; k < 4; k++)
	{
		int du = output[k].x - input[indexL].x;
		int dv = output[k].y - input[indexL].y;
		int d = du * du + dv * dv;
		unsigned long long dd = (static_cast<unsigned long long>(d) << 32) + static_cast<unsigned long long>(input[indexL].z);
		atomicMin(work + k, dd);
	}

	__syncthreads();
	if (indexL == 0)
	{
		for (int k = 0; k < 4; k++)
		{
			output[k].z = static_cast<int>(work[k] & 0x00000000FFFFFFFF);
		}

		output[4].x = output[2].x + output[2].z;
		output[4].y = output[2].y;
		output[4].z = output[2].z;
		output[5].x = output[3].x + output[3].z;
		output[5].y = output[3].y;
		output[5].z = output[3].z;
	}
}

__device__ inline void solve3x3_(float *x, const float *A, const float *b)
{
	// 3x3 linear solver based on the Cramer's rule
	// To solve this, it is practical to use Gaussian elimination
	const float det = A[0] * (A[4] * A[8] - A[5] * A[7]) + A[3] * (A[2] * A[7] - A[1] * A[8]) + A[6] * (A[1] * A[5] - A[2] * A[4]);
	if (fabsf(det) < 1E-7)
	{
		x[0] = x[1] = x[2] = 0.f;
		return;
	}
	float det1 = b[0] * (A[4] * A[8] - A[5] * A[7]) + b[1] * (A[2] * A[7] - A[1] * A[8]) + b[2] * (A[1] * A[5] - A[2] * A[4]);
	x[0] = det1 / det;
	det1 = A[0] * (b[1] * A[8] - A[5] * b[2]) + A[3] * (A[2] * b[2] - b[0] * A[8]) + A[6] * (b[0] * A[5] - A[2] * b[1]);
	x[1] = det1 / det;
	det1 = A[0] * (A[4] * b[2] - b[1] * A[7]) + A[3] * (b[0] * A[7] - A[1] * b[2]) + A[6] * (A[1] * b[1] - b[0] * A[4]);
	x[2] = det1 / det;
}

template<typename Side>
__device__ inline void computeDisparityPlaneLR_(
	float *output,
	const int3 * __restrict__ inputIndices,
	const int3 * __restrict__ inputPoints)
{
	float A[9], b[3], p[3];
	Side::pointRef(inputPoints[inputIndices->x], A[0], A[1]);
	Side::pointRef(inputPoints[inputIndices->y], A[3], A[4]);
	Side::pointRef(inputPoints[inputIndices->z], A[6], A[7]);
	A[2] = 1;
	A[5] = 1;
	A[8] = 1;
	b[0] = inputPoints[inputIndices->x].z;
	b[1] = inputPoints[inputIndices->y].z;
	b[2] = inputPoints[inputIndices->z].z;
	solve3x3_(p, A, b);
	Side::triangleRef(p, output);
	Side::pointOther(inputPoints[inputIndices->x], A[0], A[1]);
	Side::pointOther(inputPoints[inputIndices->y], A[3], A[4]);
	Side::pointOther(inputPoints[inputIndices->z], A[6], A[7]);
	solve3x3_(p, A, b);
	Side::triangleOther(p, output);
}

extern "C" __global__ void computeDisparityPlaneLR(
	float *output1,
	float *output2,
	const int3 * __restrict__ inputIndices1,
	const int3 * __restrict__ inputIndices2,
	const int3 * __restrict__ inputPoints,
	int length)
{
	const int indexL = blockIdx.x * blockDim.x + threadIdx.x;
	if ((indexL >= length))
	{
		return;
	}
	const int index = indexL * 6;
	computeDisparityPlaneLR_<Left>(output1 + index, inputIndices1 + indexL, inputPoints);
	computeDisparityPlaneLR_<Right>(output2 + index, inputIndices2 + indexL, inputPoints);
}

extern "C" __global__ void createGridMaskLR(
	unsigned char *output,
	const int3 * __restrict__ inputPoints,
	int length,
	int gridHeight,
	int gridWidth,
	int gridDepth)
{
	const int indexL = blockIdx.x * blockDim.x + threadIdx.x;
	if ((indexL >= length))
	{
		return;
	}
	int sz = gridHeight * gridWidth * gridDepth;
	unsigned char *w1 = output, *w2 = w1 + sz;

	const int dMin = max(0, inputPoints[indexL].z - 1);
	const int dMax = min(inputPoints[indexL].z + 1, (ELAS_DISPARITY_MAX));
	const int x1 = floor((inputPoints[indexL].x) / static_cast<float>(ELAS_GRID_SIZE));
	const int x2 = floor((inputPoints[indexL].x - inputPoints[indexL].z) / static_cast<float>(ELAS_GRID_SIZE));
	const int y = floor(inputPoints[indexL].y / static_cast<float>(ELAS_GRID_SIZE));
	for (int d = dMin; d <= dMax; d++)
	{
		if ((y >= 0) && (y < gridHeight))
		{
			if ((x1 >= 0) && (x1 < gridWidth))
			{
				w1[(y * gridWidth + x1) * gridDepth + d] = 1;
			}
			if ((x2 >= 0) && (x2 < gridWidth))
			{
				w2[(y * gridWidth + x2) * gridDepth + d] = 1;
			}
		}
	}
}

extern "C" __global__ void createGridLR(
	short *output,
	const unsigned char * __restrict__ input,
	int gridHeight,
	int gridWidth,
	int gridDepth)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexX >= gridWidth) || (indexY >= gridHeight))
	{
		return;
	}

	const unsigned char *inputs[] = {input, input + (gridHeight * gridWidth * gridDepth)};
	short *outputs[] = {output, output + (gridHeight * gridWidth * (gridDepth + 1))};
	for (int lr = 0; lr < 2; lr++)
	{
		const unsigned char *p11 = inputs[lr] + ((indexY * gridWidth + indexX) * gridDepth);
		const unsigned char *p10 = (indexX - 1 >= 0) ? p11 - gridDepth : p11;
		const unsigned char *p12 = (indexX + 1 < gridWidth) ? p11 + gridDepth : p11;
		const unsigned char *p01 = (indexY - 1 >= 0) ? p11 - gridWidth*gridDepth : p11;
		const unsigned char *p00 = (indexX - 1 >= 0) ? p01 - gridDepth : p11;
		const unsigned char *p02 = (indexX + 1 < gridWidth) ? p01 + gridDepth : p11;
		const unsigned char *p21 = (indexY + 1 < gridHeight) ? p11 + gridWidth*gridDepth : p11;
		const unsigned char *p20 = (indexX - 1 >= 0) ? p21 - gridDepth : p11;
		const unsigned char *p22 = (indexX + 1 < gridWidth) ? p21 + gridDepth : p11;

		short *q = outputs[lr] + ((indexY * gridWidth + indexX) * (gridDepth + 1));
		int index = 0;
		for (int k = 0; k < gridDepth; k++)
		{
			if ((*p00) || (*p01) || (*p02) || (*p10) || (*p11) || (*p12) || (*p20) || (*p21) || (*p22))
			{
				q[index + 1] = static_cast<short>(k);
				index++;
			}
			p00++;
			p01++;
			p02++;
			p10++;
			p11++;
			p12++;
			p20++;
			p21++;
			p22++;
		}
		q[0] = index;
	}
}

__device__ inline void sort3Points_(float *us, float *vs)
{
	float tmp;
#pragma unroll
	for (char l = 0; l < 3; l++)
	{
#pragma unroll
		for (char k = 0; k < l; k++)
		{
			if (us[k] > us[l])
			{
				tmp = us[l];
				us[l] = us[k];
				us[k] = tmp;
				tmp = vs[l];
				vs[l] = vs[k];
				vs[k] = tmp;
			}
		}
	}
}

template<int I, int J>
__device__ inline void getCoefficients_(float2 &c, const float *us, const float *vs)
{
	c.x = 0;
	if (static_cast<int>(us[I]) != static_cast<int>(us[J]))
	{
		c.x = (vs[I] - vs[J]) / (us[I] - us[J]);
	}
	c.y = vs[I] - c.x * us[I];
}

template<int I, int J>
__device__ inline int getUVCounts_(const float2 *Cs, const float *us, const float *vs, int width)
{
	int ret = 0;
	if (static_cast<int>(us[I]) == static_cast<int>(us[J]))
	{
		return ret;
	}

	for (int u = max(static_cast<int>(us[I]), 0); u < min(static_cast<int>(us[J]), width); u++)
	{
		int v1 = static_cast<int>(Cs[I].x * u + Cs[I].y);
		int v2 = static_cast<int>(Cs[J].x * u + Cs[J].y);
		int vmin = min(v1, v2);
		int vmax = max(v1, v2);
		for (int v = vmin; v < vmax; v++)
		{
			ret++;
		}
	}
	return ret;
}

template<int I, int J>
__device__ inline int getUVs_(int3 *output, const float2 *Cs, const float *us, const float *vs, int width, int valid)
{
	int ret = 0;
	if (static_cast<int>(us[I]) == static_cast<int>(us[J]))
	{
		return ret;
	}

	for (int u = max(static_cast<int>(us[I]), 0); u < min(static_cast<int>(us[J]), width); u++)
	{
		int v1 = static_cast<int>(Cs[I].x * u + Cs[I].y);
		int v2 = static_cast<int>(Cs[J].x * u + Cs[J].y);
		int vmin = min(v1, v2);
		int vmax = max(v1, v2);
		for (int v = vmin; v < vmax; v++)
		{
			output[ret].x = u;
			output[ret].y = v;
			output[ret].z = valid;
			ret++;
		}
	}
	return ret;
}

template<typename Side>
__device__ inline void computeTargetPixels_(
	int3 *outputUV,
	int *outputCount,
	const int3 * __restrict__ inputIndices,
	const int3 * __restrict__ inputPoints,
	int index,
	int length,
	int width
)
{
	float triangle_x[3], triangle_y[3];
	float2 Cs[3]; // {0, 1}, {0, 2}, {1, 2}
	int count = 0, offset = 0;
	Side::point(inputPoints[inputIndices[index].x], triangle_x[0], triangle_y[0]);
	Side::point(inputPoints[inputIndices[index].y], triangle_x[1], triangle_y[1]);
	Side::point(inputPoints[inputIndices[index].z], triangle_x[2], triangle_y[2]);
	sort3Points_(triangle_x, triangle_y);
	getCoefficients_<0, 1>(Cs[0], triangle_x, triangle_y);
	getCoefficients_<0, 2>(Cs[1], triangle_x, triangle_y);
	getCoefficients_<1, 2>(Cs[2], triangle_x, triangle_y);
	count = getUVCounts_<0, 1>(Cs, triangle_x, triangle_y, width);
	count += getUVCounts_<1, 2>(Cs, triangle_x, triangle_y, width);
	outputCount[index] = count;
	offset = atomicAdd(outputCount + length, count);
	count = getUVs_<0, 1>(outputUV + offset, Cs, triangle_x, triangle_y, width, index);
	getUVs_<1, 2>(outputUV + offset + count, Cs, triangle_x, triangle_y, width, index);
}

extern "C" __global__ void computeTargetPixelsLR(
	int3 *outputUV1,
	int3 *outputUV2,
	int *outputCount1,
	int *outputCount2,
	const int3 * __restrict__ inputIndices1,
	const int3 * __restrict__ inputIndices2,
	const int3 * __restrict__ inputPoints,
	int length,
	int height,
	int width
	)
{
	const int indexL = blockIdx.x * blockDim.x + threadIdx.x;
	if ((indexL >= length))
	{
		return;
	}
	computeTargetPixels_<Left>(outputUV1, outputCount1, inputIndices1, inputPoints, indexL, length, width);
	computeTargetPixels_<Right>(outputUV2, outputCount2, inputIndices2, inputPoints, indexL, length, width);
}

template<typename Side>
__device__ inline void findMatch_(
	short *output,
	int3 uv,
	const unsigned char * __restrict__ descriptor1,
	const unsigned char * __restrict__ descriptor2,
	const short * __restrict__ grid,
	const float * __restrict__ triangle,
	int width)
{
	const unsigned char *dRef, *dOther;
	Side::descriptor(descriptor1, descriptor2, &dRef, &dOther);
	const int sumT = computeDescriptorSum(dRef);
	if (sumT < (ELAS_MATCH_TEXTURE))
	{
		return;
	}

	const int nDisparity = grid[0];
	float plane[4];
	Side::plane(triangle, plane);
	bool valid = (fabsf(plane[0]) < 0.7f) && (fabsf(plane[3]) < 0.7f);
	const short disparityPlane = static_cast<short>(plane[0] * uv.x + plane[1] * uv.y + plane[2]);
	const short disparityPlaneMin = max(disparityPlane - (ELAS_PLANE_RADIUS), (ELAS_DISPARITY_MIN));
	const short disparityPlaneMax = min(disparityPlane + (ELAS_PLANE_RADIUS), (ELAS_DISPARITY_MAX) - 1);
	short bestDisparity = (ELAS_INVALID_DISPARITY);
	int bestCost = 1 << 30;
	for (int k = 0; k < nDisparity; k++)
	{
		short d = grid[k + 1];
		if ((d < disparityPlaneMin) || (d > disparityPlaneMax))
		{
			int warped = Side::warp(uv.x, d);
			int rd = warped - uv.x;
			if ((warped < 2) || (warped >= (width - 2)))
			{
				continue;
			}
			const unsigned char *dOtherD = dOther + (ELAS_DESCRIPTOR_LENGTH) * rd;
			int sum = 0;
			for (int l = 0; l < (ELAS_DESCRIPTOR_LENGTH); l++)
			{
				sum += abs(static_cast<short>(dRef[l]) - dOtherD[l]);
			}
			if (sum < bestCost)
			{
				bestCost = sum;
				bestDisparity = d;
			}
		}
	}
	for (short d = disparityPlaneMin; d <= disparityPlaneMax; d++)
	{
		int warped = Side::warp(uv.x, d);
		int rd = warped - uv.x;
		if ((warped < 2) || (warped >= (width - 2)))
		{
			continue;
		}
		int sum = ((valid ? g_prior[abs(d - disparityPlane)] : 0));
		const unsigned char *dOtherD = dOther + (ELAS_DESCRIPTOR_LENGTH) * rd;
		for (int l = 0; l < (ELAS_DESCRIPTOR_LENGTH); l++)
		{
			sum += abs(static_cast<short>(dRef[l]) - dOtherD[l]);
		}
		if (sum < bestCost)
		{
			bestCost = sum;
			bestDisparity = d;
		}
	}
	*output = bestDisparity;
}

template<typename Side>
__device__ inline void findMatch(
	short *output,
	const unsigned char * __restrict__ descriptor1,
	const unsigned char * __restrict__ descriptor2,
	const int3 * __restrict__ inputUV,
	const short * __restrict__ inputGrid,
	const float * __restrict__ inputTriangleFloat,
	int length,
	int width,
	int gridWidth,
	int gridDepth)
{
	const int indexL = blockIdx.x * blockDim.x + threadIdx.x;
	if ((indexL >= length))
	{
		return;
	}

	int3 uv = inputUV[indexL];
	const int ug = floor(uv.x / static_cast<float>(ELAS_GRID_SIZE));
	const int vg = floor(uv.y / static_cast<float>(ELAS_GRID_SIZE));
	const int index = uv.y * width + uv.x;
	const int indexD = (ELAS_DESCRIPTOR_LENGTH) * (uv.y * width + uv.x);
	const int indexT = 6 * uv.z;
	const int indexG = (vg * gridWidth + ug) * gridDepth;
	findMatch_<Side>(output + index,
				uv, descriptor1 + indexD, descriptor2 + indexD,
				inputGrid + indexG, inputTriangleFloat + indexT,
				width);
}

extern "C" __global__ void findMatchL(
	short *output,
	const unsigned char * __restrict__ descriptor1,
	const unsigned char * __restrict__ descriptor2,
	const int3 * __restrict__ inputUV,
	const short * __restrict__ inputGrid,
	const float * __restrict__ inputTriangleFloat,
	int length,
	int width,
	int gridWidth,
	int gridDepth)
{
	findMatch<Left>(output, descriptor1, descriptor2,
		inputUV, inputGrid, inputTriangleFloat,
		length, width, gridWidth, gridDepth);
}

extern "C" __global__ void findMatchR(
	short *output,
	const unsigned char * __restrict__ descriptor1,
	const unsigned char * __restrict__ descriptor2,
	const int3 * __restrict__ inputUV,
	const short * __restrict__ inputGrid,
	const float * __restrict__ inputTriangleFloat,
	int length,
	int width,
	int gridWidth,
	int gridDepth)
{
	findMatch<Right>(output, descriptor1, descriptor2,
		inputUV, inputGrid, inputTriangleFloat,
		length, width, gridWidth, gridDepth);
}

extern "C" __global__ void leftRightConsistencyCheck(
	short *output1,
	short *output2,
	const short * __restrict__ input1,
	const short * __restrict__ input2,
	int height,
	int width)
{
	const int indexL = blockIdx.x * blockDim.x + threadIdx.x;
	const int x = indexL % width;
	const int y = indexL / width;
	if ((x >= width) || (y >= height))
	{
		return;
	}

	const short d1 = input1[indexL];
	const short d2 = input2[indexL];
	const int warped1 = x - static_cast<int>(d1);
	const int warped2 = x + static_cast<int>(d2);
	if ((d1 < 0) || (warped1 < 0) || (warped1 >= width))
	{
		output1[indexL] = (ELAS_INVALIDATED_DISPARITY);
	}
	else
	{
		if(abs(input2[indexL - static_cast<int>(d1)] - d1) > (ELAS_LR_THRESHOLD))
		{
			output1[indexL] = (ELAS_INVALIDATED_DISPARITY);
		}
	}
	if ((d2 < 0) || (warped2 < 0) || (warped2 >= width))
	{
		output2[indexL] = (ELAS_INVALIDATED_DISPARITY);
	}
	else
	{
		if(abs(input1[indexL + static_cast<int>(d2)] - d2) > (ELAS_LR_THRESHOLD))
		{
			output2[indexL] = (ELAS_INVALIDATED_DISPARITY);
		}
	}
}

extern "C" __global__ void countLabels(
	int *output,
	const int * __restrict__ input,
	int height,
	int width)
{
	const int indexL = blockIdx.x * blockDim.x + threadIdx.x;
	const int x = indexL % width;
	const int y = indexL / width;
	if ((x >= width) || (y >= height))
	{
		return;
	}
	const int label = input[indexL];
	atomicAdd(output + label, 1);
}

extern "C" __global__ void removeSmallSegments(
	short *inputOutput,
	const int * __restrict__ inputCount,
	const int * __restrict__ inputLabel,
	int threshold,
	int height,
	int width)
{
	const int indexL = blockIdx.x * blockDim.x + threadIdx.x;
	const int x = indexL % width;
	const int y = indexL / width;
	if ((x >= width) || (y >= height))
	{
		return;
	}
	const int label = inputLabel[indexL];
	if (inputCount[label] < threshold)
	{
		inputOutput[indexL] = (ELAS_INVALIDATED_DISPARITY);
	}
}

template<typename Direction> __device__ inline void gapInterpolation_(
	float *output,
	int q0,
	int height,
	int width)
{
	const int length = Direction::length(height, width);
	const int stride = Direction::stride(1, width);
	int count = 0;

	for (int q1 = 0; q1 < length; q1++)
	{
		const int index = Direction::index(q0, q1, width);
		const float *pd = output + index;

		if ((*pd) < 0)
		{
			count++;
		}
		else
		{
			if (count >= 1 && count <= (ELAS_IPOL_GAP_WIDTH))
			{
				const int q10 = q1 - count;
				const int q11 = q1 - 1;
				if ((q10 > 0) && (q11 < length - 1))
				{
					float *pd0 = output + Direction::index(q0, q10 - 1, width);
					float *pd1 = output + Direction::index(q0, q11 + 1, width);
					const float dInterpolation = (fabsf(*pd0 - *pd1) < (ELAS_DISCON_THRESHOLD)) ? (((*pd0) + (*pd1))/2) : min(*pd0, *pd1);
					for (float *p = pd0 + stride; p <= pd1 - stride; p += stride)
					{
						*p = dInterpolation;
					}
				}
			}
			count = 0;
		}
	}

#ifdef ELAS_ADD_CORNERS
	float *pd0 = output + Direction::index(q0, 0, width), *pd = pd0;
	for (int q1 = 0; q1 < length; q1++)
	{
		if (!((*pd) < 0))
		{
			const float dInterpolation = *pd;
			for(; pd != pd0; pd -= stride)
			{
				*pd = dInterpolation;
			}
			*pd = dInterpolation;
			break;
		}
		pd += stride;
	}
	pd0 = output + Direction::index(q0, length - 1, width);
	pd = pd0;
	for (int q1 = 0; q1 < length; q1++)
	{
		if (!((*pd) < 0))
		{
			const float dInterpolation = *pd;
			for(; pd != pd0; pd += stride)
			{
				*pd = dInterpolation;
			}
			*pd = dInterpolation;
			break;
		}
		pd -= stride;
	}
#endif
}

extern "C" __global__ void gapInterpolationV(
	float *output,
	int height,
	int width)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	if (indexX >= width)
	{
		return;
	}
	gapInterpolation_<Vertical<float> >(output, indexX, height, width);
}

extern "C" __global__ void gapInterpolationH(
	float *output,
	int height,
	int width)
{
	const int indexY = blockIdx.x * blockDim.x + threadIdx.x;
	if (indexY >= height)
	{
		return;
	}
	gapInterpolation_<Horizontal<float> >(output, indexY, height, width);
}

template<typename Direction> __device__ inline void adaptiveMean_(
	float *output,
	const float * __restrict__ input,
	int q0,
	int q1,
	int height,
	int width)
{
	const int length = Direction::length(height, width);
	const int index = Direction::index(q0, q1, width);
	float val0 = input[index];
	float weight_sum = 0.f;
	float sum = 0.f;
	for (int k = -(ELAS_HALF_MEAN_WINDOW_SIZE); k < (ELAS_HALF_MEAN_WINDOW_SIZE); k++)
	{
		int q = min(max(q1 + k, 0), length - 1);
		float val = *(input + Direction::index(q0, q, width));
		float weight = 4.f - fabsf(val0 - val);
		weight = max(0.f, weight);
		weight_sum += weight;
		sum += (weight * val);
	}

	float d = (ELAS_INVALIDATED_DISPARITY);
	if (weight_sum > 0.f)
	{
		d = sum / weight_sum;
	}
	output[index] = (d < 0) ? val0 : d;
}

extern "C" __global__ void adaptiveMeanH(
	float *output,
	const float * __restrict__ input,
	int height,
	int width)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}
	adaptiveMean_<Horizontal<float> >(output, input, indexY, indexX, height, width);
}

extern "C" __global__ void adaptiveMeanV(
	float *output,
	const float * __restrict__ input,
	int height,
	int width)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}
	adaptiveMean_<Vertical<float> >(output, input, indexX, indexY, height, width);
}

template<typename ValueType, int N>
__device__ inline void sortValues_(ValueType *vals)
{
	ValueType tmp;
#pragma unroll
	for (char l = 0; l < N; l++)
	{
#pragma unroll
		for (char k = 0; k < l; k++)
		{
			if (vals[k] < vals[l])
			{
				tmp = vals[l];
				vals[l] = vals[k];
				vals[k] = tmp;
			}
		}
	}
}

template<typename Direction> __device__ inline void median_(
	float *output,
	const float * __restrict__ input,
	int q0,
	int q1,
	int height,
	int width)
{
	static const int ELAS_HALF_MEDIAN_WINDOW_SIZE = (ELAS_MEDIAN_WINDOW_SIZE) / 2;
	const int length = Direction::length(height, width);
	const int index = Direction::index(q0, q1, width);
	if (input[index] < 0)
	{
		output[index] = input[index];
		return;
	}

	float vals[ELAS_MEDIAN_WINDOW_SIZE];
	for (int k = -(ELAS_HALF_MEDIAN_WINDOW_SIZE), l = 0; k <= (ELAS_HALF_MEDIAN_WINDOW_SIZE); k++, l++)
	{
		int q = min(max(q1 + k, 0), length - 1);
		vals[l] = *(input + Direction::index(q0, q, width));
	}
	sortValues_<float, ELAS_MEDIAN_WINDOW_SIZE>(vals);
	output[index] = vals[ELAS_HALF_MEDIAN_WINDOW_SIZE];
}

extern "C" __global__ void medianH(
	float *output,
	const float * __restrict__ input,
	int height,
	int width)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}
	median_<Horizontal<float> >(output, input, indexY, indexX, height, width);
}

extern "C" __global__ void medianV(
	float *output,
	const float * __restrict__ input,
	int height,
	int width)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}
	median_<Vertical<float> >(output, input, indexX, indexY, height, width);
}
