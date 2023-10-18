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

__constant__ float g_LASWRange[TGV_RANGE_MAX];
__constant__ float g_LASWSpace[TGV_AGGREGATION_WINDOW_SIZE][TGV_AGGREGATION_WINDOW_SIZE];

namespace
{
inline __device__ float2 operator+(float2 x, float2 y)
{
	return make_float2(x.x + y.x, x.y + y.y);
}

inline __device__ float2 operator-(float2 x, float2 y)
{
	return make_float2(x.x - y.x, x.y - y.y);
}

inline __device__ float3 operator+(float3 x, float3 y)
{
	return make_float3(x.x + y.x, x.y + y.y, x.z + y.z);
}

inline __device__ float2 operator*(float a, float2 x)
{
	return make_float2(a * x.x, a * x.y);
}

inline __device__ float3 operator*(float a, float3 x)
{
	return make_float3(a * x.x, a * x.y, a * x.z);
}
}

extern "C" __global__ void computeCensusTransform(
	long int *census,
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
    long int val0 = 0, val = 0;
    for (int j = -(TGV_CENSUS_RADIUS); j <= (TGV_CENSUS_RADIUS); j++)
    {
        for (int i = -(TGV_CENSUS_RADIUS); i <= (TGV_CENSUS_RADIUS); i++)
        {
            val = (val << 1);
            val0 = (tex2D<unsigned char>(tex, indexX + i, indexY + j) < tex2D<unsigned char>(tex, indexX, indexY) ? 1 : 0);
            val |= val0;
        }
    }
	census[index] = val;
}

extern "C" __global__ void computeHammingDistance(
	unsigned char *hammingDistance,
	const long int * __restrict__ censusRef,
	const long int * __restrict__ censusOther,
	int width,
	int length)
{
	const int indexXYD = blockIdx.x * blockDim.x + threadIdx.x;
	if (indexXYD >= length)
	{
		return;
	}
	const int indexD = indexXYD % (TGV_MAX_DISPARITY);
	const int indexXY = indexXYD / (TGV_MAX_DISPARITY);
	const int indexX = indexXY % width;

    unsigned char val = (TGV_INVALID_HD_COST); // invalid and large, val = [0, 64]
	if (indexX - indexD >= 0)
	{
		const long int censusXOR = ((censusRef[indexXY]) ^ (censusOther[indexXY - indexD]));
		val = __popcll(static_cast<unsigned long long int>(censusXOR));
	}

	hammingDistance[indexXYD] = val;
}

extern "C" __global__ void aggregateCostLASW(
	float *costAggregated,
	cudaTextureObject_t texRef,
	cudaTextureObject_t texOther,
	const unsigned char * __restrict__ costHD,
	int height,
	int width,
	int length)
{
	const int indexXYD = blockIdx.x * blockDim.x + threadIdx.x;
	if (indexXYD >= length)
	{
		return;
	}
	const int indexD = indexXYD % (TGV_MAX_DISPARITY);
	const int indexXY = indexXYD / (TGV_MAX_DISPARITY);
	const int indexX = indexXY % width;
	const int indexY = indexXY / width;

	if (indexX - indexD < 0)
	{
		costAggregated[indexXYD] = costHD[indexXYD];
		return;
	}

	float numeSum = 0.f, denomSum = 0.f, costVal = 0.f;
	float wRef = 0.f, wOther = 0.f, wRS = 0.f;
	int indexX2 = 0, indexY2 = 0, indexXYD2 = 0, deltaValRef = 0, deltaValOther = 0;
	for (int j = -(TGV_AGGREGATION_RADIUS); j <= (TGV_AGGREGATION_RADIUS); j++)
	{
		for (int i = -(TGV_AGGREGATION_RADIUS); i <= (TGV_AGGREGATION_RADIUS); i++)
		{
			if ((i == 0) && (j == 0))
			{
				continue;
			}
			indexX2 = indexX + i;
			indexY2 = indexY + j;
			if ((indexX2  < 0) || (indexX2 >= width) || (indexY2  < 0) || (indexY2 >= height))
			{
				costVal = 1;
			}
			else
			{
				indexXYD2 = (indexX2 + indexY2 * width) * (TGV_MAX_DISPARITY) + indexD;
				costVal = costHD[indexXYD2];
			}

			if (costVal == (TGV_INVALID_HD_COST))
			{
				continue;
			}

			// constant * spatial kernel
			wRef = wOther = (TGV_AGGREGATION_K) * g_LASWSpace[j + (TGV_AGGREGATION_RADIUS)][i + (TGV_AGGREGATION_RADIUS)];
			// multiply range kernel
			deltaValRef = abs(static_cast<int>(tex2D<unsigned char>(texRef, indexX, indexY)) - tex2D<unsigned char>(texRef, indexX2, indexY2));
			deltaValOther = abs(static_cast<int>(tex2D<unsigned char>(texOther, indexX - indexD, indexY)) - tex2D<unsigned char>(texOther, indexX2 - indexD, indexY2));
			wRef *= g_LASWRange[deltaValRef];
			wOther *= g_LASWRange[deltaValOther];
			// multiply ref * other
			wRS = wRef * wOther;
			numeSum += (wRS * costVal);
			denomSum += (wRS);
		}
	}

	costAggregated[indexXYD] = numeSum / denomSum;
}

extern "C" __global__ void computeAnisotropicDiffusionTensor(
	float3 *tensor,
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
	float2 g = computeConsistentGradient_(tex, indexX, indexY);
	float gNorm = hypotf(g.x, g.y);
	float2 n = make_float2(1, 0);
	if (gNorm >= static_cast<float>(TGV_GRAD_EPS))
	{
		n = make_float2(g.x / gNorm, g.y / gNorm);
	}
	float e = expf(-static_cast<float>(TGV_TENSOR_A) * powf(gNorm, static_cast<float>(TGV_TENSOR_B)));
	e = fmaxf(e, static_cast<float>(TGV_TENSOR_EPS));
	// n = (n.y, n.x)
	// nT = (n.y, -n.x)
	tensor[index].x = e * n.x * n.x + n.y * n.y;
	tensor[index].y = e * n.x * n.y - n.y * n.x;
	tensor[index].z = e * n.y * n.y + n.x * n.x;
}

extern "C" __global__ void updateDualVariables(
	float2 *p,
	float3 *q,
	const float * __restrict__ u_,
	const float2 * __restrict__ v_,
	const float3 * __restrict__ G,
	int height,
	int width,
	int length)
{
	const int indexXY = blockIdx.x * blockDim.x + threadIdx.x;
	if (indexXY >= length)
	{
		return;
	}
	const int indexX = indexXY % width;
	const int indexY = indexXY / width;

	float u_plus_x(u_[indexXY]), u_plus_y(u_[indexXY]);
	float2 v_plus_x(v_[indexXY]), v_plus_y(v_[indexXY]);
	if (indexX + 1 < width)
	{
		u_plus_x = u_[indexXY + 1];
		v_plus_x = v_[indexXY + 1];
	}
	if (indexY + 1 < height)
	{
		u_plus_y = u_[indexXY + width];
		v_plus_y = v_[indexXY + width];
	}
	// gradient
	float2 u_grad = make_float2(u_plus_x - u_[indexXY], u_plus_y - u_[indexXY]); // [dx dy]
	float3 v_grad = make_float3(v_plus_x.x - v_[indexXY].x,
								0.5f*(v_plus_y.x - v_[indexXY].x + v_plus_x.y - v_[indexXY].y),
								v_plus_y.y - v_[indexXY].y); // [dxx (dxy+dyx)/2, dyy]
	// candidate
	float2 u_grad_v_ = u_grad - v_[indexXY];
	float2 Gu_grad_v_ = make_float2(fmaf(G[indexXY].x, u_grad_v_.x, G[indexXY].y * u_grad_v_.y),
									fmaf(G[indexXY].y, u_grad_v_.x, G[indexXY].z * u_grad_v_.y));
	float2 p_candidate = p[indexXY] + static_cast<float>(TGV_TAU_P) * Gu_grad_v_;
	float3 q_candidate = q[indexXY] + static_cast<float>(TGV_TAU_Q) * v_grad;
	// proximal mapping
	float p_norm = fmaxf(fabsf(p_candidate.x), fabsf(p_candidate.y));
	float q_norm = fmaxf(fmaxf(fabsf(q_candidate.x), fabsf(q_candidate.y)), fabsf(q_candidate.z));
	p[indexXY] = (1.f / fmaxf(1.f, p_norm / static_cast<float>(TGV_LAMBDA_S))) * p_candidate;
	q[indexXY] = (1.f / fmaxf(1.f, q_norm / static_cast<float>(TGV_LAMBDA_A))) * q_candidate;
}

extern "C" __global__ void updatePrimalVariables(
	float *u,
	float *u_,
	float2 *v,
	float2 *v_,
	float theta_n,
	const float2 * __restrict__ p,
	const float3 * __restrict__ q,
	const float * __restrict__ a,
	const float * __restrict__ L,
	const float3 * __restrict__ G,
	int height,
	int width,
	int length)
{
	const int indexXY = blockIdx.x * blockDim.x + threadIdx.x;
	if (indexXY >= length)
	{
		return;
	}
	const int indexX = indexXY % width;
	const int indexY = indexXY / width;

	float2 p_minus_x(make_float2(0, 0)), p_minus_y(make_float2(0, 0));
	float3 q_minus_x(make_float3(0, 0, 0)), q_minus_y(make_float3(0, 0, 0));
	if (indexX - 1 >= 0)
	{
		p_minus_x = p[indexXY - 1];
		q_minus_x = q[indexXY - 1];
	}
	if (indexY - 1 >= 0)
	{
		p_minus_y = p[indexXY - width];
		q_minus_y = q[indexXY - width];
	}

	const float inv_theta_n = 1.f / theta_n;
	// divergence
	float2 Gp = make_float2(fmaf(G[indexXY].x, p[indexXY].x, G[indexXY].y * p[indexXY].y),
							fmaf(G[indexXY].y, p[indexXY].x, G[indexXY].z * p[indexXY].y));
	float2 Gp_minus_x = make_float2(fmaf(G[indexXY].x, p_minus_x.x, G[indexXY].y * p_minus_x.y),
									fmaf(G[indexXY].y, p_minus_x.x, G[indexXY].z * p_minus_x.y));
	float2 Gp_minus_y = make_float2(fmaf(G[indexXY].x, p_minus_y.x, G[indexXY].y * p_minus_y.y),
									fmaf(G[indexXY].y, p_minus_y.x, G[indexXY].z * p_minus_y.y));
	const float p_div = Gp.x - Gp_minus_x.x + Gp.y - Gp_minus_y.y;
	// candidate
	const float nume_u = (u[indexXY] + static_cast<float>(TGV_TAU_U) * (p_div - L[indexXY] + a[indexXY] * inv_theta_n));
	const float denom_u = (1 + static_cast<float>(TGV_TAU_U) * inv_theta_n);
	float u_candidate = __saturatef(nume_u / denom_u);
	u_[indexXY] = 2 * u_candidate - u[indexXY];
	u[indexXY] = u_candidate;

	// divergence
	const float2 q_div = make_float2(q[indexXY].x - q_minus_x.x + q[indexXY].y - q_minus_y.y,
	q[indexXY].y - q_minus_x.y + q[indexXY].z - q_minus_y.z);
	// candidate
	float2 v_candidate = v[indexXY] + static_cast<float>(TGV_TAU_V) * (p[indexXY] + q_div);
	v_[indexXY] = 2 * v_candidate - v[indexXY];
	v[indexXY] = v_candidate;
}
