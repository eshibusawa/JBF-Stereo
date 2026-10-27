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

// Kernels of the Fast Bilateral Solver [1] (see bilateral_solver.py).
//
// [1] Barron, J. T., & Poole, B. (2016). The fast bilateral solver. In ECCV.
//
// The vertices of the bilateral grid are numbered in the order of their sorted hash, the neighbors are stored as a table
// neighbors[k * nVertices + v] (k = 2 * d + s: dimension d, s = 0 for -1 and s = 1 for +1; -1 if the neighbor does not exist),
// so that consecutive threads read consecutive addresses.

#define FBS_NUM_NEIGHBORS (2 * FBS_NUM_DIMS)
// The blur is the sum over the dimensions of the [1 2 1] kernel of each dimension: the weight of the center is 2 per dimension,
// the weight of each neighbor is 1.
#define FBS_CENTER_WEIGHT ((float)(2 * FBS_NUM_DIMS))

__device__ __forceinline__ int findVertex(const long long* sorted_hash, int n, long long key)
{
	// lower bound
	int lo = 0, hi = n;
	while (lo < hi)
	{
		int mid = (lo + hi) >> 1;
		if (sorted_hash[mid] < key) lo = mid + 1;
		else hi = mid;
	}
	return (lo < n && sorted_hash[lo] == key) ? lo : -1;
}

// ----------------------------------------------------------------------------------------------------------------------------
// Construction of the grid

// Pixel -> hash of the 5-D bilateral coordinate (x, y, Y, U, V) / sigma. The color is RGB (uint8, height x width x 3), converted
// to YUV (full range, offset 128 for U and V). The hash is the mixed radix number of the coordinates (the stride of each
// dimension is given), so the neighbor c +- 1 of a dimension is hash +- stride. The radix of a dimension is larger than its
// largest coordinate + 1, so a neighbor outside of the range (also the borrow from the next dimension at c = 0) is never the
// hash of an existing vertex.
extern "C" __global__ void computeHash(
	long long* hash,
	const unsigned char* rgb,
	const long long* stride,
	int width, int height)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	if (indexX >= width || indexY >= height)
		return;
	const int index = indexY * width + indexX;

	const unsigned char* p = rgb + 3 * (index);
	float r = p[0], g = p[1], b = p[2];
	float yy = 0.299f * r + 0.587f * g + 0.114f * b;
	float uu = -0.168736f * r - 0.331264f * g + 0.5f * b + 128.0f;
	float vv = 0.5f * r - 0.418688f * g - 0.081312f * b + 128.0f;

	long long c0 = (long long)(indexX / (FBS_SIGMA_SPATIAL));
	long long c1 = (long long)(indexY / (FBS_SIGMA_SPATIAL));
	long long c2 = (long long)(yy / (FBS_SIGMA_LUMA));
	long long c3 = (long long)(uu / (FBS_SIGMA_CHROMA));
	long long c4 = (long long)(vv / (FBS_SIGMA_CHROMA));

	hash[index] = c0 * stride[0] + c1 * stride[1] + c2 * stride[2] + c3 * stride[3] + c4 * stride[4];
}

// Table of the neighbors of each vertex by the binary search of the hash +- stride in the sorted unique hashes.
extern "C" __global__ void findNeighbors(
	int* neighbors,
	const long long* uniqueHash,
	const long long* stride,
	int nVertices)
{
	const int indexV = blockIdx.x * blockDim.x + threadIdx.x;
	if (indexV >= nVertices)
		return;

	long long h = uniqueHash[indexV];
	for (int d = 0; d < FBS_NUM_DIMS; d++)
	{
		neighbors[(2 * d + 0) * nVertices + indexV] = findVertex(uniqueHash, nVertices, h - stride[d]);
		neighbors[(2 * d + 1) * nVertices + indexV] = findVertex(uniqueHash, nVertices, h + stride[d]);
	}
}

// One step of the bistochastization: n <- sqrt(n * m / blur(n)), where blur = CENTER_WEIGHT * I + the neighbors.
extern "C" __global__ void bistochastizeStep(
	float* n_out,
	const float* n_in,
	const float* m,
	const int* neighbors,
	int nVertices)
{
	const int indexV = blockIdx.x * blockDim.x + threadIdx.x;
	if (indexV >= nVertices)
		return;

	float blurred = FBS_CENTER_WEIGHT * n_in[indexV];
	for (int k = 0; k < FBS_NUM_NEIGHBORS; k++)
	{
		int j = neighbors[k * nVertices + indexV];
		if (j >= 0) blurred += n_in[j];
	}
	n_out[indexV] = sqrtf(n_in[indexV] * m[indexV] / blurred);
}

// ----------------------------------------------------------------------------------------------------------------------------
// Splat / slice

// output[idx[index]] += input[index] (out has to be zero). The order of the additions is not deterministic.
extern "C" __global__ void splat(
	float* output,
	const float* input,
	const int* splatIndices,
	int nPixels)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= nPixels)
		return;
	atomicAdd(&output[splatIndices[index]], input[index]);
}

// output[index] = input[idx[index]]
extern "C" __global__ void slice(
	float* output,
	const float* input,
	const int* splatIndices,
	int nPixels)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= nPixels)
		return;
	output[index] = input[splatIndices[index]];
}

// ----------------------------------------------------------------------------------------------------------------------------
// The linear system A y = b,  A = lambda * (Dm - Dn B Dn) + diag(wSplat)
//
// B has the center weight CENTER_WEIGHT, and m = n * (B n) after the bistochastization, so
//   (Dm - Dn B Dn) y = n * (B n) * y - n * B (n * y) = n * sum_{j in neighbors} n_j * (y - y_j),
// the Laplacian of the weighted graph. The kernels use this form: it needs only n, it does not cancel two large numbers (in
// float) and it is positive semi-definite also in the rounding.

extern "C" __global__ void applyA(
	float* out,
	const float* y,
	const float* n,
	const float* wSplat,
	const int* neighbors,
	int nVertices)
{
	const int indexV = blockIdx.x * blockDim.x + threadIdx.x;
	if (indexV >= nVertices) return;

	float yv = y[indexV];
	float acc = 0.0f;
	for (int k = 0; k < FBS_NUM_NEIGHBORS; k++)
	{
		int j = neighbors[k * nVertices + indexV];
		if (j >= 0) acc += n[j] * (yv - y[j]);
	}
	out[indexV] = (FBS_LAMBDA) * n[indexV] * acc + wSplat[indexV] * yv;
}

// Inverse of the diagonal of A (Jacobi preconditioner): diag = lambda * n * sum(n_j) + wSplat, clamped by ADiatMin.
extern "C" __global__ void inverseDiagonal(
	float* invDiag,
	const float* n,
	const float* wSplat,
	const int* neighbors,
	int nVertices)
{
	const int indexV = blockIdx.x * blockDim.x + threadIdx.x;
	if (indexV >= nVertices) return;

	float acc = 0.0f;
	for (int k = 0; k < FBS_NUM_NEIGHBORS; k++)
	{
		int j = neighbors[k * nVertices + indexV];
		if (j >= 0) acc += n[j];
	}
	float diag = (FBS_LAMBDA) * n[indexV] * acc + wSplat[indexV];
	invDiag[indexV] = 1.0f / fmaxf(diag, (FBS_A_DIAG_MIN));
}

// ----------------------------------------------------------------------------------------------------------------------------
// Preconditioned conjugate gradient. The scalars stay on the device: s[0] = r.z, s[1] = p.Ap, s[2] = r.z of the next step.

// y += alpha p, r -= alpha Ap, z = M r  (alpha = s[0] / s[1])
extern "C" __global__ void cgUpdate(
	float* y,
	float* r,
	float* z,
	const float* p,
	const float* Ap,
	const float* invDiag,
	const float* s,
	int nVertices)
{
	const int indexV = blockIdx.x * blockDim.x + threadIdx.x;
	if (indexV >= nVertices)
		return;

	float alpha = (s[1] != 0.0f) ? s[0] / s[1] : 0.0f;
	y[indexV] += alpha * p[indexV];
	float rv = r[indexV] - alpha * Ap[indexV];
	r[indexV] = rv;
	z[indexV] = invDiag[indexV] * rv;
}

// p = z + beta p  (beta = s[2] / s[0])
extern "C" __global__ void cgDirection(
	float* p,
	const float* z,
	const float* s,
	int nVertices)
{
	const int indexV = blockIdx.x * blockDim.x + threadIdx.x;
	if (indexV >= nVertices)
		return;

	float beta = (s[0] != 0.0f) ? s[2] / s[0] : 0.0f;
	p[indexV] = z[indexV] + beta * p[indexV];
}
