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
// [1] Ando, S. (2000).
// Consistent gradient operators. IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(3), 252-265.

template<typename TextureType>
struct ShiftSampler
{
	__device__ static inline TextureType get(cudaTextureObject_t input, float x, float y)
	{
		return tex2D<TextureType>(input, x + 0.5f, y + 0.5f);
	}

	template<int Scalar>
	__device__ static inline TextureType get(cudaTextureObject_t input, float x, float y)
	{
		return Scalar * tex2D<TextureType>(input, x + 0.5f, y + 0.5f);
	}

	template<typename OutputType>
	__device__ static inline TextureType get(cudaTextureObject_t input, float x, float y)
	{
		return static_cast<OutputType>(tex2D<TextureType>(input, x + 0.5f, y + 0.5f));
	}


	template<typename OutputType, int Scalar>
	__device__ static inline TextureType get(cudaTextureObject_t input, float x, float y)
	{
		return static_cast<OutputType>(static_cast<OutputType>(Scalar) * tex2D<TextureType>(input, x + 0.5f, y + 0.5f));
	}
};

template<typename TextureType>
struct NormalSampler
{
	__device__ static inline TextureType get(cudaTextureObject_t input, float x, float y)
	{
		return tex2D<TextureType>(input, x, y);
	}

	template<int Scalar>
	__device__ static inline TextureType get(cudaTextureObject_t input, float x, float y)
	{
		return Scalar * tex2D<TextureType>(input, x, y);
	}

	template<typename OutputType>
	__device__ static inline TextureType get(cudaTextureObject_t input, float x, float y)
	{
		return static_cast<OutputType>(tex2D<TextureType>(input, x, y));
	}

	template<typename OutputType, int Scalar>
	__device__ static inline TextureType get(cudaTextureObject_t input, float x, float y)
	{
		return static_cast<OutputType>(static_cast<OutputType>(Scalar) * tex2D<TextureType>(input, x, y));
	}
};

__device__ inline float2 computeConsistentGradient_(cudaTextureObject_t input, float x, float y)
{
	float2 g;
	// the consistent gradient operator [1]
	const float c0 = 0.274526f, c1 = 0.112737f;
	g.x = fmaf(c0, GRADIENT_PIXEL_SAMPLER<GRADIENT_PIXEL_TYPE>::GRADIENT_SAMPLER_FUNCTION(input, x + 1, y), 0);
	g.x = fmaf(-c0, GRADIENT_PIXEL_SAMPLER<GRADIENT_PIXEL_TYPE>::GRADIENT_SAMPLER_FUNCTION(input, x - 1, y), g.x);
	g.x = fmaf(c1, GRADIENT_PIXEL_SAMPLER<GRADIENT_PIXEL_TYPE>::GRADIENT_SAMPLER_FUNCTION(input, x + 1, y - 1), g.x);
	g.x = fmaf(-c1, GRADIENT_PIXEL_SAMPLER<GRADIENT_PIXEL_TYPE>::GRADIENT_SAMPLER_FUNCTION(input, x - 1, y - 1), g.x);
	g.x = fmaf(c1, GRADIENT_PIXEL_SAMPLER<GRADIENT_PIXEL_TYPE>::GRADIENT_SAMPLER_FUNCTION(input, x + 1, y + 1), g.x);
	g.x = fmaf(-c1, GRADIENT_PIXEL_SAMPLER<GRADIENT_PIXEL_TYPE>::GRADIENT_SAMPLER_FUNCTION(input, x - 1, y + 1), g.x);

	g.y = fmaf(c0, GRADIENT_PIXEL_SAMPLER<GRADIENT_PIXEL_TYPE>::GRADIENT_SAMPLER_FUNCTION(input, x, y + 1), 0);
	g.y = fmaf(-c0, GRADIENT_PIXEL_SAMPLER<GRADIENT_PIXEL_TYPE>::GRADIENT_SAMPLER_FUNCTION(input, x, y - 1), g.y);
	g.y = fmaf(c1, GRADIENT_PIXEL_SAMPLER<GRADIENT_PIXEL_TYPE>::GRADIENT_SAMPLER_FUNCTION(input, x - 1, y + 1), g.y);
	g.y = fmaf(-c1, GRADIENT_PIXEL_SAMPLER<GRADIENT_PIXEL_TYPE>::GRADIENT_SAMPLER_FUNCTION(input, x - 1, y - 1), g.y);
	g.y = fmaf(c1, GRADIENT_PIXEL_SAMPLER<GRADIENT_PIXEL_TYPE>::GRADIENT_SAMPLER_FUNCTION(input, x + 1, y + 1), g.y);
	g.y = fmaf(-c1, GRADIENT_PIXEL_SAMPLER<GRADIENT_PIXEL_TYPE>::GRADIENT_SAMPLER_FUNCTION(input, x + 1, y - 1), g.y);
	return g;
}

__device__ inline float2 computeSobelGradient_(cudaTextureObject_t input, float x, float y)
{
	float2 g;
	g.x = fmaf(2, GRADIENT_PIXEL_SAMPLER<GRADIENT_PIXEL_TYPE>::GRADIENT_SAMPLER_FUNCTION(input, x + 1, y), 0);
	g.x = fmaf(-2, GRADIENT_PIXEL_SAMPLER<GRADIENT_PIXEL_TYPE>::GRADIENT_SAMPLER_FUNCTION(input, x - 1, y), g.x);
	g.x += GRADIENT_PIXEL_SAMPLER<GRADIENT_PIXEL_TYPE>::GRADIENT_SAMPLER_FUNCTION(input, x + 1, y - 1);
	g.x -= GRADIENT_PIXEL_SAMPLER<GRADIENT_PIXEL_TYPE>::GRADIENT_SAMPLER_FUNCTION(input, x - 1, y - 1);
	g.x += GRADIENT_PIXEL_SAMPLER<GRADIENT_PIXEL_TYPE>::GRADIENT_SAMPLER_FUNCTION(input, x + 1, y + 1);
	g.x -= GRADIENT_PIXEL_SAMPLER<GRADIENT_PIXEL_TYPE>::GRADIENT_SAMPLER_FUNCTION(input, x - 1, y + 1);

	g.y = fmaf(2, GRADIENT_PIXEL_SAMPLER<GRADIENT_PIXEL_TYPE>::GRADIENT_SAMPLER_FUNCTION(input, x, y + 1), 0);
	g.y = fmaf(-2, GRADIENT_PIXEL_SAMPLER<GRADIENT_PIXEL_TYPE>::GRADIENT_SAMPLER_FUNCTION(input, x, y - 1), g.y);
	g.y += GRADIENT_PIXEL_SAMPLER<GRADIENT_PIXEL_TYPE>::GRADIENT_SAMPLER_FUNCTION(input, x - 1, y + 1);
	g.y -= GRADIENT_PIXEL_SAMPLER<GRADIENT_PIXEL_TYPE>::GRADIENT_SAMPLER_FUNCTION(input, x - 1, y - 1);
	g.y += GRADIENT_PIXEL_SAMPLER<GRADIENT_PIXEL_TYPE>::GRADIENT_SAMPLER_FUNCTION(input, x + 1, y + 1);
	g.y -= GRADIENT_PIXEL_SAMPLER<GRADIENT_PIXEL_TYPE>::GRADIENT_SAMPLER_FUNCTION(input, x + 1, y - 1);
	return g;
}

extern "C" __global__ void computeConsistentGradient(
	float2 *gradient,
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
	gradient[index] = computeConsistentGradient_(tex, indexX, indexY);
}

extern "C" __global__ void computeSobelGradient(
	float2 *gradient,
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
	gradient[index] = computeSobelGradient_(tex, indexX, indexY);
}
