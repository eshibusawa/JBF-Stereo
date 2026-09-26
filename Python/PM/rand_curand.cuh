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

// Random number generator of PatchMatch based on cuRAND (Philox). It replaces the LFSR of rand_mls.cuh, whose
// consecutive outputs share 31 bits (see README.md).
// The state is not kept in memory: every kernel launch initializes it from (seed, pixel index, offset), where the
// offset is advanced by the host at each launch that consumes random numbers.
#include <curand_kernel.h>

typedef curandStatePhilox4_32_10_t RandomState;

inline __device__ void initRandom(RandomState &randomState, unsigned long long seed, int index, unsigned long long offset)
{
	curand_init(seed, index, offset, &randomState);
}

// uniform in (0, 1]
inline __device__ float unif(RandomState &randomState)
{
	return curand_uniform(&randomState);
}

inline __device__ float unifBetween(float minValue, float maxValue, RandomState &randomState)
{
	return unif(randomState) * (maxValue - minValue) + minValue;
}
