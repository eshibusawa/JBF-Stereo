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

// It is desiable to use cuRAND, however this implementation uses random generator based on maximum legth sequence
// https://github.com/cupy/cupy/issues/1431
inline __device__ unsigned long int lsfr(unsigned long int &randomState)
{
	const unsigned long int mask = ((1UL << 6) | (1UL << 4) | (1UL << 2) | (1UL << 1) | (1UL << 0));
	if (randomState & (1UL << 31))
	{
		randomState = ((randomState ^ mask) << 1) | 1UL;
	}
	else
	{
		randomState <<= 1;
	}
	return randomState;
}

inline __device__ float unif(unsigned long int &randomState)
{
	return (lsfr(randomState) & 0xffffffffUL) / static_cast<float>(0xffffffffUL + 1.f);
}

inline __device__ float unifBetween(float minValue, float maxValue, unsigned long int &randomState)
{
	float s = maxValue - minValue;
	return unif(randomState) * s + minValue;
}
