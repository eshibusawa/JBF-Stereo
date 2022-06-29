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

__constant__ float g_BFRange[BILATERAL_FILTER_COEFFICIENTS_SIZE_RANGE];
__constant__ float g_BFSpace[BILATERAL_FILTER_COEFFICIENTS_SIZE_SPACE];

extern "C" __global__ void applyJointBilateralFilter(
	short* output,
	cudaTextureObject_t texImage,
	cudaTextureObject_t texGuide,
	int radiusH,
	int radiusV,
	int width,
	int height)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;

	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}

	float sumVal = 0.f;
	float sumWeight = 0.f;
	unsigned char guideVal0 = tex2D<unsigned char>(texGuide, indexX, indexY);
    short imageVal0 = tex2D<short>(texImage, indexX, indexY);
	// Weighted Joint Bilateral Fitler
	#pragma unroll 8
	for (int j = -radiusV, k = 0; j <= radiusV; j++)
	{
		#pragma unroll 8
		for (int i = -radiusH; i <= radiusH; i++)
		{
			short imageVal = tex2D<short>(texImage, indexX + i, indexY + j);
			unsigned char guideVal = tex2D<unsigned char>(texGuide, indexX + i, indexY + j);
			float weightVal = (imageVal < 0) ? 0.f : 1.f;

			float w = weightVal * g_BFSpace[k] * g_BFRange[abs(static_cast<int>(guideVal0) - guideVal)];
			sumWeight += w;
			sumVal = fmaf(static_cast<float>(imageVal), w, sumVal);
			k++;
		}
	}

	const int index = indexX + indexY * width;
	short outVal = imageVal0;
	unsigned short minDiff = 1 << 15;
	// Joint Nearest Filter
	if (sumWeight > 0.f)
	{
		const short outVal0 = static_cast<short>(sumVal/sumWeight);
		#pragma unroll 8
		for (int j = -radiusV; j <= radiusV; j++)
		{
			#pragma unroll 8
			for (int i = -radiusH; i <= radiusH; i++)
			{
    			const short imageVal = tex2D<short>(texImage, indexX + i, indexY + j);
                if (imageVal < 0)
                {
                    continue;
                }
    			const unsigned short diff = abs(outVal0 - imageVal);
				outVal = (diff < minDiff) ? imageVal : outVal;
				minDiff = (diff < minDiff) ? diff : minDiff;
			}
		}
	}
	output[index] = outVal;
}
