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

// References
// [1] Hu, Y., Song, R., & Li, Y. (2016). Efficient coarse-to-fine patchmatch for large displacement optical flow.
//     In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 5704-5712).

// PatchMatch of CtFPM for rectified stereo images: the search is along the horizontal line, a seed has one disparity.
// The point of the other image that corresponds to (x, y) is (x + direction * disparity, y): direction is -1 when the left image
// is the reference (the disparity is positive) and +1 when the right image is the reference.
// The seeds are updated color by color. The seeds of one color are never neighbors of each other, so a seed
// reads the disparity of its neighbors (the other colors) and writes only its own disparity. That is race free.
// One thread updates one seed: propagation from the neighbors, then random search.
// The random numbers are Philox of cuRAND. The generator is initialized in every launch with
// (seed, subsequence = seed index, offset = position in the stream managed by the host).

#include <curand_kernel.h>

// sum of absolute differences of the descriptors (CPM::MatchCost)
inline __device__ int matchCost(
    const unsigned char * __restrict__ feature1,
    const unsigned char * __restrict__ feature2,
    int width,
    int height,
    int channels,
    int x1,
    int y1,
    int x2,
    int y2)
{
    x1 = min(max(x1, 0), width - 1);
    x2 = min(max(x2, 0), width - 1);
    y1 = min(max(y1, 0), height - 1);
    y2 = min(max(y2, 0), height - 1);

    const unsigned char *p1 = feature1 + ((size_t)y1 * width + x1) * channels;
    const unsigned char *p2 = feature2 + ((size_t)y2 * width + x2) * channels;

    int sum = 0;
    if ((channels % 16) == 0)
    {
        const uint4 *q1 = reinterpret_cast<const uint4 *>(p1);
        const uint4 *q2 = reinterpret_cast<const uint4 *>(p2);
        for (int k = 0; k < channels / 16; k++)
        {
            const uint4 a = q1[k];
            const uint4 b = q2[k];
            sum += __vsadu4(a.x, b.x) + __vsadu4(a.y, b.y) + __vsadu4(a.z, b.z) + __vsadu4(a.w, b.w);
        }
    }
    else
    {
        for (int k = 0; k < channels; k++)
        {
            sum += abs((int)p1[k] - (int)p2[k]);
        }
    }
    return sum;
}

extern "C" __global__ void initRandomDisparity(
    float *disparity,
    int numSeeds,
    int initRadius,
    unsigned long long rngSeed,
    unsigned long long rngOffset)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numSeeds)
    {
        return;
    }
    curandStatePhilox4_32_10_t rs;
    curand_init(rngSeed, i, rngOffset, &rs);
    disparity[i] = (int)(curand(&rs) % (2 * initRadius + 1)) - initRadius;
}

extern "C" __global__ void initCosts(
    float *bestCosts,
    const float * __restrict__ disparity,
    const int * __restrict__ seeds,
    const unsigned char * __restrict__ feature1,
    const unsigned char * __restrict__ feature2,
    int width,
    int height,
    int channels,
    int numSeeds,
    int direction)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numSeeds)
    {
        return;
    }
    const int x = seeds[2 * i];
    const int y = seeds[2 * i + 1];
    bestCosts[i] = (float)matchCost(feature1, feature2, width, height, channels, x, y, (int)(x + direction * disparity[i]), y);
}

// Update the seeds listed in indices (one color).
extern "C" __global__ void updateSeeds(
    float *disparity,
    float *bestCosts,
    int *updateCount,
    const int * __restrict__ indices,
    const unsigned char * __restrict__ feature1,
    const unsigned char * __restrict__ feature2,
    const int * __restrict__ seeds,
    const int * __restrict__ neighbors,
    const float * __restrict__ radius,
    int width,
    int height,
    int channels,
    int numIndices,
    int maxNeighbors,
    int direction,
    unsigned long long rngSeed,
    unsigned long long rngOffset)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numIndices)
    {
        return;
    }

    const int idx = indices[t];
    const int x = seeds[2 * idx];
    const int y = seeds[2 * idx + 1];
    bool updateFlag = false;

    // Propagation: try the disparities of all neighbors
    const int *nbIdx = neighbors + idx * maxNeighbors;
    for (int i = 0; i < maxNeighbors; i++)
    {
        if (nbIdx[i] < 0)
        {
            break;
        }
        const float td = disparity[nbIdx[i]];
        if (fabsf(td - disparity[idx]) < 1e-6)
        {
            continue;
        }
        const float tc = (float)matchCost(feature1, feature2, width, height, channels, x, y, (int)(x + direction * td), y);
        if (tc < bestCosts[idx])
        {
            bestCosts[idx] = tc;
            disparity[idx] = td;
            updateFlag = true;
        }
    }

    // Random search: windows of exponentially decreasing size around the current best guess
    curandStatePhilox4_32_10_t rs;
    curand_init(rngSeed, idx, rngOffset, &rs);
    for (int mag = (int)(radius[idx] + 0.5); mag >= 1; mag /= 2)
    {
        const float td = disparity[idx] + (int)(curand(&rs) % (2 * mag + 1)) - mag;
        if (fabsf(td - disparity[idx]) < 1e-6)
        {
            continue;
        }
        const float tc = (float)matchCost(feature1, feature2, width, height, channels, x, y, (int)(x + direction * td), y);
        if (tc < bestCosts[idx])
        {
            bestCosts[idx] = tc;
            disparity[idx] = td;
            updateFlag = true;
        }
    }

    if (updateFlag)
    {
        atomicAdd(updateCount, 1);
    }
}

// The search radius of the next level: half of the range of the disparities of a seed and its neighbors
// (CPM::UpdateSearchRadius is the radius of the minimal circle around the flows; they lie on a line here),
// clamped to [1, maxRadius] and scaled.
extern "C" __global__ void updateSearchRadius(
    float *outRadius,
    const float * __restrict__ disparity,
    const int * __restrict__ neighbors,
    int numSeeds,
    int maxNeighbors,
    int maxRadius,
    double scale)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numSeeds)
    {
        return;
    }

    float lowest = disparity[i];
    float highest = lowest;
    const int *nbIdx = neighbors + i * maxNeighbors;
    for (int n = 0; n < maxNeighbors; n++)
    {
        if (nbIdx[n] < 0)
        {
            break;
        }
        lowest = fminf(lowest, disparity[nbIdx[n]]);
        highest = fmaxf(highest, disparity[nbIdx[n]]);
    }

    float r = 0.5f * (highest - lowest);
    r = (r < maxRadius) ? r : (float)maxRadius;
    r = (r > 1) ? r : 1.f;
    outRadius[i] = (float)((double)r * scale);
}
