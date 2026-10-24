# This file is part of JBF-Stereo.
# Copyright (c) 2026, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
import os

import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as ndi

import add_path
from sift import SIFT

# References
# [1] Hu, Y., Song, R., & Li, Y. (2016). Efficient coarse-to-fine patchmatch for large displacement optical flow.
#     In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 5704-5712).

def gaussian(img, sigma):
    """Gaussian smoothing of every channel of a (height, width, channels) image, replicated border (the semantics of skimage.filters.gaussian)."""
    return ndi.gaussian_filter(img, sigma=(sigma, sigma, 0), mode='nearest', truncate=4.0)

def rescale(img, scale):
    """Bilinear rescaling of a (height, width, channels) image by scale (the semantics of skimage.transform.rescale with order 1):
    the output size is round(scale * size), the image is smoothed with a Gaussian of sigma (factor - 1) / 2 before it is shrunk
    (anti-aliasing), the border is mirrored (the skimage mode 'reflect' is the ndimage mode 'mirror')."""
    in_shape = np.array(img.shape[:2])
    out_shape = np.maximum(np.round(scale * in_shape), 1).astype(int)
    factors = in_shape / out_shape
    if np.any(out_shape < in_shape):
        sigma = np.maximum(0, (factors - 1) / 2)
        img = ndi.gaussian_filter(img, sigma=(sigma[0], sigma[1], 0), mode='mirror')
    return ndi.zoom(img, (1 / factors[0], 1 / factors[1], 1), order=1, mode='mirror', grid_mode=True)

class CtFPM:
    """Coarse-to-fine PatchMatch [1] for rectified stereo images (the search is along the horizontal line).

    Dense SIFT descriptors of an image pyramid are matched at the seeds of a regular grid; a seed has one disparity.
    The seeds are updated in 4 colors (parity of the grid x and y) in parallel: the 8-neighborhood of the seed grid is kept
    and the seeds of one color do not interfere. The matching is run both ways and checked for consistency.
    The disparity is x_left - x_right (positive for the left image as the reference)."""
    RNG_STRIDE = 32 # upper bound of the random numbers consumed by a seed in one launch

    def __init__(self, params):
        self.gpu_module = None
        self.params = params
        self.img_ref = self.img_other = None
        self.sift = SIFT(params)
        self.phases = None
        self.rng_seed = 0
        self.rng_offset = 0

    def compile_module(self):
        fpfn = os.path.join(os.path.dirname(__file__), 'ctfpm.cu')
        with open(fpfn, 'r') as f:
            cuda_source = f.read()
        self.gpu_module = cp.RawModule(code=cuda_source)
        self.gpu_module.compile()

    def setup_module(self):
        if self.gpu_module is None:
            self.compile_module()
        self.sift.setup_module()

    @staticmethod
    def construct_pyramid(img, ratio, min_width):
        n_levels = int(np.log(np.float32(min_width) / img.shape[1]) / np.log(np.float32(ratio)))
        n_level_constant_sigma = int(np.log(0.25, dtype=np.float32) / np.log(ratio, dtype=np.float32))
        base_sigma = (1 / np.float32(ratio) - 1)

        pyramid = list()
        pyramid.append(cp.copy(img))
        for k in range(1, n_levels):
            if (k <= n_level_constant_sigma):
                img_smoothed = gaussian(img, k*base_sigma)
                img_resized = rescale(img_smoothed, np.power(np.float32(ratio), k))
            else:
                img_smoothed = gaussian(pyramid[k - n_level_constant_sigma], n_level_constant_sigma*base_sigma)
                r = np.power(np.float32(ratio), k) * np.float32(img.shape[1]) / np.float32(img_smoothed.shape[1])
                img_resized = rescale(img_smoothed, r)
            pyramid.append(img_resized)
        return pyramid

    def compute_features(self, pyramid):
        features = list()
        for img in pyramid:
            dx = self.sift.sobel(img, axis=1)
            dy = self.sift.sobel(img, axis=0)
            grad = self.sift.compute_gradient(dx, dy)
            self.sift.compute_band(grad)
            self.sift.filter_band()
            features.append(self.sift.compute_SIFT())
        return features

    def generate_seeds(self, height, width):
        """Seeds on a regular grid, their 8-neighbors (-1 terminated), the label map for the cross check and the 4 colors."""
        step = self.params.step
        grid_w = width // step
        grid_h = height // step
        x_offset = (width - (grid_w - 1) * step) // 2
        y_offset = (height - (grid_h - 1) * step) // 2
        num_seeds = grid_w * grid_h

        grid_y, grid_x = np.divmod(np.arange(num_seeds), grid_w)
        seeds = np.stack([grid_x * step + x_offset, grid_y * step + y_offset], axis=1).astype(np.int32)

        neighbors = np.full((num_seeds, 12), -1, dtype=np.int32)
        count = np.zeros(num_seeds, dtype=np.int64)
        for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nx = grid_x + dx
            ny = grid_y + dy
            idx = np.nonzero((nx >= 0) & (nx < grid_w) & (ny >= 0) & (ny < grid_h))[0]
            neighbors[idx, count[idx]] = ny[idx] * grid_w + nx[idx]
            count[idx] += 1

        # the later seed overwrites the earlier one, as in the sequential loop of CPM.cpp
        r = step // 2
        offsets = np.arange(-r, r + 1)
        xx = np.clip(seeds[:, 0, None, None] + offsets[None, :, None] + np.zeros_like(offsets)[None, None, :], 0, width - 1)
        yy = np.clip(seeds[:, 1, None, None] + np.zeros_like(offsets)[None, :, None] + offsets[None, None, :], 0, height - 1)
        labels = np.broadcast_to(np.arange(num_seeds, dtype=np.int32)[:, None, None], xx.shape)
        klabels = np.zeros((height, width), dtype=np.int32)
        klabels[yy.ravel(), xx.ravel()] = labels.ravel()

        color = (np.arange(num_seeds) % grid_w) % 2 + 2 * ((np.arange(num_seeds) // grid_w) % 2)
        self.phases = [cp.asarray(np.arange(num_seeds)[color == c].astype(np.int32)) for c in range(4)]
        return cp.asarray(seeds), cp.asarray(neighbors), cp.asarray(klabels)

    def generate_pyramid_seeds(self, seeds, features):
        ratio = float(np.float32(self.params.ratio))
        seeds_cpu = seeds.get()
        pyd_seeds = list()
        for level, feature in enumerate(features):
            scaled = np.floor(seeds_cpu * ratio ** level)
            scaled[:, 0] = np.clip(scaled[:, 0], 0, feature.shape[1] - 1)
            scaled[:, 1] = np.clip(scaled[:, 1], 0, feature.shape[0] - 1)
            pyd_seeds.append(cp.asarray(scaled.astype(np.int32)))
        return pyd_seeds

    def srand(self, seed):
        self.rng_seed = seed
        self.rng_offset = 0

    def next_rng_offset(self):
        offset = self.rng_offset
        self.rng_offset += self.RNG_STRIDE
        return offset

    def init_random_disparity(self, num_seeds, init_radius):
        disparity = cp.zeros(num_seeds, dtype=cp.float32)
        gpu_func = self.gpu_module.get_function('initRandomDisparity')
        sz_block = 128
        gpu_func(
            block=(sz_block,), grid=(-(-num_seeds // sz_block),),
            args=(disparity, np.int32(num_seeds), np.int32(init_radius), np.uint64(self.rng_seed), np.uint64(self.next_rng_offset()))
        )
        cp.cuda.runtime.deviceSynchronize()
        return disparity

    def propagate(self, feature1, feature2, seeds, neighbors, radius, disparity, direction):
        """Improve disparity (in place) of the seeds of one level. Returns the best costs and the number of iterations.
        direction: -1 when the image of feature1 is the left image, +1 when it is the right image."""
        for a in (feature1, feature2, seeds, neighbors, radius, disparity):
            assert a.flags.c_contiguous
        assert feature1.dtype == cp.uint8 and feature2.dtype == cp.uint8
        assert feature1.shape == feature2.shape
        assert direction in (-1, 1)
        assert self.phases is not None and sum(len(p) for p in self.phases) == seeds.shape[0]

        height, width, channels = feature1.shape
        num_seeds = seeds.shape[0]
        sz_block = 128

        best_costs = cp.empty(num_seeds, dtype=cp.float32)
        gpu_func = self.gpu_module.get_function('initCosts')
        gpu_func(
            block=(sz_block,), grid=(-(-num_seeds // sz_block),),
            args=(best_costs, disparity, seeds, feature1, feature2, np.int32(width), np.int32(height), np.int32(channels), np.int32(num_seeds), np.int32(direction))
        )

        update_func = self.gpu_module.get_function('updateSeeds')
        stop_ratio = np.float32(self.params.stop_iter_ratio)
        last_update_ratio = np.float32(2)
        iteration = 0
        while iteration < self.params.max_iters:
            update_count = cp.zeros(1, dtype=cp.int32)
            # zig-zag: reverse the order of the colors on odd iterations
            for color in (range(4) if iteration % 2 == 0 else range(3, -1, -1)):
                indices = self.phases[color]
                update_func(
                    block=(sz_block,), grid=(-(-len(indices) // sz_block),),
                    args=(
                        disparity, best_costs, update_count, indices, feature1, feature2, seeds, neighbors, radius,
                        np.int32(width), np.int32(height), np.int32(channels), np.int32(len(indices)), np.int32(neighbors.shape[1]),
                        np.int32(direction), np.uint64(self.rng_seed), np.uint64(self.next_rng_offset())
                    )
                )
            update_ratio = np.float32(int(update_count[0])) / np.float32(num_seeds)
            iteration += 1
            if update_ratio < stop_ratio or last_update_ratio - update_ratio < 0.01:
                break
            last_update_ratio = update_ratio
        cp.cuda.runtime.deviceSynchronize()
        return best_costs, iteration

    def update_search_radius(self, neighbors, disparity, max_radius, scale):
        num_seeds = disparity.shape[0]
        radius = cp.empty(num_seeds, dtype=cp.float32)
        gpu_func = self.gpu_module.get_function('updateSearchRadius')
        sz_block = 128
        gpu_func(
            block=(sz_block,), grid=(-(-num_seeds // sz_block),),
            args=(radius, disparity, neighbors, np.int32(num_seeds), np.int32(neighbors.shape[1]), np.int32(max_radius), np.float64(scale))
        )
        cp.cuda.runtime.deviceSynchronize()
        return radius

    def pyramid_random_search(self, features1, features2, pyd_seeds, neighbors, direction):
        n_levels = len(features1)
        ratio = float(np.float32(self.params.ratio))
        num_seeds = pyd_seeds[0].shape[0]

        self.srand(self.params.rand_seed)
        disparities = [cp.zeros(num_seeds, dtype=cp.float32) for _ in range(n_levels)]
        # random initialization on the coarsest level
        init_radius = int(self.params.max_disparity * ratio ** (n_levels - 1) + 0.5)
        disparities[n_levels - 1] = self.init_random_disparity(num_seeds, init_radius)
        radius = cp.full(num_seeds, init_radius, dtype=cp.float32)

        for level in range(n_levels - 1, -1, -1): # coarse-to-fine
            self.propagate(features1[level], features2[level], pyd_seeds[level], neighbors, radius, disparities[level], direction)
            if level > 0:
                max_radius = int(min(32, self.params.max_disparity * ratio ** level + 0.5))
                radius = self.update_search_radius(neighbors, disparities[level], max_radius, 1.0 / ratio)
                disparities[level - 1] = disparities[level] * np.float32(1.0 / ratio)
        return disparities

    def one_pass(self, features1, features2, seeds, neighbors, direction):
        ratio = float(np.float32(self.params.ratio))
        pyd_seeds = self.generate_pyramid_seeds(seeds, features1)
        disparities = self.pyramid_random_search(features1, features2, pyd_seeds, neighbors, direction)
        # to the unit of the finest level
        return [disparity * np.float32((1.0 / ratio) ** level) for level, disparity in enumerate(disparities)]

    def cross_check(self, seeds, disparity, disparity2, klabels2):
        """1 for the seeds whose disparity of the left-to-right and of the right-to-left matching agree within check_threshold.
        A seed is not valid either if the disparity of it or of the seed of the other image that it falls on is out of
        [min_disparity, max_disparity] (the search itself is symmetric around 0)."""
        height, width = klabels2.shape
        b = self.params.border_width
        x, y = seeds[:, 0], seeds[:, 1]
        x2 = (x.astype(cp.float32) - disparity).astype(cp.int32)    # the corresponding point of the other image
        idx2 = klabels2[y, cp.clip(x2, 0, width - 1)]
        in_range = lambda d: (d >= self.params.min_disparity) & (d <= self.params.max_disparity)
        invalid = (x < b) | (x >= width - b) | (y < b) | (y >= height - b) \
            | (x2 < b) | (x2 >= width - b) \
            | ~in_range(disparity) | ~in_range(disparity2[idx2])
        diff = cp.abs(disparity - disparity2[idx2])
        return (~invalid & ~(diff > np.float32(self.params.check_threshold))).astype(cp.int32)

    def matching(self, img_left, img_right):
        """Matches the left and the right image (color, or gray, numpy float32 in 0..1, of the same size;
        magnitude_eps of the SIFT descriptor assumes this scale).
        Returns the matches that pass the cross check as an array of (x, y, disparity) of the left image, one row per seed."""
        to_color = lambda img: cp.asarray(img if img.ndim == 3 else np.stack([img] * 3, axis=2), dtype=cp.float32)
        self.img_ref = to_color(img_left)
        self.img_other = to_color(img_right)
        self.setup_module()

        pyramid_ref = self.construct_pyramid(self.img_ref, self.params.ratio, self.params.min_width)
        pyramid_other = self.construct_pyramid(self.img_other, self.params.ratio, self.params.min_width)
        features_ref = self.compute_features(pyramid_ref)
        features_other = self.compute_features(pyramid_other)

        height, width = self.img_ref.shape[:2]
        seeds, neighbors, klabels = self.generate_seeds(height, width)

        disparity_ref = self.one_pass(features_ref, features_other, seeds, neighbors, -1)[0]
        disparity_other = self.one_pass(features_other, features_ref, seeds, neighbors, 1)[0]

        valid = self.cross_check(seeds, disparity_ref, disparity_other, klabels).astype(bool)
        matches = cp.concatenate([seeds[valid].astype(cp.float32), disparity_ref[valid, None]], axis=1)
        return matches.get()
