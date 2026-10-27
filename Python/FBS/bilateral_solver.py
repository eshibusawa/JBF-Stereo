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

"""The Fast Bilateral Solver [1] on the GPU (CuPy RawModule).

    solver = BilateralSolver(BilateralSolverParams())
    solver.set_reference(reference_rgb)             # (height, width, 3) uint8, RGB: builds the bilateral grid
    result = solver.solve(target, confidence)       # (height, width) or (height, width, channels), float

The grid depends on the reference image only, so it is built once and reused for any number of targets (and channels). The
solution is linear in the target, so the target is not scaled: the result has the range of the target. The confidence is the
weight of each target pixel (0 = ignore the pixel); its scale matters relative to lam. A pixel whose vertex is not connected
(through the neighbors of the grid) to any pixel with a confidence has no data to be filled from: the result there is 0.

The kernels are in bilateral_solver_cuda.cu. The system A y = b of the bilateral space is solved matrix-free by the
preconditioned conjugate gradient, as in the reference implementation of [1] (scripts/bilateral_solver.py of this repository).
The computation is in float32, and the additions of the splat use atomics, so the result is not bit-exact between runs.

[1] Barron, J. T., & Poole, B. (2016). The fast bilateral solver. In B. Leibe, J. Matas, N. Sebe, & M. Welling (Eds.), Computer Vision - ECCV 2016 (Lecture Notes in Computer Science, Vol. 9905, pp. 617-633). Springer.
"""

import math
import os

import numpy as np
import cupy as cp


class BilateralSolverParams:
    def __init__(self):
        self.sigma_spatial = 8.0
        self.sigma_luma = 4.0
        self.sigma_chroma = 4.0
        self.lam = 128.0                    # the strength of the smoothness term
        self.A_diag_min = 1e-5              # lower bound of the diagonal of A in the Jacobi preconditioner
        self.cg_tol = 1e-5                  # stop if |r| <= cg_tol |b|
        self.cg_maxiter = 25
        self.bistochastize_maxiter = 10


class BilateralSolver:
    DIM = 5                                 # x, y, Y, U, V
    BLOCK_1D = 256
    BLOCK_2D = (16, 16)

    def __init__(self, params=None):
        self.params = params if params is not None else BilateralSolverParams()
        self.module = None
        self.height = self.width = 0
        self.nvertices = 0
        self.splat_idx = None               # (npixels,) int32: the vertex of each pixel
        self.neighbors = None               # (2 * DIM, nvertices) int32: the neighbors of each vertex, -1 if none
        self.counts = None                  # (nvertices,) float32: the number of the pixels of each vertex
        self.n = None                       # (nvertices,) float32: the diagonal Dn of the bistochastization
        self.last_cg_iterations = []        # of the channels of the last solve
        self.last_cg_residuals = []         # |r| / |b| at the end

    def compile_module(self):
        fpfn = os.path.join(os.path.dirname(__file__), 'bilateralSolver.cu')
        with open(fpfn, 'r') as f:
            cuda_source = f.read()

        cuda_source = cuda_source.replace('FBS_NUM_DIMS', str(self.DIM))
        cuda_source = cuda_source.replace('FBS_SIGMA_SPATIAL', str(self.params.sigma_spatial) + 'f')
        cuda_source = cuda_source.replace('FBS_SIGMA_LUMA', str(self.params.sigma_luma) + 'f')
        cuda_source = cuda_source.replace('FBS_SIGMA_CHROMA', str(self.params.sigma_chroma) + 'f')
        cuda_source = cuda_source.replace('FBS_LAMBDA', str(self.params.lam) + 'f')
        cuda_source = cuda_source.replace('FBS_A_DIAG_MIN', str(self.params.A_diag_min) + 'f')

        self.module = cp.RawModule(code=cuda_source)
        self.module.compile()
        self.k = {name: self.module.get_function(name) for name in (
            'computeHash', 'findNeighbors', 'bistochastizeStep', 'splat', 'slice',
            'applyA', 'inverseDiagonal', 'cgUpdate', 'cgDirection')}

    def launch_1d(self, name, count, *args):
        grid = (math.ceil(count / self.BLOCK_1D),)
        self.k[name](grid, (self.BLOCK_1D,), args)

    # ------------------------------------------------------------------------------------------------------------------------
    # grid

    def compute_strides(self, height, width):
        """Mixed radix of the hash. The radix of a dimension is its largest coordinate + 2 (the coordinate range is [0, largest]):
        the neighbor + 1 is at most largest + 1, and the neighbor - 1 of 0 borrows from the next dimension and lands on the
        value radix - 1 = largest + 1 of this dimension, where no vertex exists. The largest coordinate has 1 more than its
        value in float64: the pixel can fall on the next cell by the rounding of the float32 division. The largest value of
        the color channels is 256 (U and V reach 255.5)."""
        p = self.params
        largest = [
            math.floor((width - 1) / p.sigma_spatial) + 1,
            math.floor((height - 1) / p.sigma_spatial) + 1,
            math.floor(256.0 / p.sigma_luma) + 1,
            math.floor(256.0 / p.sigma_chroma) + 1,
            math.floor(256.0 / p.sigma_chroma) + 1]
        strides = [1]
        for c in largest[:-1]:
            strides.append(strides[-1] * (c + 2))
        if strides[-1] * (largest[-1] + 2) >= 2 ** 62:
            raise ValueError('the bilateral grid is too large for a 64 bit hash: increase the sigmas')
        return np.array(strides, dtype=np.int64)

    def set_reference(self, reference):
        """Build the bilateral grid of a reference image (height, width, 3) uint8, RGB (cv2.imread returns BGR)."""
        if self.module is None:
            self.compile_module()
        p = self.params
        ref = cp.ascontiguousarray(cp.asarray(reference))
        assert ref.dtype == cp.uint8 and ref.ndim == 3 and ref.shape[2] == 3
        self.height, self.width = ref.shape[:2]
        npixels = self.height * self.width

        strides = cp.asarray(self.compute_strides(self.height, self.width))
        pixel_hash = cp.empty(npixels, dtype=cp.int64)
        grid = (math.ceil(self.width / self.BLOCK_2D[0]), math.ceil(self.height / self.BLOCK_2D[1]))
        self.k['computeHash'](grid, self.BLOCK_2D, (
            pixel_hash, ref, strides, np.int32(self.width), np.int32(self.height)))

        # the vertices are the unique hashes (sorted); the inverse is the vertex of each pixel
        unique_hash, inverse, counts = cp.unique(pixel_hash, return_inverse=True, return_counts=True)
        self.nvertices = int(unique_hash.size)
        self.splat_idx = cp.ascontiguousarray(inverse.reshape(-1).astype(cp.int32))
        self.counts = counts.astype(cp.float32)

        self.neighbors = cp.empty((2 * self.DIM, self.nvertices), dtype=cp.int32)
        self.launch_1d('findNeighbors', self.nvertices, self.neighbors, unique_hash, strides, np.int32(self.nvertices))

        self.bistochastize()

    def bistochastize(self):
        """Iteration for the diagonal Dn such that Dn B Dn is close to bistochastic (the sums of the rows are m = n * (B n))."""
        n = cp.ones(self.nvertices, dtype=cp.float32)
        n_next = cp.empty_like(n)
        for _ in range(self.params.bistochastize_maxiter):
            self.launch_1d('bistochastizeStep', self.nvertices, n_next, n, self.counts, self.neighbors, np.int32(self.nvertices))
            n, n_next = n_next, n
        self.n = n

    # ------------------------------------------------------------------------------------------------------------------------
    # operators of the bilateral space (the arrays are float32 and contiguous)

    def splat(self, x):
        out = cp.zeros(self.nvertices, dtype=cp.float32)
        self.launch_1d('splat', x.size, out, x, self.splat_idx, np.int32(x.size))
        return out

    def slice(self, y):
        out = cp.empty(self.splat_idx.size, dtype=cp.float32)
        self.launch_1d('slice', out.size, out, y, self.splat_idx, np.int32(out.size))
        return out

    def apply_A(self, y, w_splat):
        out = cp.empty_like(y)
        self.launch_1d('applyA', self.nvertices, out, y, self.n, w_splat, self.neighbors,
                       np.int32(self.nvertices))
        return out

    def inverse_diagonal(self, w_splat):
        """The Jacobi preconditioner: 1 / max(diag(A), A_diag_min)."""
        out = cp.empty(self.nvertices, dtype=cp.float32)
        self.launch_1d('inverseDiagonal', self.nvertices, out, self.n, w_splat, self.neighbors,
                       np.int32(self.nvertices))
        return out

    # ------------------------------------------------------------------------------------------------------------------------
    # solver

    def solve(self, target, confidence):
        assert self.splat_idx is not None, 'call set_reference first'
        on_gpu = isinstance(target, cp.ndarray)
        t = cp.asarray(target, dtype=cp.float32)
        c = cp.asarray(confidence, dtype=cp.float32)
        assert t.shape[:2] == (self.height, self.width) and c.shape == (self.height, self.width)

        npixels = self.height * self.width
        t2 = t.reshape(npixels, -1)
        w = cp.ascontiguousarray(c.reshape(npixels))
        w_splat = self.splat(w)

        inv_diag = self.inverse_diagonal(w_splat)

        self.last_cg_iterations, self.last_cg_residuals = [], []
        out = cp.empty_like(t2)
        for ch in range(t2.shape[1]):
            x = cp.ascontiguousarray(t2[:, ch])
            b = self.splat(x * w)
            y = self.conjugate_gradient(b, w_splat, inv_diag)
            out[:, ch] = self.slice(y)
        out = out.reshape(t.shape)
        return out if on_gpu else cp.asnumpy(out)

    def conjugate_gradient(self, b, w_splat, inv_diag):
        p = self.params
        # flat initialization: the weighted mean of the target in each vertex (0 if the vertex has no confidence)
        has_weight = w_splat > 0
        y = cp.where(has_weight, b / cp.where(has_weight, w_splat, 1.0), 0.0).astype(cp.float32)

        r = b - self.apply_A(y, w_splat)
        b_norm = float(cp.linalg.norm(b))
        threshold = p.cg_tol * b_norm
        r_norm = float(cp.linalg.norm(r))
        iterations = 0
        if r_norm > threshold:
            z = inv_diag * r
            d = z.copy()                            # the search direction
            s = cp.zeros(3, dtype=cp.float32)       # r.z of the step, p.Ap, r.z of the next step
            s[0] = cp.vdot(r, z)
            while iterations < p.cg_maxiter:
                Ap = self.apply_A(d, w_splat)
                s[1] = cp.vdot(d, Ap)
                self.launch_1d('cgUpdate', self.nvertices, y, r, z, d, Ap, inv_diag, s, np.int32(self.nvertices))
                iterations += 1
                s[2] = cp.vdot(r, z)
                r_norm = float(cp.sqrt(cp.vdot(r, r)))
                if r_norm <= threshold:
                    break
                self.launch_1d('cgDirection', self.nvertices, d, z, s, np.int32(self.nvertices))
                s[0] = s[2]
        self.last_cg_iterations.append(iterations)
        self.last_cg_residuals.append(r_norm / b_norm if b_norm > 0 else 0.0)
        return y
