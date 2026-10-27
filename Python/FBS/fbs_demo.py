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

"""Filter a target image (a depth map, ...) with the GPU Fast Bilateral Solver, edge-aware by a reference image.

    python Python/FBS/fbs_demo.py Python/FBS/data/reference.png Python/FBS/data/target.png Python/FBS/data/confidence.png -o result.png

The target and the confidence are gray images (8 or 16 bit); they are scaled to 0 .. 1, and the result is written in the type
of the target. With --compare the result is compared with cv2.ximgproc.fastBilateralSolverFilter (the CPU version of OpenCV
contrib, 8 bit).
"""

import argparse
import time

import cv2
import numpy as np
import cupy as cp

from bilateral_solver import BilateralSolver, BilateralSolverParams


def read_gray(fn):
    img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(fn)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('reference')
    parser.add_argument('target')
    parser.add_argument('confidence', nargs='?', help='default: 1 everywhere')
    parser.add_argument('-o', '--output', default='result.png')
    parser.add_argument('--sigma-spatial', type=float, default=8.0)
    parser.add_argument('--sigma-luma', type=float, default=4.0)
    parser.add_argument('--sigma-chroma', type=float, default=4.0)
    parser.add_argument('--lam', type=float, default=128.0)
    parser.add_argument('--cg-maxiter', type=int, default=25)
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()

    reference = cv2.cvtColor(cv2.imread(args.reference, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    target_raw = read_gray(args.target)
    scale = float(np.iinfo(target_raw.dtype).max)
    target = target_raw.astype(np.float32) / scale
    if args.confidence is None:
        confidence = np.ones(target.shape, dtype=np.float32)
    else:
        conf_raw = read_gray(args.confidence)
        confidence = conf_raw.astype(np.float32) / float(np.iinfo(conf_raw.dtype).max)

    params = BilateralSolverParams()
    params.sigma_spatial, params.sigma_luma, params.sigma_chroma = args.sigma_spatial, args.sigma_luma, args.sigma_chroma
    params.lam, params.cg_maxiter = args.lam, args.cg_maxiter

    solver = BilateralSolver(params)
    solver.set_reference(reference)                         # compiles the kernels
    solver.solve(target, confidence)                        # warm-up of CuPy
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    solver.set_reference(reference)
    cp.cuda.Device().synchronize()
    t1 = time.perf_counter()
    result = solver.solve(target, confidence)
    t2 = time.perf_counter()
    print(f'{reference.shape[1]}x{reference.shape[0]}: {solver.nvertices} vertices, '
          f'grid {1000 * (t1 - t0):.1f} ms, solve {1000 * (t2 - t1):.1f} ms, '
          f'CG {solver.last_cg_iterations[0]} iterations (|r|/|b| = {solver.last_cg_residuals[0]:.2e})')

    out = np.clip(np.rint(result * scale), 0, scale).astype(target_raw.dtype)
    cv2.imwrite(args.output, out)
    print('wrote', args.output)

    if args.compare:
        t8 = np.clip(np.rint(target * 255), 0, 255).astype(np.uint8)
        c8 = np.clip(np.rint(confidence * 255), 0, 255).astype(np.uint8)
        t0 = time.perf_counter()
        cpu = cv2.ximgproc.fastBilateralSolverFilter(
            cv2.cvtColor(reference, cv2.COLOR_RGB2BGR), t8, c8, None,
            sigma_spatial=args.sigma_spatial, sigma_luma=args.sigma_luma, sigma_chroma=args.sigma_chroma,
            lambda_=args.lam, num_iter=args.cg_maxiter)
        t1 = time.perf_counter()
        gpu8 = result * 255
        d = np.abs(gpu8 - cpu.astype(np.float32))
        print(f'cv2.ximgproc (CPU): {1000 * (t1 - t0):.0f} ms; |gpu - cpu| in 8 bit: mean {d.mean():.2f}, max {d.max():.1f}; '
              f'|gpu - target|: mean {np.abs(gpu8 - t8).mean():.2f}; |cpu - target|: mean {np.abs(cpu.astype(np.float32) - t8).mean():.2f}')


if __name__ == '__main__':
    main()
