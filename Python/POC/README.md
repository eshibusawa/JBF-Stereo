# POC-GPU
***POC-GPU*** is a GPU implementation of stereo correspondence with 1D phase-only correlation (POC) [1], intended for sub-pixel estimation of rectified stereo pairs.
For every pixel a 1D window is cut out along the epipolar line, the cross spectra of several rows are accumulated, normalized to the phase only and transformed back,
and the peak is refined by the peak evaluation formula (PEF) of [1].
A window is transformed by the lanes of a warp (radix-2 butterflies with `__shfl_xor_sync`), so the window width is 8, 16, 32 or 64.
[pyramid.py](pyramid.py) adds the image pyramid and the coarse-to-fine estimation (Sec. 3 of [1]): the upper levels give pixel precision disparities, the finest level gives the sub-pixel disparity.

Options of `phase_only_correlation_parameters` ([poc.py](poc.py)):
| parameter | content |
|---|---|
| `window_width` | width of the window (8, 16, 32, 64) |
| `averaging_window_height` | number of rows whose cross spectra are accumulated |
| `max_disparity` | decides the number of pyramid levels (`floor(window_width / 4) * 2^level >= max_disparity`) |
| `is_use_hann_window` | apply the Hann window |
| `is_spectrum_weighting` | use only the lowest quarter of the frequency bins (rectangular weighting) |
| `is_use_dc_cut` | subtract the Hann-weighted mean of the window before the transform (on by default); without it the DC leaks through the Hann window into the neighboring bins |
| `address_mode` | border of the image: `mirror` (default), `wrap` or `border` (zero); also for the image pyramid |

## Usage
```python
# run in Python/POC
import numpy as np
import add_path     # makes Python/Texture importable
from poc import phase_only_correlation_parameters, phase_only_correlation
from pyramid import image_pyramid

param = phase_only_correlation_parameters()
param.window_width = 32
param.averaging_window_height = 17
param.max_disparity = 64
param.is_use_hann_window = True
param.is_spectrum_weighting = True
poc = phase_only_correlation(param, np.float32)
pyramid = image_pyramid(param)
# gray images (float32, 0..255) of a rectified pair, the left image is the reference
disparity = pyramid.estimate_disparity(poc, img_left, img_right)
```
The test is `pytest Python/POC`.

The sub-pixel estimate is smaller than the true value (towards disparity 0), more for blurred textures; a larger window, the Hann window and the spectrum weighting reduce it.
The estimate near the left and the right edge of the image is unreliable, because the windows of the coarse levels span `window_width * 2^level` pixels.

## Reference
1. Shibahara, T., Numa, N., Nagashima, S., Aoki, T., Nakajima, H., & Kobayashi, K. (2008). 一次元位相限定相関法に基づくステレオ画像の高精度サブピクセル対応付け手法 [A high-accuracy sub-pixel correspondence technique using 1D phase-only correlation]. 電子情報通信学会論文誌 D [IEICE Transactions on Information and Systems (Japanese edition)], J91-D(9), 2343–2356. (In Japanese)
