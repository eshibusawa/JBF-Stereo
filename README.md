# JBF-Stereo
***JBF-Stereo*** is an implementation for disparity refinement by using joint bilateral filtering (JBF).
The refinement filter consists of joint bilateral filtering and joint nearest filtering, and both filters are CUDA-based GPU implementation.

## Result of disparity refinement by using Joint Bilateral Filter
<img src="data/teddy/im2.png" width="237px"/><img src="imgs/disparity_block_matching.png" width="237px"/><img src="imgs/disparity_refined.png" width="237px"/><img src="imgs/disparity_gt.png" width="237px"/>

## Result of PatchMatch Stereo
<img src="data/teddy/im2.png" width="237px"/><img src="imgs/disparity_pm.png" width="237px"/>

see more detail for [PatchMatch](./Python/PM/README.md).

## Result of ELAS (Efficient Large-Scale Stereo Matching)
<img src="data/conesH/im2.png" width="237px"/><img src="imgs/disparity_elas_l.png" width="237px"/><img src="data/conesH/im6.png" width="237px"/><img src="imgs/disparity_elas_r.png" width="237px"/>

see more detail for [ELAS](./Python/ELAS/README.md).

## Requirements
***JBF-Stereo*** requires the following libraries:
+ cupy
+ opencv
+ scipy
+ nose (only for testing)
```sh
pip install -r requirement.txt
```

## Usage
```sh
# compute disparity using block matching and apply JBF, PM, ELAS
python Python/stereo_main.py
