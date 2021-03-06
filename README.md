# JBF-Stereo
***JBF-Stereo*** is an implementation for disparity refinment by using joint bilateral filtering (JBF).
The refinement filter consists of joint bilateral filtering and joint nearest filtering, and both filters are CUDA-based GPU implementation.

<img src="data/teddy/im2.png" width="237px"/><img src="imgs/disparity_block_matching.png" width="237px"/><img src="imgs/disparity_refined.png" width="237px"/><img src="imgs/disparity_gt.png" width="237px"/>

## Requirements
***JBF-Stereo*** requires the following libraries:
+ cupy
+ opencv
```sh
pip install -r requirement.txt
```

## Usage
```sh
# compute disparity using block matching and apply JBF
python Python/stereo_main.py
