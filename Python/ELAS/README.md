# ELAS-GPU
***ELAS-GPU*** is an unofficial implementation for the ELAS algorithm [1].
For understanding the algorithm the author implemented it from scratch. So it produces almost same result but has slight differences due to the following reason.
* boundary processing (feature extraction, grid point diffusion, adaptive mean filter, median filter)
* right -> left disparity handling of right image disparity plane estimation
* 3x3 linear solver (the author uses Cramer's rule for reducing branch instead of Gaussian elimination)
* Delaunay triangulation (the autor uses scipy instead of [Triangle](http://www.cs.cmu.edu/~quake/triangle.html ))

The epsillon value of Gaussian elimination in original implementation seems too small to avoid invalid solution.
The flitering implementation is not optimized and not so fast. The author think that ideally it is desiable to use [CTA shuffling](https://developer.nvidia.com/blog/cooperative-groups/) or more producutive library (e.g., [CUB](https://nvlabs.github.io/cub/)).

The author think almost full CUDA and Python implementation is useful for platform that do not support SSE especially for Jetson.

## Reference
1. A. Geiger, R. Martin, and U. Urtasun. Efficient Large-Scale Stereo Matching. In Proc. of 10th Asian Conference on Computer Vision (ACCV 2010). volume 1, pages 25-38, Queenstown, New Zealand, November 2010.