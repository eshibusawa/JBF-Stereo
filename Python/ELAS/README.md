# ELAS-GPU
***ELAS-GPU*** is an unofficial implementation for the ELAS algorithm [1].
The stereo matching pipeline runs entirely on GPU, with the exception of the Delaunay triangulation.
For understanding the algorithm the author implemented it from scratch. So it produces almost same result but has slight differences due to the following reason.
* boundary processing (feature extraction, grid point diffusion, adaptive mean filter, median filter)
* right -> left disparity handling of right image disparity plane estimation
* 3x3 linear solver (for reducing branch the author uses Cramer's rule instead of Gaussian elimination)
* Delaunay triangulation (the author uses scipy instead of [Triangle](http://www.cs.cmu.edu/~quake/triangle.html ))

The epsilon value of Gaussian elimination in original implementation seems too small to avoid invalid solution.
The filtering function of this implementation is not optimized and not so fast. The author thinks that ideally it is desirable to use [CTA shuffling](https://developer.nvidia.com/blog/cooperative-groups/) or more productive library (e.g., [CUB](https://nvlabs.github.io/cub/)).

The author thinks almost full CUDA and Python implementation is useful for platform that do not support SSE especially for Jetson.

# Reference
1. A. Geiger, R. Martin, and U. Urtasun. Efficient Large-Scale Stereo Matching. In Proc. of 10th Asian Conference on Computer Vision (ACCV 2010). volume 1, pages 25-38, Queens-town, New Zealand, November 2010.