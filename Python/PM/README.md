# PM-GPU
***PM-GPU*** is a GPU implementation for the PatchMatch stereo algorithm [1].
By modifying checkerboard diffusion [2] for rectified stereo scenario the spatial propagation as well as the perturbation are fully parallelized on GPU.
For random sampling in GPU kernel the author implemented light-weight LFSR (linear feedback shift register) pseudo random generator.
The author was concerned that it is too simple, however, resulting disparity seemed reasonable.
The implementation provides an additional experimental option consistent gradient operator [3].
However the author did not evaluate the effectiveness of it quantitatively.

## Reference
1. M. Bleyer, C. Rhemann, and C. Rother. Patchmatch stereo-stereo matching with slanted support windows. In Proc. of BMVC (Vol. 11, pp. 1-11). August 2011.
1. S. Galliani, K. Lasinger and K. Schindler. Massively parallel multiview stereopsis by surface normal diffusion. In Proc. of ICCV (pp. 873-881). December 2015.
1. S. Ando. Consistent gradient operators. IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(3), 252-265. March 2000.
