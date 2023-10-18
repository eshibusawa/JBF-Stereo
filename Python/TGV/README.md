# TGV-GPU
***TGV-GPU*** is a GPU implementation for the Total Generalized Variation (TGV) stereo algorithm [1].
Based on the original paper, for initialization, the author uses the Census transformation based locally adaptive support-weight aggregation [2] and WTA.
For the anisotropic diffusion tensor computation [3] the implementation uses the consistent gradient operator [5].
Although this is a full GPU implementation edge-segment based adaptive regularization based on the LSD [4] is NOT implemented.
For exhaustive search of regularized cost function the author adopts [CUB](https://nvlabs.github.io/cub/)'s block reduce algorithm.
The result is not so impressive, so any comments, bug fix and improvements are welcome.

## Reference
1. Kuschk, G., & Cremers, D. (2013). Fast and accurate large-scale stereo reconstruction using variational methods. In Proceedings of the IEEE International Conference on Computer Vision Workshops (pp. 700-707).
1. Yoon, K. J., & Kweon, I. S. (2006). Adaptive support-weight approach for correspondence search. IEEE transactions on pattern analysis and machine intelligence, 28(4), 650-656.
1. Werlberger, M., Trobin, W., Pock, T., Wedel, A., Cremers, D., & Bischof, H. (2009). Anisotropic Huber-L1 Optical Flow. In BMVC (Vol. 1, No. 2, p. 3).
1. Von Gioi, R. G., Jakubowicz, J., Morel, J. M., & Randall, G. (2008). LSD: A fast line segment detector with a false detection control. IEEE transactions on pattern analysis and machine intelligence, 32(4), 722-732.
1. S. Ando. Consistent gradient operators. IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(3), 252-265. March 2000.
