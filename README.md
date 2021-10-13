# defringe
CCD infrared image defringing code in python, using cupy for GPU acceleration. Based on matrix completion of noisy matrix with low-rank regularization via nuclear norm. See https://arxiv.org/abs/2109.02562 for details.

Main driver routine is algorithms.doit()
Input, preprocessed (overscan corrected, flat fielded) CCD images are available from the author on request.
