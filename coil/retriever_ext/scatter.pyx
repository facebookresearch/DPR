# cython: wraparound = False
# cython: boundscheck = False
# cython: language_level=3
import numpy as np
cimport numpy as np
import cython
cimport cython
from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
def scatter_max(
        np.ndarray[float, ndim=1] src,
        np.ndarray[long, ndim=1] indices,
        np.ndarray[float, ndim=1] tgt,
):
    cdef long i, n=src.shape[0]
    if n < 16384:
            for i in range(n):
                if src[i] > 0:
                    tgt[indices[i]] = max(tgt[indices[i]], src[i])
    else:
        for i in prange(n, nogil=True, schedule='static'):
            if src[i] > 0:
                tgt[indices[i]] = max(tgt[indices[i]], src[i])