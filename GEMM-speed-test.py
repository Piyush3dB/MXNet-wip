import sys
import time
import numpy as np
import pdb as pdb

"""
Measure speed of GEMM using numpy
"""

np.random.seed(0)

N = 1000
dtype = np.double

A = np.random.rand(N, N).astype(dtype)
B = np.random.rand(N, N).astype(dtype)

t0 = time.time()
reps = 5
for ii in range(reps):
    #C = np.dot(A, B)
    C = np.einsum("ij,jk->ik", A, B);
    #pdb.set_trace()
t1 = time.time()

FLOPS = reps * 2 * N ** 3
print 'GFLOP/s', FLOPS / (1000 ** 3) / (t1 - t0)