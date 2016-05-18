import numpy as np
import pdb as pdb


A = np.arange(0,2*3*4).reshape((2,3,4), order='C')

#A = np.random.randint(0,9,(2,4,4)) # Sample input array
                    # Sample blocksize (rows x columns)
B = [2,2]
skip=[2,2]
# Parameters 
D,M,N = A.shape
col_extent = N - B[1] + 1
row_extent = M - B[0] + 1

# Get Starting block indices
start_idx = np.arange(B[0])[:,None]*N + np.arange(B[1])

# Generate Depth indeces
didx=M*N*np.arange(D)
start_idx=(didx[:,None]+start_idx.ravel()).reshape((-1,B[0],B[1]))

# Get offsetted indices across the height and width of input array
offset_idx = np.arange(row_extent)[:,None]*N + np.arange(col_extent)

# Get all actual indices & index into input array for final output
out = np.take (A,start_idx.ravel()[:,None] + offset_idx[::skip[0],::skip[1]].ravel())

pdb.set_trace()