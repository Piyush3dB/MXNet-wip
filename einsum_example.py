#  1: Imports.
import numpy as np

#  2: Invocation. Requires format string + any # of input args.
arg0 = np.random.normal(...)
arg1 = np.random.normal(...)
...
argn = np.random.normal(...)

# ...

dst  = np.einsum("               ", arg0, arg1, ..., argn)

#  3: Format string. Incomplete example with 3 input args.
dst  = np.einsum("  ,   ,  ->    ", arg0, arg1, arg2)

#  4: Format string. Incomplete example with 3 input args.
dst  = np.einsum("  ,   ,  ->    ", arg0, arg1, arg2)




assert arg0.ndim == len("  ")                # (Order-2 Tensor)
assert arg1.ndim == len("   ")               # (Order-3 Tensor)
assert arg2.ndim == len("  ")                # (Order-2 Tensor)
assert dst .ndim == len("    ")              # (Order-4 Tensor)

#  5: Format string. Complete examples with 1 and 2 args.
s = np.einsum("a->",       v   )
T = np.einsum("ij->ji",    M   )
C = np.einsum("mn,np->mp", A, B)


assert v.ndim == len("a")
assert s.ndim == len("")

assert M.ndim == len("ij")
assert T.ndim == len("ji")

assert A.ndim == len("mn")
assert B.ndim == len("np")
assert C.ndim == len("mp")

#  6: Elaborated Example. Matrix multiplication.
#     C     =     A     *     B
#  Ni x Nj  =  Ni x Nk  *  Nk x Nj
#               |    ^-----^     |
#               |     Match      |
#               |                |
#               -------- ---------
#                       V
#  Ni x Nj  <------- Ni x Nj

C = np.empty((Ni,Nj))
for i in xrange(Ni):
      for j in xrange(Nj):
             total = 0
             
             for k in xrange(Nk):
                   total += A[i,k]*B[k,j]
             
             C[i,j] = total




C = np.einsum("ik,kj->ij", A, B)

#  7. Free Indices
C = np.empty((Ni,Nj))
for i in xrange(Ni):
      for j in xrange(Nj):
             total = 0
             
             for k in xrange(Nk):
                   
                   total += A[i,k]*B[k,j]
             
             C[i,j] = total




C = np.einsum("ik,kj->ij", A, B)

#  8. Summation Indices
C = np.empty((Ni,Nj))
for i in xrange(Ni):
      for j in xrange(Nj):
             total = 0
             
             for k in xrange(Nk):
                   
                   total += A[i,k]*B[k,j]
             
             C[i,j] = total




C = np.einsum("ik,kj->ij", A, B)

#  9. Elaborated Example. Matrix diagonal extraction.
d = np.empty((Ni))
for i in xrange(Ni):
      total = 0
      
      
      total +=       M[i,i]
      
      d[i] = total




d = np.einsum("ii->i", M)

# 10. Elaborated Example. Matrix trace.
Tr = 0   # Scalar! Has dimension 0 and no indexes


total = 0

for i in xrange(Ni):
      total +=       M[i,i]

Tr = total




Tr = np.einsum("ii->", M)

# 11. Elaborated Example. Quadratic Form.
x = 0


total = 0

for s in xrange(Ns):
      for t in xrange(Nt):
            total += v[s]*M[s,t]*v[t]

x = total




x = np.einsum("s,st,t->", v, M, v)

# 12. Elaborated example. Batched outer product.

R = np.empty((NB,ni,nj))
for B in xrange(NB):
      for i in xrange(Ni):
            for j in xrange(Nj):
                   total = 0
                   
                   
                   total          += P[B,i]*Q[B,j]
                   
                   R[B,i,j] = total




R = np.einsum("Bi,Bj->Bij", P, Q)

# 13. Natural consequences of einsum definition.
#     Requirements on size of individual axes.
C =            np.einsum("ik,kj->ij", A, B)



assert A.shape == (Ni,Nk)
assert B.shape ==         (Nk,Nj)
assert C.shape ==                 (Ni,Nj)

#     Requirement for identical size of certain axes due to
#     shared index label.
#         Example 1: Matrix Multiplication
C =            np.einsum("ik,kj->ij", A, B)



assert           A.shape[1] == B.shape[0]         #  == Nk
assert           A.shape[0] == C.shape[0]         #  == Ni
assert           B.shape[1] == C.shape[1]         #  == Nj

#         Example 2: Matrix Diagonal Extraction
d =            np.einsum("ii->i", D)



assert           D.shape[0] == D.shape[1]         #  == Ni
assert           D.shape[1] == d.shape[0]         #  == Ni

# 14: Format Strings. Rules.

#    Bad. Number of input index groups doesn't match number of
#         arguments.
np.einsum("ab,bc->ac", A)

#    Bad. Indexes must be ASCII upper/lowercase letters.
np.einsum("012,1^%->:;?", A, B)

#    Bad. Argument 0 has 3 dimensions but only 2 indexes are
#         given.
A = np.random.normal(size = (2,3,4))
B = np.random.normal(size = (4,5,6))
np.einsum("ab,bcd->a", A, B)

#    Bad. One of the output indexes isn't in the set of all
#         input indexes.
np.einsum("ab,bc->acz", A, B)

#    Bad. Output has a repeated index.
np.einsum("ab,bc->baa", A, B)

#    Bad. Mismatches in the sizes of input argument axes
#         that are labelled with the same index.
A = np.random.normal(size = (2,3,4))
B = np.random.normal(size = (3,4,5))
np.einsum("ckj,cqq->c", A, B)

assert      A.shape[0] == B.shape[0]          # ERROR: 2 != 3
assert      B.shape[1] == B.shape[2]          # ERROR: 4 != 5

# 15: MLP Backprop done easily (stochastic version).
#     h = sigmoid(Wx + b)
#     y = softmax(Vh + c)
Ni = 784
Nh = 500
No =  10

W  = np.random.normal(size = (Nh,Ni))  # Nh x Ni
b  = np.random.normal(size = (Nh,))    # Nh
V  = np.random.normal(size = (No,Nh))  # No x Nh
c  = np.random.normal(size = (No,))    # No

# Load x and t...
x, t  = train_set[k]

# With a judicious, consistent choice of index labels, we can
# express fprop() and bprop() extremely tersely; No thought
# needs to be given about the details of shoehorning matrices
# into np.dot(), such as the exact argument order and the
# required transpositions.
# 
# Let
#
#     'i' be the input  dimension label.
#     'h' be the hidden dimension label.
#     'o' be the output dimension label.
#
# Then

# Fprop
ha    = np.einsum("hi, i -> h", W, x) + b
h     = sigmoid(ha)
ya    = np.einsum("oh, h -> o", V, h) + c
y     = softmax(ya)

# Bprop
dLdya = y - t
dLdV  = np.einsum("h , o -> oh", h, dLdya)
dLdc  = dLdya
dLdh  = np.einsum("oh, o -> h ", V, dLdya)
dLdha = dLdh * sigmoidgrad(ha)
dLdW  = np.einsum("i,  h -> hi", x, dLdha)
dLdb  = dLdha

# 16: MLP Backprop done easily (batch version).
#     But we want to exploit hardware with a batch version!
#     This is trivially implemented with simple additions
#     to np.einsum's format string, in addition to the usual
#     averaging logic required when handling batches. We
#     implement even that logic with einsum for demonstration
#     and elegance purposes.
batch_size = 128

# Let
#     'B' be the batch  dimension label.
#     'i' be the input  dimension label.
#     'h' be the hidden dimension label.
#     'o' be the output dimension label.
#
# Then

# Fprop
ha    = np.einsum("hi, Bi -> Bh", W, x) + b
h     = sigmoid(ha)
ya    = np.einsum("oh, Bh -> Bo", V, h) + c
y     = softmax(ya)

# Bprop
dLdya = y - t
dLdV  = np.einsum("Bh, Bo -> oh", h, dLdya) / batch_size
dLdc  = np.einsum("Bo     -> o ",    dLdya) / batch_size
dLdh  = np.einsum("oh, Bo -> Bh", V, dLdya)
dLdha = dLdh * sigmoidgrad(ha)
dLdW  = np.einsum("Bi, Bh -> hi", x, dLdha) / batch_size
dLdb  = np.einsum("Bh     -> h ",    dLdha) / batch_size