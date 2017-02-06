import numpy as np
from scipy.optimize import check_grad
import pdb as pdb


def convolve(A, B):
    return np.convolve(A, B, mode='valid')

def forward(W, I):    
    return convolve(I, W)


def gradient_checking(func, x, index, epsilon=1e-6):
    x_curr = x.copy()
    x_curr[index] += epsilon
    err1 = func(x_curr)
    x_curr = x.copy()
    x_curr[index] -= epsilon
    err2 = func(x_curr)
    return (err2-err1)/(2*epsilon)





# Arguments:
#    W              : Weights (vector)
#    I              : Inputs (vector)
#    target         : Desired output (vector)
#    forward        : function(W, I) for the forward algorithm
#    supervise      : function(O, target) that returns the error gradient dEdO
#    backward_dEdW  : function(I, dEdO) for the backward algorithm (weights)
#    backward_dEdI  : function(W, dEdO) for the backward algorithm (inputs)
#    iters          : Number of iterations (iteger)
#    rate           : Learning rate (between 0 to 1)


def dummy_sgd_weights(W, I, target, forward, supervise, backward_dEdW, iters, rate):
    for i in xrange(iters):
        err, dEdO = supervise(forward(W, I), target)
        dEdW = backward_dEdW(I, dEdO)
        W -= rate*dEdW/np.max(np.abs(dEdW))
    return W

def dummy_sgd_inputs(W, I, target, forward, supervise, backward_dEdI, iters, rate):
    for i in xrange(iters):
        err, dEdO = supervise(forward(W, I), target)
        dEdI = backward_dEdI(W, dEdO)
        I -= rate*dEdI/np.max(np.abs(dEdI))
    return I



def supervise(O, target):
    error = np.mean(np.square(O-target))
    dEdO = 2.0*(O-target)/(len(O)+0.0)
    return error, dEdO


########################################################

def kinda_backward_inputs(W, dEdO):
    N = len(dEdO)
    M = (len(W)-1)/2
    dEdO = np.pad(dEdO, (M, M), mode='constant', constant_values=0.0)
    dEdI = np.zeros(N)
    for i in xrange(M,N+len(W)):
        dEdI[i] = np.dot(dEdO[i-M:i+M], W)
    return dEdI


def backward_dEdI(W, dEdO):
    M = len(W)-1
    return forward(np.pad(dEdO, (M, M), mode='constant', constant_values=0.0), W[::-1])



def backward_dEdW(I, dEdO):
    return forward(dEdO, I[::-1])


I0 = np.random.normal(0.0, 1.0, 1000)
W0 = np.random.normal(0.0, 1.0, 50)
target = forward(np.random.normal(0.0, 1.0, 1000), W0)

pdb.set_trace()

check_dEdI = check_grad(func=lambda I: supervise(forward(I, W0), target)[0],
                                       grad=lambda I: backward_dEdI(W0, supervise(forward(I, W0), target)[1]),
                                       x0=I0)

check_dEdW = check_grad(func=lambda W: supervise(forward(I0, W), target)[0],
                                       grad=lambda W: backward_dEdW(I0, supervise(forward(I0, W), target)[1]),
                                       x0=W0)

print 'Gradient Checking:'
print '\t dEdI = %f' % check_dEdI
print '\t dEdW = %f' % check_dEdW


