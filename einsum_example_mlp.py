import numpy as np
import pdb as pdb


sq = np.sqrt

def func1():
    np.random.seed(0)
    
    Ni = 9
    Nh = 5
    No = 1*1
    # 9,1,1
    #o = [ 6.25436202 ]
    
    x = np.random.normal(size=(Ni   ))
    W = np.random.normal(size=(Nh,Ni))
    V = np.random.normal(size=(No,Nh))
    
    # i -- input dim
    # h -- hidden dim
    # o -- output dim
    
    # Forward prop
    #h = np.einsum('hi,i->h', W, x)
    h = np.einsum('hi, i->h', W, x)
    o = np.einsum('oh, h->o', V, h)
    
    print h
    #print o
    #pdb.set_trace()
    #o = [ 6.25436202]

def func2():
    np.random.seed(0)

    # Input tensor: batch size, channels per batch, width per channel, height per channel
    b = 2
    c = 1
    w = 3
    h = 3

    # Weights tensor: number of feature maps, width, height as above
    m = 5
    
    n = np.random.normal(size=(c*w*h)).reshape((c,w,h))
    x = np.zeros(b*c*w*h).reshape((b,c,w,h))
    x[0] = n
    x[1] = n
    
    n = np.random.normal(size=(m*w*h)).reshape((m,w,h))
    #pdb.set_trace()
    W = np.zeros(b*m*w*h).reshape((b,m,w,h))
    W[0] = n
    W[1] = n
    #V = np.random.normal(size=(No,Nh)).reshape((Nh, 1, 1))
    
    # i -- input dim
    # h -- hidden dim
    # o -- output dim
    
    # Forward prop
    #h = np.einsum('hi,i->h', W, x)
    h = np.einsum('bmwh, bcwh->bm', W, x)
    #o = np.einsum('hjk, h->jk', V, h)
    
    print h
    #print o
    
    
    pdb.set_trace()

        
    





func1()
func2()

# Use tensor formulation now