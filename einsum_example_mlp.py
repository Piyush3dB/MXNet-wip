import numpy as np
import pdb as pdb


sq = np.sqrt

def func1():
    np.random.seed(0)
    
    Ni = 9
    Nh = 1*1
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
    
    Ni = 3*3
    Nh = 1*1
    No = 1*1
    
    x = np.random.normal(size=(Ni   )).reshape((3,3))
    
    W = np.random.normal(size=(Nh,Ni)).reshape((Nh, 3, 3))
    V = np.random.normal(size=(No,Nh)).reshape((Nh, 1, 1))
    
    # i -- input dim
    # h -- hidden dim
    # o -- output dim
    
    # Forward prop
    #h = np.einsum('hi,i->h', W, x)
    h = np.einsum('hjk, jk->h', W, x)
    #o = np.einsum('hjk, h->jk', V, h)
    
    print h
    #print o
    
    
    pdb.set_trace()

        
    





func1()
func2()

# Use tensor formulation now