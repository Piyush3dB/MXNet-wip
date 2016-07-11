import find_mxnet
import mxnet as mx
import logging
import pdb as pdb
import numpy as np
import pdb as pdb
from scipy import signal
from scipy import ndimage as nd


#np.random.seed(0)

def test_normal_conv():

	# Data    
    data = np.random.randn(1,1,3,3).astype(np.float32)
    weights = np.random.randn(1,1,3,3).astype(np.float32)

    # Numpy native
    nx = signal.correlate2d(data[0,0], weights[0,0], boundary='fill', mode='same')
    
    # MXNet
    dataND    = mx.nd.array(data)
    weightsND = mx.nd.array(weights)
    kernel = mx.symbol.Variable('kernel')
    img    = mx.symbol.Variable('input')
    pad    = (1,1)
    dil    = (1,1)
    krns   = (3,3)
    net    = mx.symbol.Convolution(img, num_filter=1,kernel=krns, dilate=dil, pad=pad, no_bias="true", name='conv')
    exector = net.bind(mx.cpu(), args={ 'input' : dataND, 'conv_weight' : weightsND})
    exector.forward(True)
    ot = exector.outputs[0].asnumpy()
    
    # Compare results
    print  np.sum(ot-nx)


   





if __name__ == '__main__':
	test_normal_conv()
