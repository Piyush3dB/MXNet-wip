import find_mxnet
import mxnet as mx
import logging
import pdb as pdb
import numpy as np
import pdb as pdb
from scipy import signal
from scipy import ndimage as nd


#np.random.seed(0)

def test_normal_conv(data, weights):

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
    stride = (1,1)
    net    = mx.symbol.Convolution(img, num_filter=1, kernel=krns, stride=stride, dilate=dil, pad=pad, no_bias="true", name='conv')
    exector = net.bind(mx.cpu(), args={ 'input' : dataND, 'conv_weight' : weightsND})
    exector.forward(True)
    ot = exector.outputs[0].asnumpy()
    
    # Compare results
    print  np.sum(ot-nx)


   
def test_normal_conv_ds(data, weights):

    # Numpy native
    nx = signal.correlate2d(data[0,0], weights[0,0], boundary='fill', mode='same')
    nx = nx[0::2, 0::2]
    
    # MXNet
    dataND    = mx.nd.array(data)
    weightsND = mx.nd.array(weights)
    kernel = mx.symbol.Variable('kernel')
    img    = mx.symbol.Variable('input')
    pad    = (1,1)
    dil    = (1,1)
    krns   = (3,3)
    stride = (2,2)
    net    = mx.symbol.Convolution(img, num_filter=1,kernel=krns, stride=stride, dilate=dil, pad=pad, no_bias="true", name='conv')
    exector = net.bind(mx.cpu(), args={ 'input' : dataND, 'conv_weight' : weightsND})
    exector.forward(True)
    ot = exector.outputs[0].asnumpy()
    
    # Compare results
    #pdb.set_trace()
    print  np.sum(ot-nx)


  


def test_normal_conv_wide(data, weights):

    # Numpy native

    # Interlace input into channels
    c1 = data[0,0][0::2, 0::2]
    c2 = data[0,0][0::2, 1::2]
    c3 = data[0,0][1::2, 0::2]
    c4 = data[0,0][1::2, 1::2]

    # Filter each channel
    nx1 = signal.correlate2d(c1, weights[0,0], boundary='fill', mode='same')
    nx2 = signal.correlate2d(c2, weights[0,0], boundary='fill', mode='same')
    nx3 = signal.correlate2d(c3, weights[0,0], boundary='fill', mode='same')
    nx4 = signal.correlate2d(c4, weights[0,0], boundary='fill', mode='same')
    
    # Deinterlace filtered channels to reconstruct result
    nx  = np.zeros_like(data[0,0])
    nx[0::2, 0::2] = nx1
    nx[0::2, 1::2] = nx2
    nx[1::2, 0::2] = nx3
    nx[1::2, 1::2] = nx4

    # MXNet
    dataND    = mx.nd.array(data)
    weightsND = mx.nd.array(weights)
    kernel = mx.symbol.Variable('kernel')
    img    = mx.symbol.Variable('input')
    pad    = (2,2)
    dil    = (2,2)
    krns   = (3,3)
    stride = (1,1)
    net    = mx.symbol.Convolution(img, num_filter=1, kernel=krns, stride=stride, dilate=dil, pad=pad, no_bias="true", name='conv')
    exector = net.bind(mx.cpu(), args={ 'input' : dataND, 'conv_weight' : weightsND})
    exector.forward(True)
    ot = exector.outputs[0].asnumpy()
    
    # Compare results
    #pdb.set_trace()
    print  np.sum(ot-nx)



def test_normal_conv_inds(data, weights):

    # Numpy native

    # Downsample input signal
    data2 = data[0:: ,0::, 0::2, 0::2]

    # Filter downsampled signal
    nx = signal.correlate2d(data2[0,0], weights[0,0], boundary='fill', mode='same')
    
    # MXNet
    dataND    = mx.nd.array(data)
    weightsND = mx.nd.array(weights)
    kernel = mx.symbol.Variable('kernel')
    img    = mx.symbol.Variable('input')
    pad    = (2,2)
    dil    = (2,2)
    krns   = (3,3)
    stride = (2,2)
    net    = mx.symbol.Convolution(img, num_filter=1,kernel=krns, stride=stride, dilate=dil, pad=pad, no_bias="true", name='conv')
    exector = net.bind(mx.cpu(), args={ 'input' : dataND, 'conv_weight' : weightsND})
    exector.forward(True)
    ot = exector.outputs[0].asnumpy()
    
    # Compare results
    print  np.sum(ot-nx)



if __name__ == '__main__':

    data    = np.random.randn(1, 1, 256, 256).astype(np.float32)
    weights = np.random.randn(1, 1,   3,   3).astype(np.float32)

    test_normal_conv(     data, weights)
    test_normal_conv_ds(  data, weights)
    test_normal_conv_wide(data, weights)
    test_normal_conv_inds(data, weights)
