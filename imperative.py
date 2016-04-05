import mxnet as mx
import pdb as pdb

#Initialise memory on CPU and GPU
cpu_tensor = mx.nd.zeros((10,), ctx=mx.cpu())
gpu_tensor = mx.nd.zeros((10,), ctx=mx.gpu(0))

ctx = mx.cpu() # which context to put memory on.
a = mx.nd.ones((10, ), ctx=ctx)
b = mx.nd.ones((10, ), ctx=ctx)
c = (a + b) / 10.
d = b + 1

# This will block untill the value of d has been computed.
numpy_d = d.asnumpy()

pdb.set_trace()