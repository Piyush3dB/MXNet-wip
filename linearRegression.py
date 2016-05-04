import find_mxnet
import mxnet as mx
import numpy as np
import logging
import pdb as pdb
import matplotlib.pyplot as plt


##
# Setup logging
##
head = '%(asctime)-15s Node['  + '] %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

##
# Generate 101 training points
##
trX = np.linspace(-1, 1, 101)
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33 * 1

trX = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
trY = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])

##
# Define network
##
sm  = mx.sym.Variable('softmax_label')
d   = mx.sym.Variable('data')
fc  = mx.sym.FullyConnected(data=d, num_hidden=1)
net = mx.sym.LinearRegressionOutput(data=fc, label=sm)

##
# Shape inference:
##
print net.infer_shape(data=(1,1,1,1))


##
# Setup monitoring callback
##
batch_end_callback = []
batch_end_callback.append(mx.callback.Speedometer(1,1))
#batch_end_callback.append(mx.callback.ProgressBar(100))


##
# Train the model
##
model = mx.model.FeedForward(symbol=net, num_epoch=100, learning_rate=0.01, epoch_size=1, numpy_batch_size=1, ctx=mx.cpu(0))
print "Start training:"
model.fit(X=trX, y=trY, batch_end_callback = batch_end_callback)

W = model.arg_params['fullyconnected0_weight'].asnumpy()[0]
b = model.arg_params['fullyconnected0_bias'].asnumpy()[0]


#pdb.set_trace()
##
# Show results
##
plt.plot(trX, trY, 'ro', label='Original data')
plt.plot(trX, W*trX+b, label='Fitted line')
plt.legend()
#plt.show()


print "Model params after training:"
print W
print b

##
# FeedForward has an dictionary of ndarray, arg_params. 
# Grab the parameters and convert them to numpy with asnumpy().
##
print {k:v.asnumpy() for k,v in model.arg_params.items()}


##
# For profiling:
##
#python -m cProfile -o profile_data.pyprof ./linearRegression.py
#python ./pyprof2calltree.py -i profile_data.pyprof -k
