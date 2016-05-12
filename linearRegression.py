import find_mxnet
import mxnet as mx
import numpy as np
import logging
import pdb as pdb
import matplotlib.pyplot as plt

#np.random.seed(0)

USE_ND_ITER = 1
USE_CSV_ITER = 0
##
# Setup logging
##
head = '%(asctime)-15s Node['  + '] %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

##
# Generate 101 training points
##
X = np.linspace(-1, 1, 101)
Y = 2*X + np.random.randn(*X.shape) * 0.33 * 1

#X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
#Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])









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


mon = mx.mon.Monitor(
    1,                 # Print every 100 batches
    stat_func=None,      # The statistics function defined above
    pattern='fullyconnected0_weight|fullyconnected0_bias',  # A regular expression. Only arrays with name matching this pattern will be included.
    sort=True)           # Sort output by name



##
# Train the model
##
model = mx.model.FeedForward(symbol=net, 
	                         num_epoch=300, 
	                         learning_rate=0.01, 
	                         epoch_size=1,
	                         numpy_batch_size=1, 
	                         ctx=mx.cpu(0))
print "Start training:"

if USE_ND_ITER:
    trX = mx.io.NDArrayIter(X, label=Y, batch_size=1, shuffle=True, last_batch_handle='pad')
    model.fit(X=trX, monitor=mon, batch_end_callback = batch_end_callback)

elif USE_CSV_ITER:
    trX = mx.io.CSVIter(data_csv="./X.csv" , data_shape=(1,),
                        label_csv="./Y.csv", label_shape=(1,),
                        batch_size=1)
    model.fit(X=trX, monitor=mon, batch_end_callback = batch_end_callback)

                           
else:
    trX = X
    trY = Y
    model.fit(X=trX, y=trY, monitor=mon, batch_end_callback = batch_end_callback)

W = model.arg_params['fullyconnected0_weight'].asnumpy()[0]
b = model.arg_params['fullyconnected0_bias'].asnumpy()[0]


#pdb.set_trace()
##
# Show results
##
plt.plot(X, Y, 'ro', label='Original data')
plt.plot(X, W*X+b, label='Fitted line')
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

#Model params after training:
#[ 0.29068446]
#0.145405
#{'fullyconnected0_weight': array([[ 0.29068446]], dtype=float32), 'fullyconnected0_bias': array([ 0.14540534], dtype=float32)}

