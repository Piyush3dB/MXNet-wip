import find_mxnet
import mxnet as mx
import numpy as np
import logging
import pdb as pdb

   # logging
head = '%(asctime)-15s Node['  + '] %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)
#logging.info('start with arguments %s', args)

# Generate 101 training points
trX = np.linspace(-1, 1, 101)
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33 * 0

X = mx.sym.Variable('data')
Y = mx.sym.Variable('softmax_label')

Y_ = mx.sym.FullyConnected(data=X, num_hidden=1)
cost = mx.sym.LinearRegressionOutput(data=Y_, label=Y)

# Shape inference:
cost.infer_shape(data=(1,1,1,1))


model = mx.model.FeedForward(symbol=cost, num_epoch=1000, learning_rate=0.01, epoch_size=1, numpy_batch_size=1, ctx=mx.cpu(0))


batch_end_callback = []
batch_end_callback.append(mx.callback.Speedometer(1,1))


print "model before training:"
#print model.arg_params['fullyconnected0_weight'].asnumpy()
#print model.arg_params['fullyconnected0_bias'].asnumpy()

model.fit(X=trX, y=trY, batch_end_callback = batch_end_callback)

print "model after training:"
print model.arg_params['fullyconnected0_weight'].asnumpy()
print model.arg_params['fullyconnected0_bias'].asnumpy()


##
###should print:
##
#model after training:
#[[ 2.05157518]]
#[ 0.01423269]

pdb.set_trace()