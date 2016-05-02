import find_mxnet
import mxnet as mx
import numpy as np
import logging

   # logging
head = '%(asctime)-15s Node['  + '] %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)
#logging.info('start with arguments %s', args)


trX = np.linspace(-1, 1, 101)
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33

X = mx.sym.Variable('data')
Y = mx.sym.Variable('softmax_label')

Y_ = mx.sym.FullyConnected(data=X, num_hidden=1)

cost = mx.sym.LinearRegressionOutput(data=Y_, label=Y)

model = mx.model.FeedForward(symbol=cost, num_epoch=100, learning_rate=0.05, epoch_size=1, numpy_batch_size=1, ctx=mx.gpu(0))


batch_end_callback = []
batch_end_callback.append(mx.callback.Speedometer(1,1))


model.fit(X=trX, y=trY, batch_end_callback = batch_end_callback)