import find_mxnet
import mxnet as mx
import logging
import pdb as pdb
import numpy as np
from mxnet_utils import printStats, _str2tuple

# Set for repeatability
mx.random.seed(1)


def get_xor_net():
    """
    2,2,1 MLP configuration
    """
    outLabl = mx.sym.Variable('softmax_label')
    
    data = mx.symbol.Variable('data')
    fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=2)
    act1 = mx.symbol.Activation(data = fc1, name='tanh1', act_type="tanh")
    fc2  = mx.symbol.FullyConnected(data = act1, name='fc2', num_hidden=1)
    net  = mx.sym.LogisticRegressionOutput(data=fc2, label=outLabl, name='linreg1')

    return net

#
# Load data
#
xTrain = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
yTrain = np.array([0, 1, 1, 0])


#
# Setup iterators
#
trainIter = mx.io.NDArrayIter(data = xTrain, label = yTrain, batch_size = 1)
valIter   = mx.io.NDArrayIter(data = xTrain, label = yTrain, batch_size = 1)


#for batch in trainIter:
#    print batch.data[0].asnumpy()
#    print batch.label[0].asnumpy()
#    print " Batch done "





#pdb.set_trace()

#
# Multidevice kvstore setup and logging
#
kv = mx.kvstore.create('local')
head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
#logging.basicConfig(level=logging.DEBUG, filename="XOR_trainer.log")
logging.basicConfig(level=logging.DEBUG)

#
# Monitor training
#
mon = mx.mon.Monitor(
    1,                 # Print every 100 batches
    stat_func=None,      # The statistics function defined above
    pattern='.*weight|.*bias',  # A regular expression. Only arrays with name matching this pattern will be included.
    sort=True)           # Sort output by name



#
# Get model and train
#

input_size = (1,1,1,2)
net = get_xor_net()
print "===PRINT NETOWRK STATS ==="
printStats(net, input_size)

model = mx.model.FeedForward(
        ctx                = mx.gpu(),
        symbol             = net,
        num_epoch          = 200,
        learning_rate      = 0.2,
        momentum           = 0.9,
        wd                 = 0.00001,
        initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34),
        )



model.fit(X=trainIter, 
          eval_data=valIter, 
          #monitor=mon,
          batch_end_callback=None, 
          epoch_end_callback=None, 
          eval_metric='rmse'
          )


#
# Perform prediction
#
predictions = model.predict(valIter)
print predictions


