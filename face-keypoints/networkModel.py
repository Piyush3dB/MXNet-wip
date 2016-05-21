import find_mxnet
import mxnet as mx
from load_data import load
from sklearn.cross_validation import train_test_split
import logging
import pdb as pdb
import numpy as np

# Set for repeatability
mx.random.seed(1)

def get_lenet():

    outLabl = mx.sym.Variable('softmax_label')


    data = mx.symbol.Variable('data')
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
    tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
    pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # first fullc
    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=1)
    # loss
    net  = mx.sym.LinearRegressionOutput(data=fc2, label=outLabl, name='linreg1')
    
    return net


def get_mlp():
    """
    multi-layer perceptron
    """

    outLabl = mx.sym.Variable('softmax_label')
    
    data = mx.symbol.Variable('data')
    fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=100)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    fc2  = mx.symbol.FullyConnected(data = act1, name='fc2', num_hidden=1) # set hidden to 1 since np.shape(y)=(1,)
    net  = mx.sym.LinearRegressionOutput(data=fc2, label=outLabl, name='linreg1')

    return net

#
# Load data
#
X, y = load()
# np.shape(y)=(1,).  
# TODO np.shape(y) should be (30,) representing 15 key points encoded as as 2 values per point


#
# Setup iterators
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 42)
trainIter = mx.io.NDArrayIter(data = X_train, label = y_train, batch_size = 1)
valIter   = mx.io.NDArrayIter(data = X_test , label = y_test , batch_size = 1)


#
# Multidevice kvstore setup and logging
#
kv = mx.kvstore.create('local')
head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)


#
# Get model and train
#
#net = get_lenet()
net = get_mlp()
model = mx.model.FeedForward(
        ctx                = mx.gpu(),
        symbol             = net,
        num_epoch          = 5,
        learning_rate      = 0.01,
        momentum           = 0.9,
        wd                 = 0.00001,
        initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34),
        )
model.fit(X=trainIter, eval_data=valIter, batch_end_callback=mx.callback.Speedometer(1,50), epoch_end_callback=None, eval_metric='rmse')
