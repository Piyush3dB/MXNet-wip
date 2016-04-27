import mxnet as mx
from load_data import load
from sklearn.cross_validation import train_test_split
import logging


def get_lenet():
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
    fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=30)
    # loss
    lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return lenet
def get_mlp():
    """
    multi-layer perceptron
    """
    data = mx.symbol.Variable('data')
    fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=100)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    '''
    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden =100 )
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
    '''
    fc2  = mx.symbol.FullyConnected(data = act1, name='fc2', num_hidden=30)
    mlp  = mx.symbol.SoftmaxOutput(data = fc2, name = 'softmax')
    return mlp
#load data
X, y = load()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 42)
train = mx.io.NDArrayIter(data = X_train, label = y_train, batch_size = 128)
val = mx.io.NDArrayIter(data = X_test, label = y_test, batch_size = 128)
kv = mx.kvstore.create('local')
head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)
#model

net = get_lenet()
model = mx.model.FeedForward(
        ctx                = mx.cpu(),
        symbol             = net,
        num_epoch          = 50,
        learning_rate      = 0.01,
        momentum           = 0.9,
        wd                 = 0.00001,
        initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34),
        )
model.fit(
        X                  = train,
        eval_data          = val,
        batch_end_callback = mx.callback.Speedometer(128, 50),
        epoch_end_callback = None)
