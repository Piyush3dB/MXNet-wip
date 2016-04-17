import numpy as np
import skimage.draw as draw
import skimage.io as io
import mxnet as mx
import logging
import pdb as pdb

# create logger with 'spam_application'
#logger = logging.getLogger('spam_application')
#logger.setLevel(logging.DEBUG)
# create console handler with a higher log level
#ch = logging.StreamHandler()
#ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#ch.setFormatter(formatter)
# add the handlers to the logger
#logger.addHandler(ch)


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


# global parameters
N = 100000
Reso = 28


def drawShape(f, *args):
    img = np.zeros((Reso, Reso), dtype=np.uint8)
    rr, cc = f(*args)
    img[rr, cc] = 1
    return img


def drawRandomCircle():
    r = np.random.randint(10, int(Reso*0.4))
    x = np.random.randint(r-1, Reso-r-1)
    y = np.random.randint(r-1, Reso-r-1)
    return drawShape(draw.circle, x, y, r)

def drawRandomRectange():
    a = np.random.randint(20, int(Reso*0.8))
    #left top
    x = np.random.randint(0, (Reso-a-1))
    y = np.random.randint(0, (Reso-a-1))
    vertices_yaxis = np.array([y, (y+a), (y+a), y])
    vertices_xaxis = np.array([x, x,     (x+a), (x+a)])
    return drawShape(draw.polygon, vertices_yaxis, vertices_xaxis)


X = np.zeros((N, 1, Reso, Reso), dtype=np.uint8)
for i in range(N):
    if i < N/2:
        X[i, 0, :, :] = drawRandomCircle()
    else:
        X[i, 0, :, :] = drawRandomRectange()
y = np.ones(N)
y[:N/2] = 0


shuffle_idx = np.arange(N)
np.random.shuffle(shuffle_idx)
X = X[shuffle_idx, :, :, :]
y = y[shuffle_idx]


trainN = int(N*0.6)
trainIter = mx.io.NDArrayIter(X[:trainN, :, :, :], label=y[:trainN], batch_size=32, shuffle=False, last_batch_handle='pad')
valIter = mx.io.NDArrayIter(X[trainN:, :, :, :], label=y[trainN:], batch_size=32, shuffle=False, last_batch_handle='pad')


def get_lenet():
    """
    LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick
    Haffner. "Gradient-based learning applied to document recognition."
    Proceedings of the IEEE (1998)
    """
    data = mx.symbol.Variable('data')
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    pool1 = mx.symbol.Pooling(data=relu1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
    relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # first fullc
    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    relu3 = mx.symbol.Activation(data=fc1, act_type="relu")
    # second fullc
    fc2 = mx.symbol.FullyConnected(data=relu3, num_hidden=100)
    # loss
    lenet = mx.symbol.SoftmaxOutput(fc2, name='softmax')
    return lenet


cnn = get_lenet()


print "Creating model..."
model = mx.model.FeedForward(
        ctx = mx.gpu(), 
        symbol = cnn, 
        num_epoch = 3,
        learning_rate = 0.01, 
        momentum = 0.0, 
        wd = 0.00001)   

batch_size = 32
print "Fitting..."
model.fit(X=trainIter, eval_data=valIter, batch_end_callback = mx.callback.Speedometer(batch_size, 200))

print "Predicting..."
predY = model.predict(valIter)
print predY

