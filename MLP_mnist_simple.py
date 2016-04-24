import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
import logging
import pdb as pdb

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


###
## Network Definition
###

def getLenet():
    # Variables are place holders for input arrays. We give each variable a unique name.
    data = mx.symbol.Variable('data')
    
    # The input is fed to a fully connected layer that computes Y=WX+b.
    # This is the main computation module in the network.
    # Each layer also needs an unique name. We'll talk more about naming in the next section.
    fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
    # Activation layers apply a non-linear function on the previous layer's output.
    # Here we use Rectified Linear Unit (ReLU) that computes Y = max(X, 0).
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    
    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
    
    fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
    # Finally we have a loss layer that compares the network's output with label and generates gradient signals.
    mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
    return mlp


###
## Visulaise network
###
mlp = getLenet();
net = mlp
v = mx.viz.plot_network(net, shape={"data":(1, 1, 28, 28)})
v.render("LeNet simple")

print mlp.list_arguments()


###
## Load MNIST data
###
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
np.random.seed(1234) # set seed for deterministic ordering
p = np.random.permutation(mnist.data.shape[0])
X = mnist.data[p]
Y = mnist.target[p]

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(X[i].reshape((28,28)), cmap='Greys_r')
    plt.axis('off')
plt.show()

X = X.astype(np.float32)/255
Images_train = X[:60000 ]
Labels_train = Y[:60000 ]
Images_test  = X[ 60000:]
Labels_test  = Y[ 60000:]


###
## Create data iterator NDArrayIter for loaded MNIST data
###
batch_size = 100
train_iter = mx.io.NDArrayIter(Images_train, Labels_train, batch_size=batch_size)
test_iter  = mx.io.NDArrayIter(Images_test , Labels_test , batch_size=batch_size)

pdb.set_trace()

###
## Start training
###
def norm_stat(d):
    """The statistics you want to see.
    We compute the L2 norm here but you can change it to anything you like."""
    return mx.nd.norm(d)/np.sqrt(d.size)
mon = mx.mon.Monitor(
    100,                 # Print every 100 batches
    norm_stat,           # The statistics function defined above
    pattern='.*weight',  # A regular expression. Only arrays with name matching this pattern will be included.
    sort=True)           # Sort output by name

model = mx.model.FeedForward(
    ctx = mx.gpu(0),      # Run on GPU 0
    symbol = mlp,         # Use the network we just defined
    num_epoch = 10,       # Train for 10 epochs
    learning_rate = 0.1,  # Learning rate
    momentum = 0.9,       # Momentum for SGD with momentum
    wd = 0.00001)         # Weight decay for regularization
model.fit(
    X=train_iter,  # Training data set
    eval_data=test_iter,  # Testing data set. MXNet computes scores on test set every epoch
    monitor=mon,
    batch_end_callback = mx.callback.Speedometer(batch_size, 200))  # Logging module to print out progress


###
## Evaluate model
###
plt.imshow((Images_test[0].reshape((28,28))*255).astype(np.uint8), cmap='Greys_r')
plt.show()
print 'Result:', model.predict(Images_test[0:1])[0].argmax()

# Accuracy on entire testset
print 'Accuracy:', model.score(test_iter)*100, '%'
pdb.set_trace()