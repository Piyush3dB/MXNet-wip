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

def print_inferred_shape(net, b, c, w, h):
    ar, ou, au = net.infer_shape(data=(b, c, w, h))
    print ou



def get_lenet():
    """
    LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick
    Haffner. "Gradient-based learning applied to document recognition."
    Proceedings of the IEEE (1998)
    """
    data = mx.symbol.Variable('data')

    print_inferred_shape(data, 1, 1, 28, 28)

    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                              kernel=(2,2), stride=(2,2))

    print_inferred_shape(pool1, 1, 1, 28, 28)

    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
    tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
    pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # first fullc
    flatten = mx.symbol.Flatten(data=pool2)
    fc1     = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3   = mx.symbol.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2  = mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
    # loss
    lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')

    group = mx.symbol.Group([data, conv1, tanh1, pool1, conv2, tanh2, pool2, flatten, fc1, tanh3, fc2, lenet ])
    #print group.list_outputs()

    return lenet, group







def getMLP():
    # Variables are place holders for input arrays. We give each variable a unique name.
    data = mx.symbol.Variable('data')
    
    #print_inferred_shape(data, 1, 1, 28, 28)

    # The input is fed to a fully connected layer that computes Y=WX+b.
    # This is the main computation module in the network.
    # Each layer also needs an unique name. We'll talk more about naming in the next section.
    fc1  = mx.symbol.FullyConnected(data = data,  name='fc1', num_hidden=128)
    # Activation layers apply a non-linear function on the previous layer's output.
    # Here we use Rectified Linear Unit (ReLU) that computes Y = max(X, 0).
    act1 = mx.symbol.Activation(    data = fc1,  name='relu1', act_type="relu")
    
    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(    data = fc2,  name='relu2', act_type="relu")
    
    fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
    # Finally we have a loss layer that compares the network's output with label and generates gradient signals.
    mlp  = mx.symbol.SoftmaxOutput( data = fc3,  name = 'softmax')
    return mlp


lenet, grp = get_lenet()

###
## Visulaise network
###
#mlp = getMLP();
#net = mlp
#v = mx.viz.plot_network(net, shape={"data":(1, 1, 28, 28)})
#v.render("LeNet simple")

nm = lenet.list_arguments()
ar, ou, au = lenet.infer_shape(data=(2, 1, 28, 28))

print "Layer Params... "
for i in range(0,len(ar)):
    print '%25s -> %18s' % (nm[i], str(ar[i]))



outs = grp.list_outputs()
out = grp

input_size = (2,1,28,28)

out_name = out.list_outputs()
arg_shapes, output_shapes, aux_shapes = out.infer_shape(data=input_size)
print "Layer Outputs... "
for i in xrange(len(out_name)):
    print '%25s -> %20s' % (out_name[i], output_shapes[i])


#pdb.set_trace()

