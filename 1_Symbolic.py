import mxnet as mx

## Create 2 layer perceptron network
net = mx.symbol.Variable('data')
net = mx.symbol.FullyConnected(data=net, name='fc1',   num_hidden=128)
net = mx.symbol.Activation(    data=net, name='relu1', act_type="relu")
net = mx.symbol.FullyConnected(data=net, name='fc2',   num_hidden=64)
net = mx.symbol.SoftmaxOutput( data=net, name='out')
print type(net)

print net.list_arguments()

print "Infer argument shapes"
#net = mx.symbol.Variable('data')
#net = mx.symbol.FullyConnected(data=net, name='fc1', num_hidden=10)
arg_shape, out_shape, aux_shape = net.infer_shape(data=(100,1))
print dict(zip(net.list_arguments(), arg_shape))

print out_shape