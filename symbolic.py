import mxnet as mx
import numpy as np
import pdb as pdb


PROC = mx.gpu()

np.random.seed(0)

## Define the network
a = mx.sym.Variable("A") # represent a placeholder. These can be inputs, weights, or anything else.
b = mx.sym.Variable("B")
c = (a + b) / 10
net = c + 1

c.save('net.json')

# inspect the graph's inputs and outputs
net.list_arguments()
# ['A', 'B']

net.list_outputs()
# ['_plusscalar0_output'] - Default name from adding to scalar.

# The graph allows shape inference.

# define input shapes
inp_shapes = {'A':(10,), 'B':(10,)}
arg_shapes, out_shapes, aux_shapes = net.infer_shape(**inp_shapes)

arg_shapes # the shapes of all the inputs to the graph. Order matches net.list_arguments()
# [(10, ), (10, )]

out_shapes # the shapes of all outputs. Order matches net.list_outputs()
# [(10, )]

aux_shapes # the shapes of auxiliary variables. These are variables that are not trainable such as batch normalization population statistics. For now, they are save to ignore.
# []

## Allocate space for inputs
input_arguments = {}
input_arguments['A'] = mx.nd.ones((10, ), ctx=PROC)
input_arguments['B'] = mx.nd.ones((10, ), ctx=PROC)

## Allocate space for gradients to be calculated
grad_arguments = {}
grad_arguments['A'] = mx.nd.ones((10, ), ctx=PROC)
grad_arguments['B'] = mx.nd.ones((10, ), ctx=PROC)

executor = net.bind(ctx=PROC,
                    args=input_arguments, # this can be a list or a dictionary mapping names of inputs to NDArray
                    args_grad=grad_arguments, # this can be a list or a dictionary mapping names of inputs to NDArray
                    grad_req='write') # instead of null, tell the executor to write gradients. This replaces the contents of grad_arguments with the gradients computed.
# The executor
executor.arg_dict
# {'A': NDArray, 'B': NDArray}

#pdb.set_trace()

executor.arg_dict['A'][:] = np.random.rand(10,)
executor.arg_dict['B'][:] = np.random.rand(10,)

executor.forward()
outputValue = executor.outputs[0].asnumpy()
print outputValue

out_grad = mx.nd.ones((10,), ctx=PROC)
executor.backward([out_grad]) # because the graph only has one output, only one output grad is needed.

gradsA = executor.grad_arrays[0].asnumpy()
gradsB = executor.grad_arrays[1].asnumpy()

print "Gradients for A"
print gradsA

print "Gradients for B"
print gradsB

pdb.set_trace()