import mxnet as mx

# define computation graphs
A = mx.symbol.Variable('A')
B = mx.symbol.Variable('B')
C = A + B
a = mx.nd.ones(3) * 4
b = mx.nd.ones(3) * 2
# bind the symbol with real arguments
c_exec = C.bind(ctx=mx.gpu(), args={'A' : a, 'B': b})
# do forward pass calclation.
c_exec.forward()
print c_exec.outputs[0].asnumpy()


#For neural nets, a more commonly used pattern is simple_bind, which will create all the argument arrays for you. Then you can call forward, and backward (if gradient is needed) to get the gradient.



