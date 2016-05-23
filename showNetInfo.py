import find_mxnet
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
import logging
import re
import pdb as pdb
import json
from mxnet_utils import printStats, _str2tuple, net2dot
from networks import *

from visualization import *




"""
Script to load and print network information
"""

#
# Lightened CNN
#
#input_size = (1,1,128,128)
#net, _ = lightened_cnn_b()
#net, _ = lightened_cnn_a()


#
# LeNET and MLP Convnet Style
#
input_size = (1,1,28,28)
net, _ = get_lenet_no_pooling2()
#net, _ = get_lenet()
#net, _ = get_mlp_like_convnet()


#
# Vanilla MLP 
#
#input_size = (1,1,1,28*28)
#net, _ = getMLP()

#
# Bigger nets
#
#input_size = (1,3, 224, 224)
#net, _ = get_symbol_squeeze()
#net, _ = get_symbol_vgg()
#net, _ = get_symbol_alexnet()
#net, _ = get_inception_bn_symbol()


#
# MLP for face model
#
input_size = (1, 1, 96, 96)
#net = get_mlp_for_face()
net = get_lenet_for_face()

###
## Visulaise network
###
#mlp = getMLP();
#net = mlp

#v = mx.viz.plot_network(net, shape={"data":input_size})
v = net2dot(net, shape={"data":input_size})
v.render("NNet")
net.save('NNet.json')


print "===PRINT NETOWRK STATS ==="
printStats(net, input_size)


jsonNet = net.tojson()
print jsonNet

pdb.set_trace()

conf = json.loads(net.tojson())
nodes = conf["nodes"]
heads = set([x[0] for x in conf["heads"]])


print "===DONE==="

#
#data
#weight
#bias
#output
#label