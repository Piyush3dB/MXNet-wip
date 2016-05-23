import find_mxnet
import mxnet as mx
import re
import pdb as pdb
import json
from graphviz import Digraph
import copy

def _str2tuple(string):
    """convert shape string to list, internal use only

    Parameters
    ----------
    string: str
        shape string

    Returns
    -------
    list of str to represent shape
    """
    return re.findall(r"\d+", string)




mltp = lambda x, y: x*y

def remL(s): return eval(re.sub(r"L", "", str(s)))



def net2dot(symbol, shape=None, title="plot"):


    jsonNet = symbol.tojson()

    draw_shape = True

    group = symbol.get_internals()

    # Get list of arg and output names
    # Infer arg and output shapes
    arg_names    = group.list_arguments()
    output_names = group.list_outputs()
    arg_shapes, out_shapes, _ = group.infer_shape(**shape)
    arg_shapes = map(remL, arg_shapes)
    output_shapes = map(remL, out_shapes)




    nW = 0
    nF = 0

    # pdb.set_trace()
    sizeAry = []

    # Stats calculation
    for i in xrange(len(output_names)):
        name  = output_names[i]
        shap = output_shapes[i]
        size = str(reduce(mltp,shap))
        sizeAry.append(size)

        if ('weight' in name) or ('bias' in name):
            space = ' '
            nW += int(size)
        else:
            space = ' '*40
            nF += int(size)
        




    
    interals = symbol.get_internals()
    _, out_shapes, _ = interals.infer_shape(**shape)
    if out_shapes == None:
        raise ValueError("Input shape is incompete")
    shape_dict = dict(zip(interals.list_outputs(), out_shapes))




    conf = json.loads(symbol.tojson())
    nodes = conf["nodes"]
    heads = set([x[0] for x in conf["heads"]])  # TODO(xxx): check careful

    #pdb.set_trace()


    # default attributes of node
    node_attr = {"shape": "box", "fontname": "Courier New", #"fixedsize": "true", 
                 "width": "1.3", "height": "0.8034", "style": "filled", "fontsize": "12"}
    # merge the dcit provided by user and the default one

    node_attrs = {}
    node_attr.update(node_attrs)
    dot = Digraph(name=title)
    # color map
    cm = ("#8dd3c7", "#fb8072", "#ffffb3", "#bebada", "#80b1d3",
          "#fdb462", "#b3de69", "#fccde5")

    # make nodes
    for i in range(len(nodes)):
        node = nodes[i]
        op = node["op"]
        name = node["name"]
        param = node["param"]
        shape = (str(output_shapes[i]) + "=" + str(sizeAry[i])).replace(" ", "")

        # input data
        attr = copy.deepcopy(node_attr)
        label = op

        if op == "null":
            if i in heads:
                ## This is head or tail
                print "HEAD"
                pdb.set_trace()
                label = "%s\n%s" % (op, node["name"])
                attr["fillcolor"] = cm[5]
            else:
                label = "%s\n%s" % (node["name"], shape)
                attr["fillcolor"] = cm[0]
                #continue
        elif op == "Convolution":
            #pdb.set_trace()

            #label = "Convolution\n%s" % (str(param))
            
            label = "Convolution\n%sx%s/%s, %s" % (_str2tuple(node["param"]["kernel"])[0],
                                                   _str2tuple(node["param"]["kernel"])[1],
                                                   _str2tuple(node["param"]["stride"])[0],
                                                   node["param"]["num_filter"])
            attr["fillcolor"] = cm[1]
        elif op == "FullyConnected":
            label = "FullyConnected\n%s" % node["param"]["num_hidden"]
            attr["fillcolor"] = cm[1]
        elif op == "BatchNorm":
            attr["fillcolor"] = cm[3]
        elif op == "Activation" or op == "LeakyReLU":
            label = "%s\n%s" % (op, node["param"]["act_type"])
            attr["fillcolor"] = cm[2]
        elif op == "Pooling":
            label = "Pooling\n%s, %sx%s/%s" % (node["param"]["pool_type"],
                                               _str2tuple(node["param"]["kernel"])[0],
                                               _str2tuple(node["param"]["kernel"])[1],
                                               _str2tuple(node["param"]["stride"])[0])
            attr["fillcolor"] = cm[4]
        elif op == "Concat" or op == "Flatten" or op == "Reshape":
            attr["fillcolor"] = cm[5]
        elif op == "Softmax":
            attr["fillcolor"] = cm[6]
        else:
            attr["fillcolor"] = cm[7]
        
        #print 'name  ' + name
        #print 'label ' + label

        print str(i)

        #pdb.set_trace()

        dot.node(name=name, label=label, **attr)

    
#    return dot


    # add edges
    for i in range(len(nodes)):
        node = nodes[i]
        op = node["op"]
        name = node["name"]
        if op == "null":
            #continue
            print "null"
        else:
            inputs = node["inputs"]
            for item in inputs:
                input_node = nodes[item[0]]
                input_name = input_node["name"]
                if input_node["op"] != "null" or item[0] in heads:
                #if 1:
                    attr = {"dir": "back", 'arrowtail':'open'}
                    # add shapes
                    if draw_shape:
                        if input_node["op"] != "null":
                        #if 1:
                            key = input_name + "_output"
                            shape = shape_dict[key][1:]
                            label = "x".join([str(x) for x in shape])
                            attr["label"] = label
                        else:
                            key = input_name
                            shape = shape_dict[key][1:]
                            label = "x".join([str(x) for x in shape])
                            attr["label"] = label
                    dot.edge(tail_name=name, head_name=input_name, **attr)

    return dot





def printStats(net, input_size):

    group = net.get_internals()

    # Get list of arg and output names
    # Infer arg and output shapes
    arg_names    = group.list_arguments()
    output_names = group.list_outputs()
    arg_shapes, output_shapes, _ = group.infer_shape(data=input_size)
    arg_shapes = map(remL, arg_shapes)
    output_shapes = map(remL, output_shapes)


    nW = 0
    nF = 0

    # pdb.set_trace()
    sizeAry = []

    # Stats calculation
    for i in xrange(len(output_names)):
        name  = output_names[i]
        shape = output_shapes[i]
        size = str(reduce(mltp,shape))
        sizeAry.append(size)

        if ('weight' in name) or ('bias' in name):
            space = ' '
            nW += int(size)
        else:
            space = ' '*40
            nF += int(size)
        
    # Print
    print ' '
    print ' '
    print '      %25s -> %41s   %36s' % ('LAYER', 'WEIGHTS', 'ACTIVATION MAP')
    print '-'*115
    for i in xrange(len(output_names)):
        name  = output_names[i]
        shape = output_shapes[i]
        size = sizeAry[i]

        if ('weight' in name) or ('bias' in name):
            space = ' '
            pct = 100.*float(int(size))/float(int(nW))
        else:
            space = ' '*40
            pct = 100.*float(int(size))/float(int(nF))


        # Print line as is:
        print '[%3d] %25s -> %s %20s = %8s [%2.2f%%]' % (i, name, space, shape, size, pct)
        
        # Print in markdown format:
        #print '|%3d. | %25s -> | %s %20s = %8s [%2.1f%%] |' % (i, name, space, shape, size, pct)


    print ' '
    print 'Total weights    = %10d [%3.2f K] [%3.2f M]' % (nW, nW/1000., nW/1000000.)
    print 'Total activation = %10d [%3.2f K] [%3.2f M]' % (nF, nF/1000., nF/1000000.)
    print ' '
