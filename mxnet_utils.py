import find_mxnet
import mxnet as mx
import re
import pdb as pdb


mltp = lambda x, y: x*y

def remL(s): return eval(re.sub(r"L", "", str(s)))

def printStats(net, input_size):

    group = net.get_internals()

    # Get list of arg and output names
    # Infer arg and output shapes
    arg_names    = group.list_arguments()
    output_names = group.list_outputs()
    arg_shapes, output_shapes, aux_shapes = group.infer_shape(data=input_size)
    arg_shapes = map(remL, arg_shapes)
    output_shapes = map(remL, output_shapes)

    mltp = lambda x, y: x*y

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
