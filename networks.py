
### Lightened CNN


def group(data, num_r, num, kernel, stride, pad, layer):
    if num_r > 0:
        conv_r = mx.symbol.Convolution(data=data, num_filter=num_r, kernel=(1,1), name=('conv%s_r' % layer))
        slice_r = mx.symbol.SliceChannel(data=conv_r, num_outputs=2, name=('slice%s_r' % layer))
        mfm_r = mx.symbol.maximum(slice_r[0], slice_r[1])
        conv = mx.symbol.Convolution(data=mfm_r, kernel=kernel, stride=stride, pad=pad, num_filter=num, name=('conv%s' % layer))
    else:
        conv = mx.symbol.Convolution(data=data, kernel=kernel, stride=stride, pad=pad, num_filter=num, name=('conv%s' % layer))
    slice = mx.symbol.SliceChannel(data=conv, num_outputs=2, name=('slice%s' % layer))
    mfm = mx.symbol.maximum(slice[0], slice[1])
    pool = mx.symbol.Pooling(data=mfm, pool_type="max", kernel=(2, 2), stride=(2,2), name=('pool%s' % layer))
    return pool

def lightened_cnn_a_feature():
    data = mx.symbol.Variable(name="data")
    pool1 = group(data, 0, 96, (9,9), (1,1), (0,0), str(1))
    pool2 = group(pool1, 0, 192, (5,5), (1,1), (0,0), str(2))
    pool3 = group(pool2, 0, 256, (5,5), (1,1), (0,0), str(3))
    pool4 = group(pool3, 0, 384, (4,4), (1,1), (0,0), str(4))
    flatten = mx.symbol.Flatten(data=pool4)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=512, name="fc1")
    slice_fc1 = mx.symbol.SliceChannel(data=fc1, num_outputs=2, name="slice_fc1")
    mfm_fc1 = mx.symbol.maximum(slice_fc1[0], slice_fc1[1])
    drop1 = mx.symbol.Dropout(data=mfm_fc1, p=0.7, name="drop1")
    return drop1

def lightened_cnn_a(num_classes=10575):
     drop1 = lightened_cnn_a_feature()
     fc2 = mx.symbol.FullyConnected(data=drop1, num_hidden=num_classes, name="fc2")
     softmax = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
     
     group = mx.symbol.Group([drop1, fc2, softmax])

     return softmax, group

def lightened_cnn_b_feature():
     data = mx.symbol.Variable(name="data")
     pool1 = group(data, 0, 96, (5,5), (1,1), (2,2), str(1))
     pool2 = group(pool1, 96, 192, (3,3), (1,1), (1,1), str(2))
     pool3 = group(pool2, 192, 384, (3,3), (1,1), (1,1), str(3))
     pool4 = group(pool3, 384, 256, (3,3), (1,1), (1,1), str(4))
     pool5 = group(pool4, 256, 256, (3,3), (1,1), (1,1), str(5))
     flatten = mx.symbol.Flatten(data=pool5)
     fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=512, name="fc1")
     slice_fc1 = mx.symbol.SliceChannel(data=fc1, num_outputs=2, name="slice_fc1")
     mfm_fc1 = mx.symbol.maximum(slice_fc1[0], slice_fc1[1])
     drop1 = mx.symbol.Dropout(data=mfm_fc1, p=0.7, name="drop1")
     return drop1

def lightened_cnn_b(num_classes=10575):
     drop1 = lightened_cnn_b_feature()
     fc2 = mx.symbol.FullyConnected(data=drop1, num_hidden=num_classes, name="fc2")
     softmax = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
     
     group = mx.symbol.Group([drop1, fc2, softmax])
     
     return softmax, group


###


def get_mlp_like_convnet():
    """
    multi-layer perceptron
    """
    data = mx.symbol.Variable('data')

    #fc1  = mx.symbol.FullyConnected(data=data, name='fc1',   num_hidden=128 )
    conv1 = mx.symbol.Convolution(data=data ,  kernel=(28,28), num_filter=128)
    act1  = mx.symbol.Activation( data=conv1,  name='relu1'  , act_type="relu")

    #fc2  = mx.symbol.FullyConnected(data=act1, name='fc2',   num_hidden = 64)
    conv2 = mx.symbol.Convolution(data=act1 ,  kernel=(1,1), num_filter=64)
    act2  = mx.symbol.Activation( data=conv2  ,  name='relu2', act_type="relu")

    #fc3  = mx.symbol.FullyConnected(data=flatten, name='fc3',   num_hidden=10  )
    conv3 = mx.symbol.Convolution(data=act2 ,  kernel=(1,1), num_filter=10)
    
    flatten = mx.symbol.Flatten(   data=conv3)
    
    mlp  = mx.symbol.SoftmaxOutput(data=flatten,  name='softmax'               )
    
    group = mx.symbol.Group([data, conv1, act1, conv2, act2, conv3, flatten, mlp])


    return mlp, group





def get_symbol_alexnet(num_classes = 1000):
    input_data = mx.symbol.Variable(name="data")
    # stage 1
    conv1 = mx.symbol.Convolution(
        data=input_data, kernel=(11, 11), stride=(4, 4), num_filter=96)
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    pool1 = mx.symbol.Pooling(
        data=relu1, pool_type="max", kernel=(3, 3), stride=(2,2))
    lrn1 = mx.symbol.LRN(data=pool1, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
    # stage 2
    conv2 = mx.symbol.Convolution(
        data=lrn1, kernel=(5, 5), pad=(2, 2), num_filter=256)
    relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, kernel=(3, 3), stride=(2, 2), pool_type="max")
    lrn2 = mx.symbol.LRN(data=pool2, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
    # stage 3
    conv3 = mx.symbol.Convolution(
        data=lrn2, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    conv4 = mx.symbol.Convolution(
        data=relu3, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
    conv5 = mx.symbol.Convolution(
        data=relu4, kernel=(3, 3), pad=(1, 1), num_filter=256)
    relu5 = mx.symbol.Activation(data=conv5, act_type="relu")
    pool3 = mx.symbol.Pooling(data=relu5, kernel=(3, 3), stride=(2, 2), pool_type="max")
    # stage 4
    flatten = mx.symbol.Flatten(data=pool3)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096)
    relu6 = mx.symbol.Activation(data=fc1, act_type="relu")
    dropout1 = mx.symbol.Dropout(data=relu6, p=0.5)
    # stage 5
    fc2 = mx.symbol.FullyConnected(data=dropout1, num_hidden=4096)
    relu7 = mx.symbol.Activation(data=fc2, act_type="relu")
    dropout2 = mx.symbol.Dropout(data=relu7, p=0.5)
    # stage 6
    fc3 = mx.symbol.FullyConnected(data=dropout2, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')

    group = mx.symbol.Group([input_data, conv1, relu1, pool1, lrn1, conv2, relu2, pool2, lrn2, conv3, relu3, conv4, relu4, conv5, relu5, pool3, flatten, fc1, relu6, dropout1, fc2, relu7, dropout2, fc3, softmax])


    return softmax, group




def FireModelFactory(data, size):
    if size == 1:
        n_s1x1 = 16
        n_e1x1 = 64
        n_e3x3 = 64
    elif size == 2:
        n_s1x1 = 32
        n_e1x1 = 128
        n_e3x3 = 128
    elif size == 3:
        n_s1x1 = 48
        n_e1x1 = 192
        n_e3x3 = 192
    elif size == 4:
        n_s1x1 = 64
        n_e1x1 = 256
        n_e3x3 = 256

    squeeze1x1 = mx.symbol.Convolution(
            data   = data, 
            kernel = (1,1), 
            pad    = (0,0),
            num_filter = n_s1x1 )

    relu_squeeze1x1 = mx.symbol.Activation( data=squeeze1x1, act_type="relu" )

    expand1x1 = mx.symbol.Convolution(
            data   = relu_squeeze1x1,
            kernel = (1,1),
            pad    = (0,0),
            num_filter = n_e1x1 )

    relu_expand1x1 = mx.symbol.Activation(data=expand1x1, act_type="relu" )

    expand3x3 = mx.symbol.Convolution(
            data   = relu_squeeze1x1,
            kernel = (3,3),
            pad    = (1,1),
            num_filter = n_e3x3 )

    relu_expand3x3 = mx.symbol.Activation(data=expand3x3, act_type="relu" )
    
    concat = mx.symbol.Concat( *[relu_expand1x1, relu_expand3x3] )

    return concat 

def get_symbol_squeeze(num_classes = 1000):
    data = mx.symbol.Variable(name="data")
    #pdb.set_trace()

    conv1 = mx.symbol.Convolution(data=data, kernel=(7,7), stride=(2,2), num_filter=96, pad=(2,2))
    # buggyconv1 = mx.symbol.Convolution(data=data, kernel=(7,7), stride=(2,2), num_filter=96)
    relu_conv1 = mx.symbol.Activation(data=conv1, act_type="relu")
    maxpool1 = mx.symbol.Pooling(data=relu_conv1, kernel=(3,3), stride=(2,2), pool_type="max")

    fire2 = FireModelFactory(data=maxpool1, size=1)
    fire3 = FireModelFactory(data=fire2, size=1)
    fire4 = FireModelFactory(data=fire3, size=2)

    maxpool4 = mx.symbol.Pooling(data=fire4, kernel=(3,3), stride=(2,2), pool_type="max")

    fire5 = FireModelFactory(data=maxpool4, size=2)
    fire6 = FireModelFactory(data=fire5, size=3)
    fire7 = FireModelFactory(data=fire6, size=3)
    fire8 = FireModelFactory(data=fire7, size=4)

    maxpool8 = mx.symbol.Pooling(data=fire8, kernel=(3,3), stride=(2,2), pool_type="max")

    fire9 = FireModelFactory(data=maxpool8, size=4)
    dropout_fire9 = mx.symbol.Dropout(data=fire9, p=0.5)

    conv10 = mx.symbol.Convolution(data=dropout_fire9, kernel=(1,1), num_filter=num_classes)
    # buggyconv10 = mx.symbol.Convolution(data=dropout_fire9, kernel=(1,1), pad=(1,1), num_filter=1000)
    relu_conv10 = mx.symbol.Activation(data=conv10, act_type="relu")
    avgpool10 = mx.symbol.Pooling(data=relu_conv10, kernel=(13,13), stride=(1,1), pool_type="avg")

    flatten = mx.symbol.Flatten(data=avgpool10, name='flatten')

    softmax = mx.symbol.SoftmaxOutput(data=flatten, name="softmax")
    
    group = mx.symbol.Group([data, conv1, relu_conv1, maxpool1, fire2, fire3, fire4, maxpool4, fire5, fire6, fire7, fire8, maxpool8,  fire9, dropout_fire9, conv10, relu_conv10, avgpool10, flatten, softmax ])

    return softmax, group





def get_symbol_vgg(num_classes = 1000):
    ## define alexnet
    data = mx.symbol.Variable(name="data")
    # group 1
    conv1_1 = mx.symbol.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    pool1   = mx.symbol.Pooling(
              data=relu1_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
            data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    pool2 = mx.symbol.Pooling(
            data=relu2_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
            data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
            data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    pool3 = mx.symbol.Pooling(
            data=relu3_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
            data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1")
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
            data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2")
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    pool4 = mx.symbol.Pooling(
            data=relu4_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
            data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1")
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(
            data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2")
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="conv1_2")
    pool5 = mx.symbol.Pooling(
            data=relu5_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool5")
    # group 6
    flatten = mx.symbol.Flatten(data=pool5, name="flatten")
    fc6     = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7   = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # output
    fc8     = mx.symbol.FullyConnected(data=drop7, num_hidden=num_classes, name="fc8")
    softmax = mx.symbol.SoftmaxOutput(data=fc8, name='softmax')


    group = mx.symbol.Group([data, conv1_1, relu1_1, pool1, conv2_1, relu2_1, pool2, conv3_1, relu3_1, conv3_2, relu3_2, pool3, conv4_1, relu4_1, conv4_2, relu4_2, pool4, conv5_1, relu5_1, conv5_2, relu5_2, pool5, flatten, fc6, relu6, drop6, fc7, relu7, drop7, fc8, softmax])
    return softmax, group






def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0), name=None, suffix=''):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, name='conv_%s%s' %(name, suffix))
    bn = mx.symbol.BatchNorm(data=conv, name='bn_%s%s' %(name, suffix))
    act = mx.symbol.Activation(data=bn, act_type='relu', name='relu_%s%s' %(name, suffix))
    return act

def InceptionFactoryA(data, num_1x1, num_3x3red, num_3x3, num_d3x3red, num_d3x3, pool, proj, name):
    # 1x1
    c1x1 = ConvFactory(data=data, num_filter=num_1x1, kernel=(1, 1), name=('%s_1x1' % name))
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), name=('%s_3x3' % name), suffix='_reduce')
    c3x3 = ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1), name=('%s_3x3' % name))
    # double 3x3 reduce + double 3x3
    cd3x3r = ConvFactory(data=data, num_filter=num_d3x3red, kernel=(1, 1), name=('%s_double_3x3' % name), suffix='_reduce')
    cd3x3 = ConvFactory(data=cd3x3r, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), name=('%s_double_3x3_0' % name))
    cd3x3 = ConvFactory(data=cd3x3, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), name=('%s_double_3x3_1' % name))
    # pool + proj
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=('%s_pool_%s_pool' % (pool, name)))
    cproj = ConvFactory(data=pooling, num_filter=proj, kernel=(1, 1), name=('%s_proj' %  name))
    # concat
    concat = mx.symbol.Concat(*[c1x1, c3x3, cd3x3, cproj], name='ch_concat_%s_chconcat' % name)
    return concat

def InceptionFactoryB(data, num_3x3red, num_3x3, num_d3x3red, num_d3x3, name):
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), name=('%s_3x3' % name), suffix='_reduce')
    c3x3 = ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_3x3' % name))
    # double 3x3 reduce + double 3x3
    cd3x3r = ConvFactory(data=data, num_filter=num_d3x3red, kernel=(1, 1),  name=('%s_double_3x3' % name), suffix='_reduce')
    cd3x3 = ConvFactory(data=cd3x3r, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name=('%s_double_3x3_0' % name))
    cd3x3 = ConvFactory(data=cd3x3, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_double_3x3_1' % name))
    # pool + proj
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pool_type="max", name=('max_pool_%s_pool' % name))
    # concat
    concat = mx.symbol.Concat(*[c3x3, cd3x3, pooling], name='ch_concat_%s_chconcat' % name)
    return concat

def get_inception_bn_symbol(num_classes=1000):
    # data
    data = mx.symbol.Variable(name="data")
    # stage 1
    conv1 = ConvFactory(data=data, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3), name='conv1')
    pool1 = mx.symbol.Pooling(data=conv1, kernel=(3, 3), stride=(2, 2), name='pool1', pool_type='max')
    # stage 2
    conv2red = ConvFactory(data=pool1, num_filter=64, kernel=(1, 1), stride=(1, 1), name='conv2red')
    conv2    = ConvFactory(data=conv2red, num_filter=192, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name='conv2')
    pool2    = mx.symbol.Pooling(data=conv2, kernel=(3, 3), stride=(2, 2), name='pool2', pool_type='max')
    # stage 2
    in3a = InceptionFactoryA(pool2, 64, 64, 64, 64, 96, "avg", 32, '3a')
    in3b = InceptionFactoryA(in3a, 64, 64, 96, 64, 96, "avg", 64, '3b')
    in3c = InceptionFactoryB(in3b, 128, 160, 64, 96, '3c')
    # stage 3
    in4a = InceptionFactoryA(in3c, 224, 64, 96, 96, 128, "avg", 128, '4a')
    in4b = InceptionFactoryA(in4a, 192, 96, 128, 96, 128, "avg", 128, '4b')
    in4c = InceptionFactoryA(in4b, 160, 128, 160, 128, 160, "avg", 128, '4c')
    in4d = InceptionFactoryA(in4c, 96, 128, 192, 160, 192, "avg", 128, '4d')
    in4e = InceptionFactoryB(in4d, 128, 192, 192, 256, '4e')
    # stage 4
    in5a = InceptionFactoryA(in4e, 352, 192, 320, 160, 224, "avg", 128, '5a')
    in5b = InceptionFactoryA(in5a, 352, 192, 320, 192, 224, "max", 128, '5b')
    # global avg pooling
    avg = mx.symbol.Pooling(data=in5b, kernel=(7, 7), stride=(1, 1), name="global_pool", pool_type='avg')
    # linear classifier
    flatten = mx.symbol.Flatten(data=avg, name='flatten')
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes, name='fc1')
    softmax = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')

    group = mx.symbol.Group([data, conv1, pool1,  conv2red, conv2, pool2, in3a, in3b, in3c, in4a, in4b, in4c, in4d, in4e, in5a, in5b, avg, flatten, fc1, softmax ])

    return softmax, group

import find_mxnet
import mxnet as mx
import re
import pdb as pdb

"""
Collection of network definitions
"""

def get_lenet_no_pooling2():
    """
    Lenet with the Conv layer kernel resized so that no pooling layers are required.
    """
    data = mx.symbol.Variable('data')


    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20, stride=(2,2), pad=(2,2))
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    #pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
    pool1 = tanh1

    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50, stride=(2,2), pad=(1,1))
    tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
    #pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
    pool2 = tanh2
    
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


def get_lenet_no_pooling():
    """
    Lenet with the Conv layer kernel resized so that no pooling layers are required.
    """
    data = mx.symbol.Variable('data')


    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(6,6), num_filter=20, stride=(2,2))
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    #pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
    pool1 = tanh1

    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(6,6), num_filter=50, stride=(2,2))
    tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
    #pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
    pool2 = tanh2

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


def get_lenet():
    """
    LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick
    Haffner. "Gradient-based learning applied to document recognition."
    Proceedings of the IEEE (1998)
    """
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

    group = mx.symbol.Group([data, fc1, act1, fc2, act2, fc3, mlp])

    return mlp, group

