import find_mxnet
import mxnet as mx
import numpy as np
import logging
import pdb as pdb
import matplotlib.pyplot as plt



trX = mx.io.CSVIter(data_csv="/home/piyush/Downloads/GitHub/mxnet/example/kaggle-ndsb2/train-64x64-data.csv", data_shape=(30, 64, 64),
                    label_csv="/home/piyush/Downloads/GitHub/mxnet/example/kaggle-ndsb2/train-systole.csv", label_shape=(600,),
                    batch_size=1)

for bt in trX:
    pdb.set_trace()