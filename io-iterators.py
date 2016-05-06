import find_mxnet
import mxnet as mx
import sys
import numpy as np
import pdb as pdb
import math as math


def test_NDArrayIter():

    # Set Numper of Samples
    N = 1000
    
    # Set Samples per Batch
    BS = 10

    # Number of batches
    NB = math.ceil(1000/float(BS))

    datas  = np.arange(N)
    labels = np.arange(N)

    

    print ('Mean of population %f' % np.mean(datas))


    dataiter = mx.io.NDArrayIter(datas, labels, BS, True, last_batch_handle='pad')
    batchidx = 0
    for batch in dataiter:
        batchidx += 1
        mn = np.mean(batch.data[0].asnumpy())
        #dm = np.shape(batch.data[0].asnumpy())
        print ('Batch %d mean = %f' % (batchidx, mn))
        pdb.set_trace()
        #print np.shape(batch.data[0].asnumpy().T)
        #print batch.data[0].asnumpy().T


    assert(batchidx == NB)

    dataiter = mx.io.NDArrayIter(datas, labels, BS, False, last_batch_handle='pad')
    
    
    pdb.set_trace()


if __name__ == '__main__':
    sys.exit(test_NDArrayIter())
