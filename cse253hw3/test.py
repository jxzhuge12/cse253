import cPickle
import numpy as np
import caffe
import lmdb
import os

#=======================================================
#Using cPickle to transfer file to Dict
#From https://www.cs.toronto.edu/~kriz/cifar.html
#=======================================================
def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

#=======================================================
#Funciton defined for get the cifar-100 dataset
#=======================================================
def get_cifar100(folder):
    train_fname = os.path.join(folder,'train')
    test_fname  = os.path.join(folder,'test')
    data_dict = unpickle(train_fname)
    train_data = data_dict['data']
    train_fine_labels = data_dict['fine_labels']
    train_coarse_labels = data_dict['coarse_labels']

    data_dict = unpickle(test_fname)
    test_data = data_dict['data']
    test_fine_labels = data_dict['fine_labels']
    test_coarse_labels = data_dict['coarse_labels']

    bm = unpickle(os.path.join(folder, 'meta'))
    clabel_names = bm['coarse_label_names']
    flabel_names = bm['fine_label_names']

    return train_data, np.array(train_coarse_labels), np.array(train_fine_labels), test_data, np.array(test_coarse_labels), np.array(test_fine_labels), clabel_names, flabel_names

#=======================================================
#Specify datapath and get data
#=======================================================
if __name__ == '__main__':
    datapath = "../../caffe/data/cifar-100-python"
    tr_data, tr_clabels, tr_flabels, te_data, te_clabels, te_flabels, clabel_names, flabel_names = get_cifar100(datapath)


tr_N = len(tr_flabels);
te_N = len(te_flabels);

X = np.zeros((te_N, 32, 32, 3), dtype=np.uint8)
y = np.zeros(te_N, dtype=np.int64)

for i in range(te_N):
    X[i] = np.rollaxis(te_data[0].reshape((3,32,32)),0,3)
    y[i] = te_flabels[i]

mapSize1 = X.nbytes * 10
env = lmdb.open('test_data_lmdb', map_size=mapSize1)

with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(te_N):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = X.shape[3]
        datum.height = X.shape[1]
        datum.width = X.shape[2]
        datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
        datum.label = int(y[i])
        str_id = '{:08}'.format(i)

        # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'), datum.SerializeToString())