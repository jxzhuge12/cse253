
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[2]:

# Make sure that caffe is on the python path:
caffe_root = '../../caffe/'  # this file is expected to be in {caffe_root}/examples
import caffe


# In[24]:

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# In[4]:

# GPU mode
caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net('train_val.prototxt',
                './tunefine/examples_iter_50000.caffemodel',
                caffe.TEST)


# In[5]:

net.forward()  # call once for allocation
get_ipython().magic(u'timeit net.forward()')


# In[6]:

[(k, v.data.shape) for k, v in net.blobs.items()]


# In[7]:

[(k, v[0].data.shape) for k, v in net.params.items()]


# In[8]:

def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)


# In[25]:

filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1))


# In[10]:

feat = net.blobs['conv1'].data[0, :36]
vis_square(feat, padval=1)


# In[11]:

filters = net.params['conv2'][0].data
vis_square(filters[:32].reshape(32**2, 5, 5))


# In[12]:

feat = net.blobs['conv2'].data[0, :36]
vis_square(feat, padval=1)


# In[13]:

feat = net.blobs['conv3'].data[0]
vis_square(feat, padval=0.5)


# In[14]:

feat = net.blobs['conv4'].data[0]
vis_square(feat, padval=0.5)


# In[15]:

feat = net.blobs['pool4'].data[0]
vis_square(feat, padval=1)


# In[16]:

feat = net.blobs['ip1'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)


# In[17]:

feat = net.blobs['ip2'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)

