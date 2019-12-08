#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import moxing.tensorflow as mox
import gzip
import os

import numpy
from scipy import ndimage

from six.moves import urllib

import tensorflow as tf
tf.reset_default_graph()


# In[40]:


####### your coding place： begin  ###########
# 此处必须修改为用户数据桶位置

#数据在OBS的存储位置。
# eg. s3:// ：统一路径输入
#     /uBucket ：桶名，用户的私有桶的名称 eg. bucket
#     /notebook/data/： 文件路径

data_url = 's3://zhh-obs001/test-modelarts/dataset-mnist/' 
if not mox.file.exists(data_url):
    raise ValueError('Plese verify your data url!')
####### your coding place： end  ###########


# In[42]:


# 本地创建数据存储文件夹
local_url = './cache/local_data/'
if mox.file.exists(local_url):
    mox.file.remove(local_url,recursive=True)
os.makedirs(local_url)

#将私有桶中的数据拷贝到本地mox.file.copy_parallel（）
"""
  Copy all files in src_url to dst_url. Same usage as `shutil.copytree`.
  Note that this method can only copy a directory. If you want to copy a single file,
  please use `mox.file.copy`

  Example::

    copy_parallel(src_url='/tmp', dst_url='s3://bucket_name/my_data')

  Assuming files in `/tmp` are:

  * /tmp:
      * |- train
          * |- 1.jpg
          * |- 2.jpg
      * |- eval
          * |- 3.jpg
          * |- 4.jpg

  Then files after copy in `s3://bucket_name/my_data` are:

  * s3://bucket_name/my_data:
      * |- train
          * |- 1.jpg
          * |- 2.jpg
      * |- eval
          * |- 3.jpg
          * |- 4.jpg

  Directory `tmp` will not be copied. If `file_list` is `['train/1.jpg', 'eval/4.jpg']`,
  then files after copy in `s3://bucket_name/my_data` are:

  * s3://bucket_name/my_data
      * |- train
          * |- 1.jpg
      * |- eval
          * |- 4.jpg

  :param src_url: Source path or s3 url
  :param dst_url: Destination path or s3 url
  :param file_list: A list of relative path to `src_url` of files need to be copied.
  :param threads: Number of threads or processings in Pool.
  :param is_processing: If True, multiprocessing is used. If False, multithreading is used.
  :param use_queue: Whether use queue to manage downloading list.
  :return: None
"""
mox.file.copy_parallel(data_url, local_url)
data_url = local_url
os.listdir(data_url)


# In[45]:


DATA_DIRECTORY = data_url
# Params for MNIST
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.


# In[46]:


def maybe_download(filename):
    """Download the data from Yann's website, unless it's already here."""
    if not tf.gfile.Exists(DATA_DIRECTORY):
        tf.gfile.MakeDirs(DATA_DIRECTORY)
    filepath = os.path.join(DATA_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        print(filepath," does not exist")
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath

# Extract the images
def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].

    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        data = numpy.reshape(data, [num_images, -1])
    return data

# Extract the labels
def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
        num_labels_data = len(labels)
        one_hot_encoding = numpy.zeros((num_labels_data,NUM_LABELS))
        one_hot_encoding[numpy.arange(num_labels_data),labels] = 1
        one_hot_encoding = numpy.reshape(one_hot_encoding, [-1, NUM_LABELS])
    return one_hot_encoding

# Augment training data
def expend_training_data(images, labels):

    expanded_images = []
    expanded_labels = []

    j = 0 # counter
    for x, y in zip(images, labels):
        j = j+1
        if j%100==0:
            print ('expanding data : %03d / %03d' % (j,numpy.size(images,0)))

        # register original data
        expanded_images.append(x)
        expanded_labels.append(y)

        # get a value for the background
        # zero is the expected value, but median() is used to estimate background's value 
        bg_value = numpy.median(x) # this is regarded as background's value        
        image = numpy.reshape(x, (-1, 28))

        for i in range(4):
            # rotate the image with random degree
            angle = numpy.random.randint(-15,15,1)
            new_img = ndimage.rotate(image,angle,reshape=False, cval=bg_value)

            # shift the image with random distance
            shift = numpy.random.randint(-2, 2, 2)
            new_img_ = ndimage.shift(new_img,shift, cval=bg_value)

            # register new training data
            expanded_images.append(numpy.reshape(new_img_, 784))
            expanded_labels.append(y)

    # images and labels are concatenated for random-shuffle at each epoch
    # notice that pair of image and label should not be broken
    expanded_train_total_data = numpy.concatenate((expanded_images, expanded_labels), axis=1)
    numpy.random.shuffle(expanded_train_total_data)

    return expanded_train_total_data

# Prepare MNISt data
def prepare_MNIST_data(use_data_augmentation=True):
    # Get the data.
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)

    # Generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE, :]
    validation_labels = train_labels[:VALIDATION_SIZE,:]
    train_data = train_data[VALIDATION_SIZE:, :]
    train_labels = train_labels[VALIDATION_SIZE:,:]

    # Concatenate train_data & train_labels for random shuffle
    if use_data_augmentation:
        train_total_data = expend_training_data(train_data, train_labels)
    else:
        train_total_data = numpy.concatenate((train_data, train_labels), axis=1)

    train_size = train_total_data.shape[0]

    return train_total_data, train_size, validation_data, validation_labels, test_data, test_labels

