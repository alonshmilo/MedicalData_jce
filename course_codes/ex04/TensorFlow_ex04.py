#Alon Shmilovich 034616359 JCE, Jerusalem College of Engineering
#Machine Learning to Medical data - ex04


# coding: utf-8

# In[6]:

import os
import random
import tarfile
import sys
import time
import tensorflow as tf
from IPython.display import clear_output
from scipy import ndimage
import numpy as np
from six.moves.urllib.request import urlretrieve
import matplotlib.pyplot as plt
import cPickle
from PIL import Image
import glob
get_ipython().magic(u'matplotlib inline')

get_ipython().system(u'pip install scikit-learn # for Cyst')
from sklearn.cross_validation import train_test_split

#Ask for configurations from user

dataset = int(input('Enter data-set number:\n 1. Cyst 2. CIFAR-10 3. notMNIST 4. MNIST\n'))
network = int(input('Choose a network you would like to use:\n 1. Custom Network 2. AlexNet\n'))

#configurations: 
#1. One Convolutional Layer, Regulatization=0, Learning Rate=0.06
#2. One Convolutional Layer, Regulatization=0, Learning Rate=0.09
#2. Two Convolutional Layers, Regularization=L1, Learning Rate=0.1
#3. Two Convolutional Layers, Regularization=L2, Learning Rate=0.25

if network == 1: 
    if dataset == 1: # Cyst
        configuration = int(input('Please choose configuration: 1 - 2 - 3 - 4'))
    elif dataset == 2: # CIFAR-10
        configuration = int(input('Please choose configuration: 1 - 2 - 3 - 4'))
    elif dataset == 3: # notMNIST
        configuration = int(input('Please choose configuration: 1 - 2 - 3 - 4'))
    elif dataset == 4: # MNIST
        configuration = int(input('Please choose configuration: 1 - 2 - 3 - 4'))
        
elif network == 2:
    if dataset == 1: # Cyst
        configuration = int(input('Please choose configuration: 1 - 2 - 3 -4'))
    elif dataset == 2: # CIFAR-10
        configuration = int(input('Please choose configuration: 1 - 2 - 3 -4'))
    elif dataset == 3: # notMNIST
        configuration = int(input('Please choose configuration: 1 - 2 - 3 -4'))
    elif dataset == 4: # MNIST
        configuration = int(input('Please choose configuration: 1 - 2 - 3 -4'))
    


# In[3]:

# Some Helper Functions

def download(filename):
    destination_file = "data/" + filename
    if not os.path.exists(destination_file):
        print("Dowloading ", filename, "into ", destination_file)
        urlretrieve(url + filename, destination_file)
    else:
        print "File already exists: %s" %filename
    return destination_file

def untar(filename):
    folder = filename.split(".tar")[0]
    
    if os.path.isdir(folder):
        print("%s already extracted" %filename)
    else:
        print("Extracting %s, please wait" %filename)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall("data")
        tar.close()

    extracted_folders = [
        os.path.join(folder, subfolder) for subfolder in sorted(os.listdir(folder))
        if os.path.isdir(os.path.join(folder, subfolder))]
    print(extracted_folders)
    return extracted_folders

def untar_cifar(filename):
    folder = filename.split(".tar")[0]
    
    if os.path.isdir(folder):
        print("%s already extracted" %filename)
    else:
        print("Extracting %s, please wait" %filename)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(folder)
        tar.close()
        
    extracted_folders = [
        os.path.join(folder, subfolder) for subfolder in sorted(os.listdir(folder))
        if os.path.isdir(os.path.join(folder, subfolder))]
    return extracted_folders

# Helper functions for Cyst dataset
def loadImagesByList( file_pattern ):
    image_list = map(Image.open, glob.glob(file_pattern))
    imSizeAsVector = image_list[0].size[0] * image_list[0].size[1]
    images = np.zeros([len(image_list), imSizeAsVector])
    for idx, im in enumerate(image_list):
        images[idx,:] = np.array(im, np.uint8).reshape(imSizeAsVector,1).T
    return images

def loadImages():
    """Here we load the sets of images"""
    OK_file_pattern = 'image_patches/*OK*axial.png'
    Cyst_file_pattern = 'image_patches/*Cyst*axial.png'

    # OK-images
    OK_image = loadImagesByList(OK_file_pattern)

    # Cyst-images
    Cyst_image = loadImagesByList(Cyst_file_pattern)

    # concatenate the two types
    image_class = np.concatenate( (np.zeros([OK_image.shape[0],1]) ,
                                   np.ones([Cyst_image.shape[0],1]) ) )
    all_images = np.concatenate((OK_image, Cyst_image))
    return (all_images, image_class)

def get_splitted_data_Cyst():

    # Load the raw data
    (all_images, image_class) = loadImages()

    # test / train split
    X_, X_test, y_, y_test =         train_test_split(all_images, image_class, test_size=0.20, random_state=42)

    X_train, X_val, y_train, y_val =         train_test_split(X_, y_, test_size=0.20, random_state=42)

    print "Total: ", len(all_images), "Train ", str(len(X_train)), ", Val: "         , len(X_val) , ", Test: ", len(X_test)

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    #return X_train, y_train.T[0], X_val, y_val.T[0], X_test, y_test.T[0]
    return X_train, y_train, X_val, y_val, X_test, y_test

def unpickle(file):
    import cPickle
    # TODO load all data_batches
    fo = open(file + '/data_batch_1', 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

# display a two random image from each letter in the train folder
def print_images(folders):
    images = []
    for folder in folders:
        images_path = os.listdir(folder)
        for i in range(2):
            image_path = os.path.join(folder,
                                      random.choice(images_path))
            images.append(ndimage.imread(image_path))
#             images.append(plt.imread(image_path))
#             images.append(cv2.imread(image_path))
    plt.figure(figsize=(20, 1))
    plt_imshow(np.hstack(images))
    print 'Maximum pixel intensity value: %.2f' %np.amax(images[0])
    print 'Shape of images ' + str(images[0].shape) 


# In[7]:

# Cyst data

if dataset == 1:
    num_imgs = 250 # images per class
    num_classes = 2
    img_size = 39 # pixel size
    print("[MAIN]init network for Cysts...")
    # Load the raw data
    (all_images, image_class) = loadImages()
    
    # test / train split
    X_, X_test, y_, y_test =         train_test_split(all_images, image_class, test_size=0.20, random_state=42)

    X_train, X_val, y_train, y_val =         train_test_split(X_, y_, test_size=0.20, random_state=42)

    print "Total: ", len(all_images), "Train ", str(len(X_train)), ", Val: "         , len(X_val) , ", Test: ", len(X_test)

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image


# In[5]:

# CIFAR-10 data 

if dataset == 2:
    num_imgs = 6000 # images per class
    num_classes = 10
    img_size = 32 # pixel size
#     from tensorflow.models.image.cifar10 import cifar10
    import cifar10_input as cifar10
    
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    
#     data_filename = download('cifar-10-binary.tar.gz')
    folder = untar_cifar('data/cifar-10-binary.tar.gz')
    dataset, labels = cifar10.distorted_inputs('data/cifar-10-binary/cifar-10-batches-bin', 128)
    
#     folder = untar_cifar('data/cifar-10-python.tar.gz')
#     data_folders = unpickle(folder[0])


# In[ ]:

# notMNIST data

if dataset == 3:
    num_imgs = 1500 # images per class
    num_classes = 10
    img_size = 28 # pixel size
    url = 'http://yaroslavvb.com/upload/notMNIST/'

    def plt_imshow(image):
        plt.imshow(image, cmap = 'gray');
        plt.axis('off')
        plt.show()
        
    data_filename = download('notMNIST_small.tar.gz')
    
    print '\nData folders:'

    data_folders = untar(data_filename)



# In[ ]:

# MNIST data

if dataset == 4:
    num_imgs = 1500 # images per class
    num_classes = 10
    img_size = 28 # pixel size
    from tensorflow.examples.tutorials.mnist import input_data
    
    # load MNIST data
    data_folders = input_data.read_data_sets("MNIST_data/", one_hot=True)
    


# In[ ]:

if dataset == 1:
    print_images(data_folders)


# In[ ]:


# num_imgs = 1500 # images per class
# num_classes = 10
# img_size = 28 # pixel size

def build_dataset(folders):
    dataset = np.ndarray((num_imgs * num_classes, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(num_imgs * num_classes, dtype=np.int32)
    counter = 0
    for img_class, folder in enumerate(folders):
        per_class_counter = 0
        for img_name in os.listdir(folder):
            if per_class_counter < num_imgs:
                img_path = os.path.join(folder, img_name)
                try:
                    img = ndimage.imread(img_path).astype(float) # Convert to float
                    img = (img - 255 / 2) / 255 # Normalization
                    if img.shape == (img_size, img_size):
                        dataset[counter] = img
                        labels[counter] = img_class
                        counter += 1
                        per_class_counter += 1
                    else:
                        raise Exception("Unexpected image shape")
                except Exception as e:
                    print 'Unable to use image: ' + str(e)
    return dataset, labels


# In[ ]:

if dataset == 1:
    dataset, labels = build_dataset(data_folders)
    print '\nTotal number of images: %d' %dataset.shape[0]
    print 'Images Shape:' + str(dataset[0].shape)
    print 'Dataset Shape: ' + str(dataset.shape)


# In[ ]:

train_size = 1000
valid_size = 250
test_size = 250

train_ds = np.ndarray((train_size * num_classes, img_size, img_size),dtype=np.float32)
train_lb = np.ndarray(train_size * num_classes, dtype=np.int32)

valid_ds = np.ndarray((valid_size * num_classes, img_size, img_size), dtype=np.float32)
valid_lb = np.ndarray(valid_size * num_classes, dtype=np.int32)

test_ds = np.ndarray((test_size * num_classes, img_size, img_size), dtype=np.float32)
test_lb = np.ndarray(test_size * num_classes, dtype=np.int32)


# In[ ]:

if dataset == 1:
    for i in range(10):
        start_set, end_set = i * num_imgs, (i + 1) * num_imgs
        start_train, end_train = i * train_size, (i + 1) * train_size
        start_valid, end_valid = i * valid_size, (i + 1) * valid_size
        start_test, end_test = i * test_size, (i + 1) * test_size

        letter_set = dataset[start_set : end_set]
        np.random.shuffle(letter_set)

        train_ds[start_train : end_train] = letter_set[0: train_size]
        train_lb[start_train : end_train] = i
        valid_ds[start_valid : end_valid] = letter_set[train_size: train_size + valid_size]
        valid_lb[start_valid : end_valid] = i
        test_ds[start_test : end_test] = letter_set[train_size + valid_size: train_size + valid_size + test_size]
        test_lb[start_test : end_test] = i

    print("Train Shapes -->  Dataset: %s   Labels: %s" %(train_ds.shape, train_lb.shape))
    print("Valid Shapes --> Dataset: %s    Labels: %s" %(valid_ds.shape, valid_lb.shape))
    print("Test Shapes --> Dataset: %s    Labels: %s" %(test_ds.shape, test_lb.shape))


# In[ ]:

# Visualize Datasets
for i in np.random.randint(0, 2500, 10):
    clear_output(wait="True")
    plt.imshow(np.hstack((train_ds[i], test_ds[i], valid_ds[i])), cmap = 'gray')
    plt.title("Train Set " + str(train_lb[i]) + 
              "  -  Test Set" + str(test_lb[i]) + 
              "  -  Validation Set" + str(valid_lb[i]))
    plt.axis('off')
    plt.show()
    time.sleep(0.1)


# In[ ]:

# Randomize Dataset
def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_ds = dataset[permutation]
    shuffled_lb = labels[permutation]
    return shuffled_ds, shuffled_lb

train_ds, train_lb = randomize(train_ds, train_lb)
test_ds, test_lb = randomize(test_ds, test_lb)
valid_ds, valid_lb = randomize(valid_ds, valid_lb)

for i in np.random.randint(0, 2500, 5):
    clear_output(wait="Ture")
    plt.imshow(np.hstack((train_ds[i], test_ds[i], valid_ds[i])), cmap = 'gray')
    plt.title("Train Set " + str(train_lb[i]) + 
              "  -  Test Set" + str(test_lb[i]) + 
              "  -  Validation Set" + str(valid_lb[i]))
    plt.axis('off')
    plt.show()
    time.sleep(0.1)


# In[ ]:

# Reformat Input Shape
num_channels = 1 # grayscale

def reformat(dataset, labels):
    # as.type is not needed as the array is already float32 but just in case
    dataset = dataset.reshape((-1, img_size, img_size, num_channels)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_classes) == labels[:, None]).astype(np.float32)
    return dataset, labels

train_ds, train_lb = reformat(train_ds, train_lb)
valid_ds, valid_lb = reformat(valid_ds, valid_lb)
test_ds, test_lb = reformat(test_ds, test_lb)

print("Train Shapes --> Dataset: %s   Labels: %s" %(train_ds.shape, train_lb.shape))
print("Valid Shapes --> Dataset: %s    Labels: %s" %(valid_ds.shape, valid_lb.shape))
print("Test Shapes --> Dataset: %s    Labels: %s" %(test_ds.shape, test_lb.shape))


# In[ ]:

def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


# In[ ]:

# Custom Network
if network == 1:
    
    batch_size = 50
    patch_size = 5
    patch_size2 = 4
    depth1 = 2
    depth2 = 16
    num_hidden = 4

    graph = tf.Graph()

    with graph.as_default():

        #Input data
        tf_train_ds = tf.placeholder(tf.float32, shape=(batch_size, img_size, img_size, num_channels))
        tf_train_lb = tf.placeholder(tf.float32, shape=(batch_size, num_classes))
        tf_valid_ds = tf.constant(valid_ds)
        tf_test_ds = tf.constant(test_ds)

        # Variables.
        patch1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth1], stddev=0.1))
        patch1_biases = tf.Variable(tf.zeros([depth1]))

        patch2_weights = tf.Variable(tf.truncated_normal([patch_size2, patch_size2, depth1, depth2], stddev=0.1))
        patch2_biases = tf.Variable(tf.constant(1.0, shape=[depth2]))

        # divided by four because that is the size once the patches have scanned the image
#         layer1_weights = tf.Variable(tf.truncated_normal(
#                                      [img_size // 4 * img_size // 4 * depth1, num_classes], stddev=0.1))
        layer1_weights = tf.Variable(tf.truncated_normal(
                                     [img_size * img_size * num_channels, num_classes], stddev=0.1))
        layer1_biases = tf.Variable(tf.constant(1.0, shape=[num_classes]))


        layer2_weights = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
        layer2_biases = tf.Variable(tf.constant(1.0, shape=[num_classes]))

        # Model
        def model(data, training):
            
            # first convolution layer. Stride only matter in two elements in the middle
            conv = tf.nn.conv2d(data, patch1_weights, [1, 4, 4, 1], padding="SAME")
            conv = tf.nn.max_pool(conv + patch1_biases, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME" )
            conv = tf.nn.relu(conv)

            # second convolution layer
            if configurarion == 2 or configurarion == 3:
                conv = tf.nn.conv2d(conv, patch2_weights, [1, 2, 2, 1], padding="SAME")
                conv = tf.nn.max_pool(conv + patch2_biases, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME" )
                conv = tf.nn.relu(conv)

            # reshape to apply fully connected layer
            shape_conv = conv.get_shape().as_list()
            input_hidden = tf.reshape(conv, [shape_conv[0], shape_conv[1] * shape_conv[2] * shape_conv[3]])
            input_hidden = tf.reshape(data, [-1, img_size * img_size * num_channels])
            hidden_layer = tf.nn.relu(tf.matmul(input_hidden, layer1_weights) + layer1_biases)

            # adding dropout layer
            if training:
                hidden_layer = tf.nn.dropout(hidden_layer, 0.6)

            return tf.matmul(input_hidden, layer1_weights) + layer1_biases
#             return tf.matmul(hidden_layer, layer2_weights) + layer2_biases

        # training computation
        logits = model(tf_train_ds, True)
        
        #set rugularization
        if configuration == 1:
            regularization = 0
        elif configuration == 2:
            regularization = tf.nn.l2_loss(layer1_weights)
#             regularization = tf.nn.l2_loss(layer2_weights)
        elif configuration == 3:
            regularization = tf.nn.l2_loss(layer2_weights)
            
        regularization_param = 0.0005
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_lb)) +                regularization_param * regularization

        # Optimizer
        global_step = tf.Variable(0)
        
        # Set the learning rate due to configuration
        if configuration == 1:
            learning_rate = 0.06
        elif configuration == 2:
            learning_rate = 0.09            
        elif configuration == 3:
            learning_rate = 0.1           
        elif configuration == 4:
            decay = 0.05
            learning_rate = tf.train.exponential_decay(decay, global_step, 200, 0.95, staircase = True)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)


        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(model(tf_valid_ds, False))
        test_prediction = tf.nn.softmax(model(tf_test_ds, False))
    


# In[ ]:

if network_num == 2:

    ################################################################################
    #Michael Guerzhoy, 2016
    #AlexNet implementation in TensorFlow, with weights
    #Details: 
    #http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
    #
    #With code from https://github.com/ethereon/caffe-tensorflow
    #Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
    #Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
    #
    #
    ################################################################################

    from numpy import *
    import os
    from pylab import *
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cbook as cbook
    import time
    from scipy.misc import imread
    from scipy.misc import imresize
    import matplotlib.image as mpimg
    from scipy.ndimage import filters
    import urllib
    from numpy import random


    import tensorflow as tf

    from caffe_classes import class_names

    train_x = zeros((1, 227,227,3)).astype(float32)
    train_y = zeros((1, 1000))
    xdim = train_x.shape[1:]
    ydim = train_y.shape[1]



    ################################################################################
    #Read Image

#     x_dummy = (random.random((1,)+ xdim)/255.).astype(float32)
#     i = x_dummy.copy()
#     i[0,:,:,:] = (imread("poodle.png")[:,:,:3]).astype(float32)
#     i = i-mean(i)

    i[0,:,:,:] = imread(data_folders)

    ################################################################################

    # (self.feed('data')
    #         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    #         .lrn(2, 2e-05, 0.75, name='norm1')
    #         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    #         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
    #         .lrn(2, 2e-05, 0.75, name='norm2')
    #         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    #         .conv(3, 3, 384, 1, 1, name='conv3')
    #         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
    #         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
    #         .fc(4096, name='fc6')
    #         .fc(4096, name='fc7')
    #         .fc(1000, relu=False, name='fc8')
    #         .softmax(name='prob'))


    net_data = load("bvlc_alexnet.npy").item()

    def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
        '''From https://github.com/ethereon/caffe-tensorflow
        '''
        c_i = input.get_shape()[-1]
        assert c_i%group==0
        assert c_o%group==0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)


        if group==1:
            conv = convolve(input, kernel)
        else:
            input_groups = tf.split(3, group, input)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
            conv = tf.concat(3, output_groups)
        return  tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())



    x = tf.Variable(i)

    #conv1
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)


    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)


    #conv5
    #conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv5W = tf.Variable(net_data["conv5"][0])
    conv5b = tf.Variable(net_data["conv5"][1])
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)

    #maxpool5
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #fc6
    #fc(4096, name='fc6')
    fc6W = tf.Variable(net_data["fc6"][0])
    fc6b = tf.Variable(net_data["fc6"][1])
    fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

    #fc7
    #fc(4096, name='fc7')
    fc7W = tf.Variable(net_data["fc7"][0])
    fc7b = tf.Variable(net_data["fc7"][1])
    fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

    #fc8
    #fc(1000, relu=False, name='fc8')
    fc8W = tf.Variable(net_data["fc8"][0])
    fc8b = tf.Variable(net_data["fc8"][1])
    fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)


    #prob
    #softmax(name='prob'))
    prob = tf.nn.softmax(fc8)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    output = sess.run(prob)
    ################################################################################

    #Output:

    inds = argsort(output)[0,:]
    for i in range(5):
        print class_names[inds[-1-i]], output[0, inds[-1-i]]



# In[ ]:

num_steps = 201

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        # randomize offset
        offset = (step * batch_size) % (train_lb.shape[0] - batch_size)
        batch_ds = train_ds[offset:(offset + batch_size)]
        batch_lb = train_lb[offset:(offset + batch_size)]
        
        feed_dict = {tf_train_ds : batch_ds, tf_train_lb : batch_lb}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_lb))
            print('Validation accuracy: %.1f%%' % accuracy(
            valid_prediction.eval(), valid_lb))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_lb))

