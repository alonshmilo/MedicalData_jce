
# coding: utf-8

# In[1]:

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
get_ipython().magic(u'matplotlib inline')


# In[4]:

img_size = 28 # pixel size
num_imgs = 1500 # images per class
num_classes = 10

url = 'http://commondatastorage.googleapis.com/books1000/'

def download(filename):
    destination_file = "data/" + filename
    if not os.path.exists(destination_file):
        print("Dowloading ", filename, "into ", destination_file)
        urlretrieve(url + filename, destination_file)
    else:
        print "File already exists: %s" %filename
    return destination_file


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


if not os.path.exists('data/pictures_tf'):
    os.makedirs('data/pictures_tf')

data_filename = download('notMNIST_small.tar.gz')
print '\nData folders:'
data_folders = untar(data_filename)


# In[5]:

dataset, labels = build_dataset(data_folders)
print '\nTotal number of images: %d' %dataset.shape[0]
print 'Images Shape:' + str(dataset[0].shape)
print 'Dataset Shape: ' + str(dataset.shape)


# In[6]:

train_size = 1000
valid_size = 250
test_size = 250

train_ds = np.ndarray((train_size * num_classes, img_size, img_size),dtype=np.float32)
train_lb = np.ndarray(train_size * num_classes, dtype=np.int32)

valid_ds = np.ndarray((valid_size * num_classes, img_size, img_size), dtype=np.float32)
valid_lb = np.ndarray(valid_size * num_classes, dtype=np.int32)

test_ds = np.ndarray((test_size * num_classes, img_size, img_size), dtype=np.float32)
test_lb = np.ndarray(test_size * num_classes, dtype=np.int32)


# In[7]:

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

print("Train Shapes --> Dataset: %s   Labels: %s" %(train_ds.shape, train_lb.shape))
print("Valid Shapes --> Dataset: %s    Labels: %s" %(valid_ds.shape, valid_lb.shape))
print("Test Shapes  --> Dataset: %s    Labels: %s" %(test_ds.shape, test_lb.shape))


# In[8]:

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_ds = dataset[permutation]
    shuffled_lb = labels[permutation]
    return shuffled_ds, shuffled_lb

train_ds, train_lb = randomize(train_ds, train_lb)
test_ds, test_lb = randomize(test_ds, test_lb)
valid_ds, valid_lb = randomize(valid_ds, valid_lb)

    
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
print("Test Shapes  --> Dataset: %s    Labels: %s" %(test_ds.shape, test_lb.shape))    


# In[9]:

num_channels = 1 # grayscale

batch_size = 50
patch_size = 5
depth1 = 2
# depth2 = 16
# num_hidden = 4

graph = tf.Graph()

with graph.as_default():
    
    #Input data
    tf_train_ds = tf.placeholder(tf.float32, shape=(batch_size, img_size, img_size, num_channels))
    tf_train_lb = tf.placeholder(tf.float32, shape=(batch_size, num_classes))
    tf_valid_ds = tf.constant(valid_ds)
    tf_test_ds = tf.constant(test_ds)
    
    # Variables.
# patch1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth1], stddev=0.1))
# patch1_biases = tf.Variable(tf.zeros([depth1]))

# patch2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth1, depth2], stddev=0.1))
# patch2_biases = tf.Variable(tf.constant(1.0, shape=[depth2]))
    
# divided by four because that is the size once the patches have scanned the image
#   layer1_weights = tf.Variable(tf.truncated_normal(
#                                  [img_size // 4 * img_size // 4 * depth1, num_classes], stddev=0.1))
    layer1_weights = tf.Variable(tf.truncated_normal(
                                 [img_size * img_size * num_channels, num_classes], stddev=0.1))
    layer1_biases = tf.Variable(tf.constant(1.0, shape=[num_classes]))
    
    
# layer2_weights = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
# layer2_biases = tf.Variable(tf.constant(1.0, shape=[num_classes]))
    
    # Model
    def model(data, training):
        # first convolution layer. Stride only matter in two elements in the middle
#         conv = tf.nn.conv2d(data, patch1_weights, [1, 4, 4, 1], padding="SAME")
#         conv = tf.nn.max_pool(conv1 + patch1_biases, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME" )
#         conv = tf.nn.relu(conv)
        
        # second convolution layer
#         conv = tf.nn.conv2d(conv, patch2_weights, [1, 2, 2, 1], padding="SAME")
#         conv = tf.nn.max_pool(conv + patch2_biases, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME" )
#         conv = tf.nn.relu(conv)

        # reshape to apply fully connected layer
#         shape_conv = conv.get_shape().as_list()
#         input_hidden = tf.reshape(conv, [shape_conv[0], shape_conv[1] * shape_conv[2] * shape_conv[3]])
        input_hidden = tf.reshape(data, [-1, img_size * img_size * num_channels])
#         hidden_layer = tf.nn.relu(tf.matmul(input_hidden, layer1_weights) + layer1_biases)
        
        # adding dropout layer
#         if training:
#             hidden_layer = tf.nn.dropout(hidden_layer, 0.6)
        
        return tf.matmul(input_hidden, layer1_weights) + layer1_biases
#         return tf.matmul(hidden_layer, layer2_weights) + layer2_biases
    
    # training computation
    logits = model(tf_train_ds, True)
    regularization = 0 #tf.nn.l2_loss(layer1_weights) #+ tf.nn.l2_loss(layer2_weights)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_lb)) +            .0005 * regularization
    
    # Optimizer
    global_step = tf.Variable(0)
#     learning_rate = tf.train.exponential_decay(0.05, global_step, 200, 0.95, staircase = True)
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    optimizer = tf.train.GradientDescentOptimizer(.05).minimize(loss, global_step=global_step)


    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_ds, False))
    test_prediction = tf.nn.softmax(model(tf_test_ds, False))



# In[11]:

def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


# In[12]:

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

