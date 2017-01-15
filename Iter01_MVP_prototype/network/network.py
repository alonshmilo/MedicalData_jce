"""
A convolutional Neural Network implementation for images usinf TensorFlow.
"""


import tensorflow as tf
from network import layers

num_layers = input("How many layers in the network? ")


# Import data


# Parameters
learning_rate = 0.1
training_iters = 20000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784 # data input (img shape = 28*28)
img_shape = 28
n_classes = 2 # number of classes for calssification - bone, not bone ######################## consider 2???
dropout = 0.5 # Dropout, probobility to keep units

# Store weights and biases

weights = {

    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5,5,1,32])),
    # 5x5 conv, 32 input, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5,5,32,64])),
    # fully connected layer,
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs - classification prediction
    'out': tf.Variable(tf.random_normal([1024, n_classes])),
}

biases = {
    # For wc1 - 32 outputs
    'bc1': tf.Variable(tf.random_normal([32])),
    # For wc2 - 64 outputs
    'bc2': tf.Variable(tf.random_normal([64])),
    # For bwd1 - 1024 outputs
    'bd1': tf.Variable(tf.random_normal([1024])),
    # This is the out - classifictation - number of classes
    'out': tf.Variable(tf.random_normal([n_classes])),
}



def conv_net(t, weights, biases, dropout, num_layers):
    """Method for constructing a Convolutional Neural Network
    Args: t input tensor
          weights
          bises
          droput
          num_layers"""

    input = tf.reshape(t, shape=[-1,img_shape,img_shape,1])
    filter = tf.Variable([5,5,1,1], dtype='float32', name='filter')

    # Covolution Layer
    conv1 = layers.Convolution.conv2d(input, weights['wc1'], biases['bc1'])
    # Max Pooling (Down-sampling)
    conv1 = layers.Maxpool.maxpool2d(conv1, k=2)

    # Covolution Layer
    conv2 = layers.Convolution.conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (Down-sampling)
    conv2 = layers.Maxpool.maxpool2d(conv2, k=2)

    # Fully-connected Layer
    #Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1,weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1'], biases['bd1']))
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1,dropout)

    # Output builder - class prediction
    output = tf.add(tf.matmul(fc1, weights['out'], biases['out']))
    return output

prediction = conv_net(x, weights, biases, )