import tensorflow as tf

class Convolution:

    def conv2d (input, W, b, strides=1):
        # Conv2d wrapper, with bias and relu actiation
        x = tf.nn.conv2d(input, W, strides = [1,strides, strides,1], padding='SAME')
        x = tf.nn.bias_add(input,b)
        return tf.nn.relu(input)

class Maxpool:

    def maxpool2d(x, k=2):
        # MaxPool2d wrapper
        ksize = [1,k,k,1]
        strides = [1,k,k,1]
        return tf.nn.max_pool(x, ksize, strides, padding='SAME')