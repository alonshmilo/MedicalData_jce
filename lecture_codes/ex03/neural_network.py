import numpy as np
import itertools
from PIL import Image
import collections
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import datasets
import matplotlib.pyplot as plt
import glob


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Define the non-linear functions used
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def loadImagesByList( file_pattern ):
    image_list = map(Image.open, glob.glob(file_pattern))
    imSizeAsVector = image_list[0].size[0] * image_list[0].size[1]
    images = np.zeros([len(image_list), imSizeAsVector])
    for idx, im in enumerate(image_list):
        images[idx,:] = np.array(im, np.uint8).reshape(imSizeAsVector,1).T

    return images

def loadImages():
    """..."""
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


def get_splitted_data_CYST():

    # Load the raw data
    (all_images, image_class) = loadImages()

    # test / train split
    X_, X_test, y_, y_test = \
        train_test_split(all_images, image_class, test_size=0.20, random_state=42)

    X_train, X_val, y_train, y_val = \
        train_test_split(X_, y_, test_size=0.20, random_state=42)

    print "Total: ", len(all_images), "Train ", str(len(X_train)), ", Val: " \
        , len(X_val) , ", Test: ", len(X_test)

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    #return X_train, y_train.T, X_val, y_val.T, X_test, y_test.T

def get_splitted_data_MNIST():
# load the data from scikit-learn
    digits = datasets.load_digits()

# Load the targets
# Note that the targets are stored as digits, these need to be
#   converted to one-hot-encoding for the output softmax layer.
    T = np.zeros((digits.target.shape[0],10))
    T[np.arange(len(T)), digits.target] += 1

# Divide the data into a train and test set.
    X_train, X_test, T_train, T_test = cross_validation.train_test_split(
    digits.data, T, test_size=0.4)
# Divide the test set into a validation set and final test set.
    X_validation, X_test, T_validation, T_test = cross_validation.train_test_split(
        X_test, T_test, test_size=0.5)
    return X_train, X_test, T_train, T_test, X_validation, T_validation

def get_splitted_data_CIFAR():
    pass

def logistic(z):
    z[z>700]=700
    z[z<700]=-700
    return 1 / (1 + np.exp(-z))

def logistic_deriv(y):  # Derivative of logistic function
    return np.multiply(y, (1 - y))

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Define the layers used in this model
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
class Layer(object):
    """Base class for the different layers.
    Defines base methods and documentation of methods."""

    def get_params_iter(self):
        """Return an iterator over the parameters (if any).
        The iterator has the same order as get_params_grad.
        The elements returned by the iterator are editable in-place."""
        return []

    def get_params_grad(self, X, output_grad):
        """Return a list of gradients over the parameters.
        The list has the same order as the get_params_iter iterator.
        X is the input.
        output_grad is the gradient at the output of this layer.
        """
        return []

    def get_output(self, X):
        """Perform the forward step linear transformation.
        X is the input."""
        pass

    def get_input_grad(self, Y, output_grad=None, T=None):
        """Return the gradient at the inputs of this layer.
        Y is the pre-computed output of this layer (not needed in this case).
        output_grad is the gradient at the output of this layer
         (gradient at input of next layer).
        Output layer uses targets T to compute the gradient based on the
         output error instead of output_grad"""
        pass


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  LogisticLayer
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
class LogisticLayer(Layer):
    """The logistic layer applies the logistic function to its"""

    def get_output(self, X):
        """Perform the forward step transformation."""
        return logistic(X)

    def get_input_grad(self, Y, output_grad):
        """Return the gradient at the inputs of this layer."""
        return np.multiply(logistic_deriv(Y), output_grad)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SoftmaxOutputLayer
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class LinearLayer(Layer):
    """The linear layer performs a linear transformation to its input."""

    def __init__(self, n_in, n_out):
        """Initialize hidden layer parameters.
        n_in is the number of input variables.
        n_out is the number of output variables."""
        self.W = np.random.randn(n_in, n_out) * 0.1
        self.b = np.zeros(n_out)

    def get_params_iter(self):
        """Return an iterator over the parameters."""
        return itertools.chain(np.nditer(self.W, op_flags=['readwrite']),
                               np.nditer(self.b, op_flags=['readwrite']))

    def get_output(self, X):
        """Perform the forward step linear transformation."""
        return X.dot(self.W) + self.b

    def get_params_grad(self, X, output_grad):
        """Return a list of gradients over the parameters."""
        JW = X.T.dot(output_grad)
        Jb = np.sum(output_grad, axis=0)
        return [g for g in itertools.chain(np.nditer(JW), np.nditer(Jb))]

    def get_input_grad(self, Y, output_grad):
        """Return the gradient at the inputs of this layer."""
        return output_grad.dot(self.W.T)

class SoftmaxOutputLayer(Layer):
    """The softmax output layer computes the classification propabilities at the output."""

    def get_output(self, X):
        """Perform the forward step transformation."""
        return softmax(X)

    def get_input_grad(self, Y, T):
        """Return the gradient at the inputs of this layer."""
        return (Y - T) / Y.shape[0]

    def get_cost(self, Y, T):
        """Return the cost at the output of this output layer."""
        return - np.multiply(T, np.log(Y)).sum() / Y.shape[0]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# forward
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Define the forward propagation step as a method.
def forward_step(input_samples, layers):
    """
    Compute and return the forward activation of each layer in layers.
    Input:
        input_samples: A matrix of input samples (each row is an input vector)
        layers: A list of Layers
    Output:
        A list of activations where the activation at each index i+1 corresponds to
        the activation of layer i in layers. activations[0] contains the input samples.
    """
    activations = [input_samples] # List of layer activations
    # Compute the forward activations for each layer starting from the first
    X = input_samples
    for layer in layers:
        Y = layer.get_output(X)  # Get the output of the current layer
        activations.append(Y)  # Store the output for future processing
        X = activations[-1]  # Set the current input as the activations of the previous layer
    return activations  # Return the activations of each layer

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Backward step
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Define the backward propagation step as a method
def backward_step(activations, targets, layers):
    """
    Perform the backpropagation step over all the layers and return the parameter gradients.
    Input:
        activations: A list of forward step activations where the activation at
            each index i+1 corresponds to the activation of layer i in layers.
            activations[0] contains the input samples.
        targets: The output targets of the output layer.
        layers: A list of Layers corresponding that generated the outputs in activations.
    Output:
        A list of parameter gradients where the gradients at each index corresponds to
        the parameters gradients of the layer at the same index in layers.
    """
    param_grads = collections.deque()  # List of parameter gradients for each layer
    output_grad = None  # The error gradient at the output of the current layer
    # Propagate the error backwards through all the layers.
    #  Use reversed to iterate backwards over the list of layers.
    for layer in reversed(layers):
        Y = activations.pop()  # Get the activations of the last layer on the stack
        # Compute the error at the output layer.
        # The output layer error is calculated different then hidden layer error.
        if output_grad is None:
            input_grad = layer.get_input_grad(Y, targets)
        else:  # output_grad is not None (layer is not output layer)
            input_grad = layer.get_input_grad(Y, output_grad)
        # Get the input of this layer (activations of the previous layer)
        X = activations[-1]
        # Compute the layer parameter gradients used to update the parameters
        grads = layer.get_params_grad(X, output_grad)
        param_grads.appendleft(grads)
        # Compute gradient at output of previous layer (input of current layer):
        output_grad = input_grad
    return list(param_grads)  # Return the parameter gradients

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# update_params
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Define a method to update the parameters
def update_params(layers, param_grads, learning_rate):
    """
    Function to update the parameters of the given layers with the given gradients
    by gradient descent with the given learning rate.
    """
    for layer, layer_backprop_grads in zip(layers, param_grads):
        for param, grad in itertools.izip(layer.get_params_iter(), layer_backprop_grads):
            # The parameter returned by the iterator point to the memory space of
            #  the original layer and can thus be modified inplace.
            param -= learning_rate * grad  # Update each parameter




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Define the network
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def define_network(layers, X_train, T_train):
    hidden_neurons_1 = 20  # Number of neurons in the first hidden-layer
    hidden_neurons_2 = 20  # Number of neurons in the second hidden-layer
    # Create the model
    # Add first hidden layer
    layers.append(LinearLayer(X_train.shape[1], hidden_neurons_1))
    layers.append(LogisticLayer())
    # Add second hidden layer
    layers.append(LinearLayer(hidden_neurons_1, hidden_neurons_2))
    layers.append(LogisticLayer())
    # Add output layer
    layers.append(LinearLayer(hidden_neurons_2, T_train.shape[1]))
    layers.append(SoftmaxOutputLayer())

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# train the network
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def train(layers, X_train, T_train, X_validation, T_validation ):
    # Create the minibatches
    batch_size = 25 # Approximately 25 samples per batch
    nb_of_batches = X_train.shape[0] / batch_size # Number
    XT_batch = zip(
        np.array_split(X_train, nb_of_batches, axis=0), # X samples
        np.array_split(T_train, nb_of_batches, axis=0)) # Y targets

    # Perform backpropagation

    # initalize some lists to store the cost for future analysis
    minibatch_costs = []
    training_costs = []
    validation_costs = []

    max_nb_of_iterations = 300  # Train for a maximum of 300 iterations
    learning_rate = 0.9  # Gradient descent learning rate

    # Train for the maximum number of iterations
    for iteration in range(max_nb_of_iterations):
        for X, T in XT_batch:  # For each minibatch sub-iteration
            activations = forward_step(X, layers)  # Get the activations
            minibatch_cost = layers[-1].get_cost(activations[-1], T)  # Get cost
            minibatch_costs.append(minibatch_cost)
            param_grads = backward_step(activations, T, layers)  # Get the gradients
            update_params(layers, param_grads, learning_rate)  # Update the parameters
        # Get full training cost for future analysis (plots)
        activations = forward_step(X_train, layers)
        train_cost = layers[-1].get_cost(activations[-1], T_train)
        training_costs.append(train_cost)
        # Get full validation cost
        activations = forward_step(X_validation, layers)
        validation_cost = layers[-1].get_cost(activations[-1], T_validation)
        validation_costs.append(validation_cost)
        if len(validation_costs) > 3:
            # Stop training if the cost on the validation set doesn't decrease
            #  for 3 iterations
            if validation_costs[-1] >= validation_costs[-2] >= validation_costs[-3]:
                break

    nb_of_iterations = iteration + 1  # The number of iterations that have been executed
    return nb_of_batches, nb_of_iterations, minibatch_costs, training_costs, validation_costs


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# plot the network performance
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def plot(nb_of_iterations, nb_of_batches, minibatch_costs, training_costs, validation_costs):
    # Plot the minibatch, full training set, and validation costs
    minibatch_x_inds = np.linspace(0, nb_of_iterations, num=nb_of_iterations*nb_of_batches)
    iteration_x_inds = np.linspace(1, nb_of_iterations, num=nb_of_iterations)
    # Plot the cost over the iterations
    plt.plot(minibatch_x_inds, minibatch_costs, 'k-', linewidth=0.5, label='cost minibatches')
    plt.plot(iteration_x_inds, training_costs, 'r-', linewidth=2, label='cost full training set')
    plt.plot(iteration_x_inds, validation_costs, 'b-', linewidth=3, label='cost validation set')
    # Add labels to the plot
    plt.xlabel('iteration')
    plt.ylabel('$\\xi$', fontsize=15)
    plt.title('Decrease of cost over backprop iteration')
    plt.legend()
    x1,x2,y1,y2 = plt.axis()
    plt.axis((0,nb_of_iterations,0,2.5))
    plt.grid()
    plt.show()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get results of test data
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_data_results(T_test, X_test, layers):
    y_true = np.argmax(T_test, axis=1)  # Get the target outputs
    activations = forward_step(X_test, layers)  # Get activation of test samples
    y_pred = np.argmax(activations[-1], axis=1)  # Get the predictions made by the network
    # test_accuracy = metrics.accuracy_score(y_true, y_pred)  # Test set accuracy
    # print('The accuracy on the test set is {:.2f}'.format(test_accuracy))

def print_layers(layers):
    for layer in layers:
        print layer



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  MAIN
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -








def main(set):
    """3 different sets: 1 for Cyst, 2 for MNIST, 3 for CIFAR
    """
    layers = []
    if set == 1:
        #X_train, X_test, T_train, T_test, X_validation, T_validation = get_splitted_data_CYST()

        # Load the raw data
        (all_images, image_class) = loadImages()

        # test / train split
        X_, X_test, y_, y_test = \
            train_test_split(all_images, image_class, test_size=0.20, random_state=42)

        X_train, X_val, y_train, y_val = \
            train_test_split(X_, y_, test_size=0.20, random_state=42)

        print "Total: ", len(all_images), "Train ", str(len(X_train)), ", Val: " \
            , len(X_val), ", Test: ", len(X_test)

        # Normalize the data: subtract the mean image
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

        define_network(layers, X_train, y_train)
        nb_of_batches, nb_of_iterations, minibatch_costs, training_costs, validation_costs = train(layers, X_train, y_train, X_val, y_val)

        plot(nb_of_iterations, nb_of_batches, minibatch_costs, training_costs, validation_costs)

    elif set == 2:
        X_train, X_test, y_train, y_test, X_validation, T_validation = get_splitted_data_MNIST()
        define_network(layers, X_train, y_train)
        nb_of_batches, nb_of_iterations, minibatch_costs, training_costs, validation_costs = train(layers, X_train, y_train,
                                                                                               X_test, y_test)
        plot(nb_of_iterations, nb_of_batches, minibatch_costs, training_costs, validation_costs)

    elif set == 3:
        X_train, X_test, y_train, y_test, X_validation, T_validation = get_splitted_data_CIFAR()
        define_network(layers, X_train, y_train)
        nb_of_batches, nb_of_iterations, minibatch_costs, training_costs, validation_costs = train(layers, X_train,
                                                                                                   y_train,
                                                                                                   X_test, y_test)
        plot(nb_of_iterations, nb_of_batches, minibatch_costs, training_costs, validation_costs)
if __name__ == "__main__":
    main(2)


#learning rate line - 314