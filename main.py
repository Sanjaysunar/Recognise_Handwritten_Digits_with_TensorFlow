import tensorflow as tf

# Import MNIST dataset and store the image data in the variable 'mnist'
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True) # ylabels are oh-encoded

# split dataset
n_train = mnist.train.num_examples  # 55,000
n_validation = mnist.validation.num_examples  # 5000
n_test = mnist.test.num_examples  # 10,000


# Defining the Neural Network Architecture
n_input = 784  # input layer (28x28 pixels)
n_hidden1 = 512  # 1st hidden layer
n_hidden2 = 256  # 2nd hidden layer
n_hidden3 = 128  # 3rd hidden layer
n_output = 10  # output layer (0-9 digits)

learning_rate = 1e-4
n_iterations = 1000
batch_size = 128
dropout = 0.5

# Building the TensorFlow Graph
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])
keep_prob = tf.placeholder(tf.float32)

#Random Values from a truncated normal distribution
# For the weights
weights = {
    'w1' : tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
    'w2' : tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
    'w3' : tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
    'out' : tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1))
}

# For the biases
biases = {
    'b1' : tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
    'b2' : tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
    'b3' : tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
    'out' : tf.Variable(tf.constant(0.1, shape=[n_output]))
}

# Setup the layers of the network
layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
layer_drop = tf.nn.dropout(layer_3, keep_prob)
output_layer = tf.matmul(layer_3, weights['out']) + biases['out']

# Using Adam Optimizer

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        labels=Y, logits=output_layer
    ))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)