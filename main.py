import tensorflow as tf

# Import MNIST dataset and store the image data in the variable 'mnist'
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True) # ylabels are oh-encoded

# split dataset
n_train = mnist.train.num_examples  # 55,000
n_validation = mnist.validation.num_examples  # 5000
n_test = mnist.test.num_examples  # 10,000


