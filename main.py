import tensorflow as tf

# Import MNIST dataset and store in 'mnist'
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
