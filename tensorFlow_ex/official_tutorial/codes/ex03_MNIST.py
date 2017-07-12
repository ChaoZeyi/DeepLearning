import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist.validation.labels
mnist.train.images
#####################input###############
x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
#####################parameter and bias#############
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
###############output################################
y = tf.nn.softmax(tf.matmul(x,w) + b)
sess = tf.Session()
sess.run(w)
