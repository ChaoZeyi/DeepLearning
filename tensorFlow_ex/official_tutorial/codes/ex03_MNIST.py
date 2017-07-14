import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("F:\github\DeepLearning/tensorFlow_ex\official_tutorial\codes\MNIST_data/", one_hot=True)
mnist.validation.labels
mnist.train.images
#####################input 28*28=784 pixes ###############
x = tf.placeholder(dtype=tf.float32, shape=[None, 784])

#####################parameter and bias#############
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

###############output################################
y = tf.nn.softmax(tf.matmul(x,w) + b)  #predicted
y_ = tf.placeholder(dtype=tf.float32, shape=[None,10])

#####################loss##########################
loss = -tf.reduce_sum(y_ * tf.log(y))

######################train########################
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

######################run########################
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(1000):
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train, {x:batch_x, y_:batch_y})

correction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correction,'float32'))
print(sess.run(accuracy, {x:mnist.test.images, y_:mnist.test.labels}))
sess.close()
