import tensorflow as tf

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0, dtype=tf.float32)
c = a+b
d = a*b
e = c+d

sess = tf.Session()
print(sess.run(e))


writer = tf.summary.FileWriter('F:\github\DeepLearning/tensorFlow_ex/official_tutorial\codes/myGraph', sess.graph)
writer.close()
sess.close()
