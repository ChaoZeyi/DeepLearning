import tensorflow as tf
hello = tf.constant('hello google tensorflow')
sess = tf.Session()
result = sess.run(hello)
print result

node1 = tf.constant(3.0, dtype=tf.float64)
node2 = tf.constant(4.0)
print(node1, node2)
node3 = tf.constant(5.0)
print node3
print sess.run(node3)
node4 = tf.add(node2, node3)
print node4
print ("node4:", node4)
sess.close()
