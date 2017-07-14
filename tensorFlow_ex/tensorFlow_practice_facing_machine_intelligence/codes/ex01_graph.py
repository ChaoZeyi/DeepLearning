import tensorflow as tf

g1 = tf.get_default_graph()
g2 = tf.Graph()

with g1.as_default():
    a = tf.constant(1, dtype=tf.float32)
    b = tf.constant(2, dtype=tf.float32)
    e1 = a+b

with g2.as_default():
    c = tf.constant(3, dtype=tf.float32)
    d = tf.constant(3, dtype=tf.float32)
    e2 = c+d

sess = tf.Session()
writer = tf.summary.FileWriter\
    ('F:\github\DeepLearning/tensorFlow_ex/tensorFlow_practice_facing_machine'\
    '_intelligence\codes/myGraph', sess.graph)
writer.close()
sess.close()
