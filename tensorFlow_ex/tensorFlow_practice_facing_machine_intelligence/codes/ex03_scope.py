import tensorflow as tf

with tf.name_scope('scope1'):
    a = 1+1
    b = a*3

with tf.name_scope('scope2'):
    c = 2+2
    d = c*3

e = b+d
sess = tf.Session()
writer = tf.summary.FileWriter\
    ('F:\github\DeepLearning/tensorFlow_ex/tensorFlow_practice_facing_machine'\
    '_intelligence\codes/myGraph', sess.graph)
writer.close()
