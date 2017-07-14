import tensorflow as tf

sess = tf.Session()
sess.close()

with tf.Sesion() as sess:
    a = tf.constant(3, dtype=tf.float32)
    b = a+1
    sess.run(b, {a:5})
