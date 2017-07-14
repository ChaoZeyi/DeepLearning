import tensorflow as tf

###############basic operations including digits and matrixs####################
########################digits#################################
a = tf.constant(3, dtype=tf.float32)
b = tf.constant(5, dtype=tf.float32)
sess = tf.Session()

##############add########
c = tf.add(a,b)
d = a + b
print(sess.run(c))
print(sess.run(d))

############subtraction#######
c = tf.subtract(a,b)
d = a-b
print(sess.run(c))
print(sess.run(d))

########multiply###########
c = tf.multiply(a,b)
d = a*b
print(sess.run(c))
print(sess.run(d))

#######divide############
c = tf.divide(a,b)
d = a/b
print(sess.run(c))
print(sess.run(d))
sess.close()

#################matrix####################
a = tf.constant([[1,2],[2,3]], dtype=tf.float32)
b = tf.constant([[3,4],[4,5]], dtype=tf.float32)
sess = tf.Session()

###########add###############
c = tf.add(a, b)
d = a+b
print(sess.run(c))
print(sess.run(d))

##########subtract##########
c = tf.subtract(a, b)
d = a-b
print(sess.run(c))
print(sess.run(d))

###########multiply##########
a = tf.constant([[1,2],[2,3]], dtype=tf.float32)
b = tf.constant([[3,4],[4,5]], dtype=tf.float32)
c = tf.multiply(a, b)
d = a*b
e = tf.matmul(a, b)
print(sess.run(c))
print(sess.run(d))
print(sess.run(e))
f = tf.matmul(b,a)
print(sess.run(f))
###############divide###########
c = tf.divide(a, b)
d = a/b
e = tf.matrix_inverse(c)#############求逆矩阵########
f = tf.matmul(c,e)
print(sess.run(c))
print(sess.run(d))
print(sess.run(f))

#####################求最大值处的索引################
a = tf.constant([[1,2,3],[0,1,0]])
with tf.Session() as sess:
    print(sess.run(tf.argmax(a,1)))

#####################取模运算####################
c = tf.mod(10,3)
print(sess.run(c))
sess.close()
