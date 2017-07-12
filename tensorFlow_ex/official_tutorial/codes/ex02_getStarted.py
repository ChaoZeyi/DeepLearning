import tensorflow as tf


###################常量#############################
sess = tf.Session()
node1 = tf.constant(3.0, dtype=tf.float64)
node2 = tf.constant(4.0)
print(node1, node2)
node3 = tf.constant(5.0)
print node3
print sess.run(node3)
node4 = tf.add(node2, node3)
print node4
print ("node4:", node4)
print sess.run(node4)


####################占位符##############################
a = tf.placeholder(dtype=tf.float32)
b = tf.placeholder(dtype=tf.float32)
adder_node = a+b
print sess.run(adder_node, {a:3, b:4.5})
print sess.run(adder_node, {a:[1, 2], b:[3,4]})


######################变量################################
w = tf.Variable(0.3, dtype=tf.float32)
w
b = tf.Variable(-0.3)
x = tf.placeholder(dtype=tf.float32)
linear_model = w*x + b

init = tf.global_variables_initializer()
sess.run(init)
print sess.run(w)
print sess.run(linear_model, {x:[1,2,3,4]})

y = tf.placeholder(dtype=tf.float32)
squared_delta = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_delta)
loss
print sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]})

a = tf.assign(w, -1)
print sess.run(w)

c = tf.assign(b, 1)
sess.run([a, c])

print sess.run(w+b)
print sess.run(w)

print sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]})


####################train######################
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init)
print sess.run(w)
for i in range(1000):
    sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
print sess.run([w,b])
print sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]})
sess.close()
