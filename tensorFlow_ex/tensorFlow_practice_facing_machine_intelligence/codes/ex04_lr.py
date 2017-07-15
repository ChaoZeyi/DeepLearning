import tensorflow as tf

##################return input data, including data and expected output#########
def inputs():
    weight_age = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25], \
    [63, 28], [72, 36], [79, 57], [75, 44], [27, 24], [89, 31], [65, 52], \
    [57, 23], [59, 60], [69, 48], [60, 34], [79, 51], [75, 50], [82, 34], \
    [59, 46], [67, 23], [85, 37], [55, 40], [63, 30]]
    blood_fat_content = [[354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209,\
     290, 346, 254, 395, 434, 220, 374, 308, 220, 311, 181, 274, 303, 244]]
    return tf.to_float(weight_age), tf.to_float(blood_fat_content)
######################get output y accoring to model input######################
def inference(x):
    y = tf.matmul(x, w) + b
    return y
######################distance between expected output and model output#########
def loss(x, y):
    y_predicted = inference(x)
    losses = tf.reduce_sum(tf.squared_difference(y_predicted, y))
    return losses
##########################train process minimize loss###########################
def train(losses):
    optimizer = tf.train.GradientDescentOptimizer(0.0000001)
    train_ = optimizer.minimize(losses)
    return train_
################evaluate model accuracy using test data#########################
def evaluate(sess, x):
    return (sess.run(inference(x)))

x = tf.placeholder(dtype=tf.float32, shape=[None,2])
y = tf.placeholder(dtype=tf.float32, shape=[None,1])
w = tf.Variable(tf.ones([2,1]), dtype=tf.float32)
b = tf.Variable(1, dtype=tf.float32)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
a = sess.run(inputs())
a[1]
y_ = sess.run(tf.transpose(a[1]))

train_ = train(loss(x, y))

for i in range(500):
    sess.run(train_, {x:a[0], y:y_})
print(sess.run((inference([[84.0,46.0]]))))
print(sess.run(w))
print(sess.run(b))
