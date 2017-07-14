import tensorflow as tf

##################return input data, including data and expected output#########
def inputs():
    weight_age = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25], \
    [63, 28], [72, 36], [79, 57], [75, 44], [27, 24], [89, 31], [65, 52], \
    [57, 23], [59, 60], [69, 48], [60, 34], [79, 51], [75, 50], [82, 34], \
    [59, 46], [67, 23], [85, 37], [55, 40], [63, 30]]
    blood_fat_content = [354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209,\
     290, 346, 254, 395, 434, 220, 374, 308, 220, 311, 181, 274, 303, 244]
    return weight_age, blood_fat_content
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
def train(loss):
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train = optimizer.minimize(loss)
    return train
################evaluate model accuracy using test data#########################
def evaluate(sess, x, y):
    print(sess.run())

w = tf.Variable(tf.zeros([2,1]), dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)
print(y)
