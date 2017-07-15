import tensorflow as tf

######################get combination output from input#########################
def combine_input(x):
    y = tf.matmul(x, w) + b
    return y
######################get output y accoring to model input######################
def inference(x):
    y = tf.sigmoid(combine_input(x))
    return y
######################distance between expected output and model output#########
def loss(x, y):
    y_predicted = inference(x)
    losses = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(combine_input(x), y))
    return losses
##########################train process minimize loss###########################
def train(losses):
    optimizer = tf.train.GradientDescentOptimizer(0.0000001)
    train_ = optimizer.minimize(losses)
    return train_
################evaluate model accuracy using test data#########################
def evaluate(sess, x):
    return (sess.run(inference(x)))

a = 0.8
with tf.Session() as sess:
    print(a>0.3)
