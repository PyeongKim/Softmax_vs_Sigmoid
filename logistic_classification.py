import tensorflow as tf
import numpy as np

tf.set_random_seed(787)  # for reproducibility
xy = np.loadtxt('wdbc.data.txt', delimiter=',', dtype=np.float64)


x_data = xy[:, 2:-1]
y_data = xy[:, [1]]
num_variable = x_data.shape[1]



print(x_data.shape, y_data.shape)

X = tf.placeholder(dtype=tf.float32, shape=[None, num_variable], name="input")
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="label")

W = tf.Variable(tf.random_normal(shape=[num_variable, 1]), name="weight")
b = tf.Variable(tf.random_normal(shape=[1]),  name="bias")

learning_rate = 0.01
epoch = 10000
logit = tf.matmul(X, W) + b
hypothesis = tf.sigmoid(logit)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for steps in range(epoch):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if steps % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X: x_data, Y: y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.3%}".format(
                steps, loss, acc))
    print("The Accuracy of model is {:.3%}".format(acc))

