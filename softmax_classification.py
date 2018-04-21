import tensorflow as tf
import numpy as np

tf.set_random_seed(787)  # for reproducibility
xy = np.loadtxt('wdbc.data.txt', delimiter=',', dtype=np.float32)
print(xy.shape)

x_data = xy[:, 2:-1]
y_data = xy[:, [1]]
print(x_data.shape)
num_class = 2
num_variables = x_data.shape[1]



X = tf.placeholder(dtype=tf.float32, shape=[None, num_variables], name="input")
Y = tf.placeholder(dtype=tf.int32, shape=[None, 1], name="label")

Y_one_hot = tf.one_hot(Y, depth=num_class)
Y_one_hot = tf.reshape(Y_one_hot, [-1, num_class])

W = tf.Variable(tf.random_normal(shape=[num_variables, num_class]), dtype=tf.float32,name="weight")
b = tf.Variable(tf.random_normal(shape=[num_class]), dtype = tf.float32,name="bias")



learning_rate = 0.01
epoch = 10000
logit = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logit)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=Y_one_hot))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for steps in range(epoch):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if steps % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X: x_data, Y: y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.3%}".format(
                steps, loss, acc))
    print("The Accuracy of model is {:.3%}".format(acc))



