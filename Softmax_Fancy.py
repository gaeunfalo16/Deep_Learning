import tensorflow as tf # Calling tensorflow module
import numpy as np # Calling numpy module

xy = tf.loadtext('data-04-zoo.csv', delimeter = ',', dtype = np.float32) # loadtext

x_data = xy[:, 0:-1] # x_data from xy[first to end, first to end - 1]
y_data = xy[:, -1] # x_data from xy[first to end, end -1]

nb_classes = 7 # We need to classify from 7 categories

X = tf.placeholder(tf.float32, [None, 16]) 
Y = tf.placeholder(tf.int32, [None, 1])

Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshpae(Y_one_hot, [-1, nb_classes]) # -1 means everything

W = tf.Variable(tf.random_normal([16, nb_classes]), name = 'weight')
b = tf.Variable(tf.fandom_normal([nb_classes]), name = 'bias')

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

#Cross_Entropy
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1) # Find the maximum value from the array
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess :
	sess.run(tf.global_variables_initializer())

	for step in range (2001) :
		sess.run(optimizer, feed_dict = {X: x_data, Y: y_data})

		if step % 100 == 0 :
			loss, acc = sess.run([cost, accuracy], feed_dict = {X: x_data, Y: y_data})
			print("Step : {:5}\t Loss : {:.3f}\t Acc : {:.2%}".format(step, loss, acc))


	pred = sess.run(prediction, feed_dict = {X: x_data, Y: y_data})

	for p, y in zip(pred, y_data.flatten()) :
		print("[{}] prediction: {} True Y : {}".format(p == int(y), p, int(y)))





