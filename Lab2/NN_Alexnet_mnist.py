#coding = utf-8
from __future__ import print_function
 
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
ops.reset_default_graph()



learning_rate = 0.002
training_iters = 400000
batch_size = 64
display_step = 20


n_input = 784 
n_classes = 10 
dropout = 0.75 

###
from keras.datasets.mnist import load_data

(data,label),(test_data,test_label) = load_data()


data_size = len(data)
data = tf.cast(data, np.float32)
data = tf.nn.l2_normalize(data, dim = 1)
data = tf.reshape(data, [data_size, n_input])
label = tf.one_hot(label, n_classes)
label = tf.cast(label, np.int32)

test_size = len(test_data)
test_data = tf.cast(test_data, tf.float32)
test_data = tf.nn.l2_normalize(test_data, dim = 1)
test_data = tf.reshape(test_data, [test_size, n_input])
test_label = tf.one_hot(test_label, n_classes)
test_label = tf.cast(test_label, tf.int32)



###

lr_decay = 0.1
num_gens_to_wait = 480.
 
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
 
def conv2d(name, l_input, w, b):
	return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)
 
def max_pool(name, l_input, k):
	return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)
 
def norm(name, l_input, lsize=4):
	return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)
 
weights = {
    'wc1': tf.Variable(tf.random_normal([11, 11, 1, 64])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 64, 192])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 192, 384])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
    'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
    'wd1': tf.Variable(tf.random_normal([4*4*256, 4096])),
    'wd2': tf.Variable(tf.random_normal([4096, 4096])),
    'out': tf.Variable(tf.random_normal([4096, 10]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([192])),
    'bc3': tf.Variable(tf.random_normal([384])),
    'bc4': tf.Variable(tf.random_normal([384])),
    'bc5': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([4096])),
    'bd2': tf.Variable(tf.random_normal([4096])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
 
def alex_net(_X, _weights, _biases, _dropout):
	_X = tf.reshape(_X, shape=[-1, 28, 28, 1])
 
	conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
	pool1 = max_pool('pool1', conv1, k=2)
	norm1 = norm('norm1', pool1, lsize=4)
 
	conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
	pool2 = max_pool('pool2', conv2, k=2)
	norm2 = norm('norm2', pool2, lsize=4)
 
	conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
	norm3 = norm('norm3', conv3, lsize=4)
 
	conv4 = conv2d('conv4', norm3, _weights['wc4'], _biases['bc4'])
	norm4 = norm('norm4', conv4, lsize=4)
 
	conv5 = conv2d('conv5', norm4, _weights['wc5'], _biases['bc5'])
	pool5 = max_pool('pool5', conv5, k=2)
	norm5 = norm('norm5', pool5, lsize=4)
 
	dense1 = tf.reshape(norm5, [-1, _weights['wd1'].get_shape().as_list()[0]])
	dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1')
	dense1 = tf.nn.dropout(dense1, _dropout)
 
	dense2 = tf.reshape(dense1, [-1, _weights['wd2'].get_shape().as_list()[0]])
	dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation
	dense2 = tf.nn.dropout(dense2, _dropout)

	out = tf.matmul(dense2, _weights['out']) + _biases['out']
	return out
 
pred = alex_net(x, weights, biases, keep_prob)
 
# loss1 = tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y)[0]
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))


generation_num = tf.Variable(0, trainable=False)
model_learning_rate = tf.train.exponential_decay(learning_rate, generation_num, num_gens_to_wait, lr_decay, staircase=True)

optimizer = tf.train.AdamOptimizer(learning_rate=model_learning_rate).minimize(cost)
 
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
 
 
init = tf.initialize_all_variables()
 
saver = tf.train.Saver()

loss_col = []
train_acc_col = []
test_acc_col = []
step_jump_col = []
flag = False

with tf.Session() as sess:
	sess.run(init)
	data = sess.run(data)
	label = sess.run(label)
	test_data = sess.run(test_data)
	test_label = sess.run(test_label)
	step = 1
	# Keep training until reach max iterations
	while step * batch_size < training_iters:
		# batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		rand_index = np.random.choice(data_size, size=batch_size)
		batch_xs = data[rand_index]
		batch_ys = label[rand_index]

		sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout, generation_num:step})
		if step % display_step == 0:
			acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
			loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
			print ("Generation: " + str(step) + ", Batch Loss = " + "{:.6f}".format(loss) + ", Training Accuracy = " + "{:.5f}".format(acc))
			# pred_out = sess.run(pred[0], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
			# print ("Pred is " + str(pred_out))\
			# pred_out = sess.run(pred[0], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})

			# print ('y_out is ' + str( batch_ys[0]) )
			# print ("Pred is " + str(pred_out))
			rand_index = np.random.choice(test_size, size=256*2)
			rand_x = test_data[rand_index]
			rand_y = test_label[rand_index]
			acc_test = sess.run(accuracy, feed_dict={x: rand_x, y: rand_y, keep_prob: 1.})
			print ('Testing Accuracy: ' + str(acc_test) )

			step_jump_col.append(step)
			loss_col.append(loss)
			test_acc_col.append(acc_test)
			train_acc_col.append(acc)
			flag = False
			if (acc_test >= 0.965):
				rand_index = np.random.choice(test_size, size=256*2)
				rand_x = test_data[rand_index]
				rand_y = test_label[rand_index]
				acc_test = sess.run(accuracy, feed_dict={x: rand_x, y: rand_y, keep_prob: 1.})
				flag = (acc_test >= 0.965) and (step >= 2500)
		if (flag):
			break

		step += 1
		# if (step == 2):
	saver.save(sess,'D:\model\mnist_model')

	print ("Optimization Finished!")

plt.plot(step_jump_col, loss_col, 'b-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')

plt.savefig('./fig/mnist_loss')
plt.show()


plt.plot(step_jump_col, test_acc_col, 'b-')
plt.title('test_acc_col per Generation')
plt.xlabel('Generation')
plt.ylabel('test_acc_col')
plt.savefig('./fig/mnist_test_acc')


plt.plot(step_jump_col, train_acc_col, 'b-')
plt.title('train_acc_col per Generation')
plt.xlabel('Generation')
plt.ylabel('train_acc_col')
plt.savefig('./fig/mnist_train_acc')