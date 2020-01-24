#coding = utf-8
from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.framework import ops
from keras.preprocessing.image import ImageDataGenerator
ops.reset_default_graph()

batch_size = 100
step = 0
train_iter = 400000
traget_acc = 0.84
traget_step = 3000

argumentation = False

display_step = 10
def one_hot(label):
    l = len(label)
    out = np.zeros([l, 10])
    for i in range (l):
        ind = label[i]
        out[i][ind] = 1
    
    return out

from keras.datasets.cifar10 import load_data

(data,label),(test_data,test_label) = load_data()


data_size = len(data)

data = data / 255.0
data = data.astype(np.float32) 
label = one_hot(label)
label = label.astype(np.int32)


test_size = len(test_data)
test_data = test_data / 255.0
test_data = test_data.astype(np.float32)
test_label = one_hot(test_label)

test_label = test_label.astype(np.int32)    


input_x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)
is_traing = tf.placeholder(tf.bool)

if argumentation:
    datagen = ImageDataGenerator(horizontal_flip = True, vertical_flip = True, channel_shift_range = 0.2 )
    datagen.fit(data)
    data_iter = datagen.flow(data, y=label, batch_size=100)
    train_iter = train_iter*2
    traget_acc += 0.03
    traget_step = traget_step*2


 
####conv1
W1 = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=5e-2))
conv_1 = tf.nn.conv2d(input_x, W1, strides=(1, 1, 1, 1), padding="VALID")
print(conv_1)
bn1 = tf.layers.batch_normalization(conv_1, training=is_traing) 
relu_1 = tf.nn.relu(bn1)
print(relu_1)

pool_1 = tf.nn.max_pool(relu_1, strides=[1, 2, 2, 1], padding="VALID", ksize=[1, 3, 3, 1])
print(pool_1)

####conv2
W2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], dtype=tf.float32, stddev=5e-2))
conv_2 = tf.nn.conv2d(pool_1, W2, strides=[1, 1, 1, 1], padding="SAME")
print(conv_2)
bn2 = tf.layers.batch_normalization(conv_2, training=is_traing)
relu_2 = tf.nn.relu(bn2)
print(relu_2)
pool_2 = tf.nn.max_pool(relu_2, strides=[1, 2, 2, 1], ksize=[1, 3, 3, 1], padding="VALID")
print(pool_2)

####conv3

W3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256], dtype=tf.float32, stddev=1e-1))
conv_3 = tf.nn.conv2d(pool_2, W3, strides=[1, 1, 1, 1], padding="SAME")
print(conv_3)
bn3 = tf.layers.batch_normalization(conv_3, training=is_traing)
relu_3 = tf.nn.relu(bn3)
print(relu_3)
pool_3 = tf.nn.max_pool(relu_3, strides=[1, 2, 2, 1], ksize=[1, 3, 3, 1], padding="VALID")
print(pool_3)

#fc1
dense_tmp = tf.reshape(pool_3, shape=[-1, 2*2*256])
print(dense_tmp)
fc1 = tf.Variable(tf.truncated_normal(shape=[2*2*256, 1024], stddev=0.04))
bn_fc1 = tf.layers.batch_normalization(tf.matmul(dense_tmp, fc1), training=is_traing)
dense1 = tf.nn.relu(bn_fc1)
dropout1 = tf.nn.dropout(dense1, keep_prob)

#fc2
fc2 = tf.Variable(tf.truncated_normal(shape=[1024, 10], stddev=0.04))
out = tf.matmul(dropout1, fc2)
print(out)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)



correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
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
    # saver.restore(sess, "model_tmp/cifar10_demo.ckpt")
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < train_iter:
        step += 1
        rand_index = np.random.choice(data_size, size=batch_size)
        batch_xs = data[rand_index]
        batch_ys = label[rand_index]
        if argumentation: 
            batch_xs, batch_ys = data_iter.next()
        
        # print (batch_xs)
# 

        opt, acc, loss = sess.run([optimizer, accuracy, cost],
                                  feed_dict={input_x: batch_xs, y: batch_ys, keep_prob: 0.6, is_traing: True})
        if step % display_step == 0:
            print ("Generation: " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
            
            # pred_out = sess.run(out[0], feed_dict={input_x: batch_xs, y: batch_ys, keep_prob: 1.0, is_traing:True})
            # print ('y_out is ' + str( batch_ys[0]) )
            # print ("Pred is " + str(pred_out))
            
            rand_index = np.random.choice(test_size, size=batch_size)
            d = test_data[rand_index]
            l = test_label[rand_index]    
            acc_test = sess.run(accuracy, feed_dict={input_x: d, y: l, keep_prob: 1.0, is_traing: True})
            step_jump_col.append(step)
            loss_col.append(loss)
            test_acc_col.append(acc_test)
            train_acc_col.append(acc)
            print ("Testing Accuracy:", acc_test)
            flag = False
            if (acc_test >= traget_acc):
                rand_index = np.random.choice(test_size, size=batch_size)
                rand_x = test_data[rand_index]
                rand_y = test_label[rand_index]
                acc_test = sess.run(accuracy, feed_dict={input_x: rand_x, y: rand_y, keep_prob: 1.0, is_traing:True})
                flag = (acc_test >= traget_acc) and (step >= traget_step)
        if (flag):
            break
    mode = "model_tmp/cifar10_model"
    if argumentation:
        mode = "model_tmp_argumented/cifar10_model"
    saver.save(sess, mode)
    print ("Optimization Finished!")

p1 = './fig/cifar10_loss'
p2 = './fig/cifar10_test_acc'
p3 = './fig/cifar10_train_acc'
if argumentation:
    p1 = p1 + '_argumentation'
    p2 = p2 + '_argumentation'
    p3 = p3 + '_argumentation'

plt.plot(step_jump_col, loss_col, 'b-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.savefig(p1)
plt.show()

plt.plot(step_jump_col, test_acc_col, 'b-')
plt.title('test_acc_col per Generation')
plt.xlabel('Generation')
plt.ylabel('test_acc_col')
plt.savefig(p2)



plt.plot(step_jump_col, train_acc_col, 'r-')
plt.title('train_acc_col per Generation')
plt.xlabel('Generation')
plt.ylabel('train_acc_col')
plt.savefig(p3)
