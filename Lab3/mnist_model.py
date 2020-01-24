#coding:utf-8
import os
import numpy as np
import scipy.misc
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data #as mnist_data
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
ops.reset_default_graph()

class MnistModel:

    def __init__(self):
        self.DEPTH = 28
        self.OUTPUT_SIZE = 28
        self.BATCH_SIZE = 64
        self.LAMBDA = 10
        self.EPOCH = 41
        self.LEARNING_RATE = 0.01
        self.BETA1 = 0.5
        self.BETA2 = 0.9

    def lrelu(self,name,x, leak=0.2):
        return tf.maximum(x, leak * x, name=name)

    def Discriminator(self,name,inputs,reuse):
        with tf.variable_scope(name, reuse=reuse):
            o = tf.reshape(inputs, [-1, 28, 28, 1])
            with tf.variable_scope('d_conv_1'):
                w1 = tf.get_variable('w1', [5, 5, o.get_shape()[-1],self.DEPTH], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))
                var1 = tf.nn.conv2d(o,w1,[1,2, 2,1],padding='SAME')
                b1 = tf.get_variable('b1', [self.DEPTH], 'float32',initializer=tf.constant_initializer(0.01))
                o1 = tf.nn.bias_add(var1, b1)

            o2 = self.lrelu('d_lrelu_1', o1)
            with tf.variable_scope('d_conv_2'):
                w2 = tf.get_variable('w2', [5, 5, o2.get_shape()[-1],2*self.DEPTH], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))
                var2 = tf.nn.conv2d(o2,w2,[1,2, 2,1],padding='SAME')
                b2 = tf.get_variable('b2', [2*self.DEPTH], 'float32',initializer=tf.constant_initializer(0.01))
                o3 = tf.nn.bias_add(var2, b2)

            o4 = self.lrelu('d_lrelu_2', o3)

            with tf.variable_scope('d_conv_3'):
                w3 = tf.get_variable('w3', [5, 5, o4.get_shape()[-1],4*self.DEPTH], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))
                var3 = tf.nn.conv2d(o4,w3,[1,2, 2,1],padding='SAME')
                b3 = tf.get_variable('b3', [4*self.DEPTH], 'float32',initializer=tf.constant_initializer(0.01))
                o5 = tf.nn.bias_add(var3, b3)

            o6 = self.lrelu('d_lrelu_3', o5)
            chanel = o6.get_shape().as_list()
            o7 = tf.reshape(o6, [self.BATCH_SIZE, chanel[1]*chanel[2]*chanel[3]])
            o8 = self.dense('d_fc', o7, 1)
            return o8

    def Generator(self,name, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            noise = tf.random_normal([self.BATCH_SIZE, 128])
            noise = tf.reshape(noise, [self.BATCH_SIZE, 128], 'noise')
            output = self.dense('g_fc_1', noise, 2*2*8*self.DEPTH)
            output = tf.reshape(output, [self.BATCH_SIZE, 2, 2, 8*self.DEPTH], 'g_conv')

            output = self.deconv2d('g_deconv_1', output, ksize=5, outshape=[self.BATCH_SIZE, 4, 4, 4*self.DEPTH])
            output = tf.nn.relu(output)
            output = tf.reshape(output, [self.BATCH_SIZE, 4, 4, 4*self.DEPTH])

            output = self.deconv2d('g_deconv_2', output, ksize=5, outshape=[self.BATCH_SIZE, 7, 7, 2* self.DEPTH])
            output = tf.nn.relu(output)

            output = self.deconv2d('g_deconv_3', output, ksize=5, outshape=[self.BATCH_SIZE, 14, 14, self.DEPTH])
            output = tf.nn.relu(output)

            output = self.deconv2d('g_deconv_4', output, ksize=5, outshape=[self.BATCH_SIZE, self.OUTPUT_SIZE, self.OUTPUT_SIZE, 1])
            output = tf.nn.sigmoid(output)
            return tf.reshape(output,[-1,784])


    def deconv2d(self,name, tensor, ksize, outshape, stddev=0.01, stride=2, padding='SAME'):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [ksize, ksize, outshape[-1], tensor.get_shape()[-1]], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=stddev))
            var = tf.nn.conv2d_transpose(tensor, w, outshape, strides=[1, stride, stride, 1], padding=padding)
            b = tf.get_variable('b', [outshape[-1]], 'float32', initializer=tf.constant_initializer(0.01))
            return tf.nn.bias_add(var, b)

    def save_images(self, images, size, path):
        img = (images + 1.0) / 2.0
        h, w = img.shape[1], img.shape[2]
        merge_img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            merge_img[j * h:j * h + h, i * w:i * w + w, :] = image
        return scipy.misc.imsave(path, merge_img)

    def dense(self, name,value, o_shape):
        with tf.variable_scope(name, reuse=None) as scope:
            shape = value.get_shape().as_list()
            w = tf.get_variable('w', [shape[1], o_shape], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))
            b = tf.get_variable('b', [o_shape], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            return tf.matmul(value, w) + b
    
    def acc_cnt(self, inp, size):
        cnt = 0
        for i in inp:
            if (i > 0.5):
                cnt += 1
        acc = float(cnt) / float(size)
        return acc

    def train(self):
        with tf.variable_scope(tf.get_variable_scope()):
            real_data = tf.placeholder(tf.float32, shape=[self.BATCH_SIZE,784])

            with tf.variable_scope(tf.get_variable_scope()):
                fake_data = self.Generator('gen',reuse=False)
                disc_real = self.Discriminator('dis_r',real_data,reuse=False)
                disc_fake = self.Discriminator('dis_r',fake_data,reuse=True)

            t_vars = tf.trainable_variables()
            d_vars = [var for var in t_vars if 'd_' in var.name]
            g_vars = [var for var in t_vars if 'g_' in var.name]

            gen_cost = tf.reduce_mean(disc_fake)
            disc_cost = -tf.reduce_mean(disc_fake) + tf.reduce_mean(disc_real)
            real_in_box = tf.reduce_mean(tf.sigmoid(disc_real) )
            fake_in_box = tf.reduce_mean(tf.sigmoid(disc_fake))

            alpha = tf.random_uniform(shape=[self.BATCH_SIZE, 1],minval=0.,maxval=1.)
            differences = fake_data - real_data
            interpolates = real_data + (alpha * differences)
            gradients = tf.gradients(self.Discriminator('dis_r',interpolates,reuse=True), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            disc_cost += self.LAMBDA * gradient_penalty

            with tf.variable_scope(tf.get_variable_scope(), reuse=None):
                gen_train_op = tf.train.GradientDescentOptimizer(learning_rate=self.LEARNING_RATE).minimize(gen_cost,var_list=g_vars)
                disc_train_op = tf.train.GradientDescentOptimizer(learning_rate=self.LEARNING_RATE).minimize(disc_cost,var_list=d_vars)

            saver = tf.train.Saver()

            # sess = tf.Session()
            sess = tf.InteractiveSession()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            if not os.path.exists('img'):
                os.mkdir('img')

            init = tf.global_variables_initializer()
            sess.run(init)
            mnist = input_data.read_data_sets("data", one_hot=True)
            cnt = 0
            step_jump_col = []
            g_loss_col = []
            d_loss_col = []
            gen_acc_col = []
            for EPOCH in range (1,self.EPOCH):
                idxs = 1000
                for iters in range(1, idxs):
                    img, _ = mnist.train.next_batch(self.BATCH_SIZE)

                    for x in range (0,5):
                        _, d_loss = sess.run([disc_train_op, disc_cost], feed_dict={real_data: img})
                    _, g_loss = sess.run([gen_train_op, gen_cost])
                    
                    if (iters % 10 == 0):
                        step_jump_col.append(cnt)
                        g_loss_col.append(g_loss)
                        d_loss_col.append(d_loss)
                        real_confidence = sess.run(real_in_box, feed_dict = {real_data:img})
                        fake_confidence = sess.run(fake_in_box)
                        gen_acc_col.append(fake_confidence)
                        print("( EPOCH:" + str(EPOCH) + "  iters: " + str(iters) +  ") d_loss: " + str(d_loss) + ", g_loss: " + str(g_loss) +  ", gen_acc: " + str(fake_confidence))
                    cnt +=1

                with tf.variable_scope(tf.get_variable_scope()):
                    samples = self.Generator('gen', reuse=True)
                    samples = tf.reshape(samples, shape=[self.BATCH_SIZE, 28,28,1])
                    samples=sess.run(samples)
                    self.save_images(samples, [8,8], os.getcwd()+'/img/'+'sample_%d_epoch.png' % (EPOCH))

                if EPOCH>=(self.EPOCH - 1):
                    checkpoint_path = os.path.join(os.getcwd(), 'generator_mnist')
                    saver.save(sess, checkpoint_path)

            # coord.request_stop()
            # coord.join(threads)

            p1 = './fig/d_loss'
            p2 = './fig/g_loss'
            p3 = './fig/gen_acc'
            plt.plot(step_jump_col, d_loss_col, 'b-')
            plt.title('d_loss per iter')
            plt.xlabel('iter')
            plt.ylabel('d_loss')
            plt.savefig(p1)
            plt.show()

            plt.plot(step_jump_col, g_loss_col, 'b-')
            plt.title('gen_loss per iter')
            plt.xlabel('iter')
            plt.ylabel('gen_loss')
            plt.savefig(p2)
            plt.show()



            plt.plot(step_jump_col, gen_acc_col, 'r-')
            plt.title('gen_acc per iter')
            plt.xlabel('iter')
            plt.ylabel('gen_acc')
            plt.savefig(p3)
            plt.show()

            coord.request_stop()
            coord.join(threads)
            sess.close()
if __name__ == '__main__':
    mnist = MnistModel()
    mnist.train()
    