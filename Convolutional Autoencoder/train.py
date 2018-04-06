
# coding: utf-8

# https://github.com/tkuanlun350/Tensorflow-SegNet/blob/master/model.py
# https://ithelp.ithome.com.tw/articles/10188326

# In[1]:


import tensorflow as tf
import numpy as np
import random
import cv2
import os
import sys

print('pythpm : ',sys.version)
print('tensorflow : ',tf.__version__)


# In[2]:


# paremeters
img_size = 256


# In[3]:


img_size = 256
# load training data
def next_batch(batch_size):
    filenames = []
    for root, dirs, files in os.walk('./data/x'):
        for name in files:
            filenames.append(os.path.join(root, name).split('/')[-1])

    data_shape = (batch_size, img_size, img_size)
    X = np.zeros(data_shape)
    Y = np.zeros(data_shape)
    
    for i in range(batch_size):
        f = random.choice(filenames)
        img = np.array(cv2.imread('./data/x/' + f, 0))
        img2 = np.array(cv2.imread('./data/y/' + f, 0))
        X[i, :, :] = img
        Y[i, :, :] = img2
    
    X = X.reshape(batch_size, img_size*img_size)
    Y = Y.reshape(batch_size, img_size*img_size)
    return X, Y


# In[4]:


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial, name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial, name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')

def deconv2d(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    _, argmax = tf.nn.max_pool_with_argmax(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    pool = tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    return pool, argmax

def max_unpool_2x2(x, shape): # input shape
    inference = tf.image.resize_nearest_neighbor(x, tf.stack([shape[1]*2, shape[2]*2]))
    return inference


# In[5]:



tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape = [None, 256*256])
y = tf.placeholder(tf.float32, shape = [None, 256*256])
x_origin = tf.reshape(x, [-1, 256, 256, 1])
y_origin = tf.reshape(y, [-1, 256, 256, 1])

# conv1 256,1 > 256,64
W_e_conv1 = weight_variable([5, 5, 1, 64], "w_e_conv1") # filter, channel, features
b_e_conv1 = bias_variable([64], "b_e_conv1")
h_e_conv1 = tf.nn.relu(tf.add(conv2d(x_origin, W_e_conv1), b_e_conv1))
# pool1 256,64 > 128,64
h_e_pool1, argmax_e_pool1 = max_pool_2x2(h_e_conv1)

# conv2 128,64 > 128,128
W_e_conv2 = weight_variable([5, 5, 64, 128], "w_e_conv2")
b_e_conv2 = bias_variable([128], "b_e_conv2")
h_e_conv2 = tf.nn.relu(tf.add(conv2d(h_e_pool1, W_e_conv2), b_e_conv2))
# pool2 128,128 > 64,128
h_e_pool2, argmax_e_pool2 = max_pool_2x2(h_e_conv2)

# code 64,128
code_layer = h_e_pool2

# deconv1 64,128 > 64,64
W_d_conv1 = weight_variable([5, 5, 64, 128], "w_d_conv1")
output_shape_d_conv1 = tf.stack([tf.shape(x)[0], 64, 64, 64])
h_d_conv1 = tf.nn.sigmoid(deconv2d(code_layer, W_d_conv1, output_shape_d_conv1))
# unpool1 64,64 > 128,64
h_d_pool1 = max_unpool_2x2(h_d_conv1, [-1, 64, 64, 64]) # input size

# deconv2 128,64 > 128,1
W_d_conv2 = weight_variable([5, 5, 1, 64], "w_d_conv2")
output_shape_d_conv2 = tf.stack([tf.shape(x)[0], 128, 128, 1])
h_d_conv2 = tf.nn.sigmoid(deconv2d(h_d_pool1, W_d_conv2, output_shape_d_conv2))
# unpool 2 128,1 > 256,1
h_d_pool2 = max_unpool_2x2(h_d_conv2, [-1, 128, 128, 1])

x_reconstruct = h_d_pool2

print("input layer shape : %s" % x_origin.get_shape())
print("code layer shape : %s" % code_layer.get_shape())
print("reconstruct layer shape : %s" % x_reconstruct.get_shape())

# optimizer
with tf.name_scope('loss'):
    cost = tf.reduce_mean(tf.pow(x_reconstruct - y_origin, 2))
    tf.summary.scalar('loss', cost)
optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)


# In[11]:


# GPU config
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
#config = tf.ConfigProto(gpu_options=gpu_options)

#sess = tf.Session(config = config)
sess = tf.InteractiveSession()

# records
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)
saver = tf.train.Saver()

batch_size = 60
init_op = tf.global_variables_initializer()
sess.run(init_op)


for i in range(5000):
    #batch = mnist.train.next_batch(batch_size)
    batch_x, batch_y = next_batch(batch_size)
    if i%50 == 0: # loss logs
        rs = sess.run(merged,feed_dict={x:batch_x, y:batch_y})
        writer.add_summary(rs, i)
    if i%100 == 0: # print loss
        print("step %d, loss %g"%(i, cost.eval(feed_dict={x:batch_x, y:batch_y})))
    if i%1000 == 0: # save
        save_path = saver.save(sess, 'save/' + str(i) + '.ckpt')
        print('model saved')

    optimizer.run(feed_dict={x:batch_x, y:batch_y})
    
#print("final loss %g" % cost.eval(feed_dict={x: mnist.test.images}))

