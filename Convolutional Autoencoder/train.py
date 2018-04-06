
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
'''
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')
'''

def conv2d_layer(x, W_shape, b_shape, name, padding='SAME'):
    W = weight_variable(W_shape, name+'_W')
    b = bias_variable([b_shape], name+'_b')
    return tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding) + b)
'''
def deconv2d_(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides = [1, 1, 1, 1], padding = 'SAME')
 '''   
def deconv_layer(x, W_shape, b_shape, name, padding='SAME'):
    W = weight_variable(W_shape, name+'_W')
    b = bias_variable([b_shape], name+'_b')
    x_shape = tf.shape(x)
    out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])
    return tf.nn.conv2d_transpose(x, W, out_shape, [1, 1, 1, 1], padding=padding) + b

def max_pool_2x2_layer(x):
    #_, argmax = tf.nn.max_pool_with_argmax(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    pool = tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    return pool

def max_unpool_2x2_layer(x, shape): # input shape
    inference = tf.image.resize_nearest_neighbor(x, tf.stack([shape[1]*2, shape[2]*2]))
    return inference


# In[7]:



tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape = [None, 256*256], name='x')
y = tf.placeholder(tf.float32, shape = [None, 256*256], name='y')
x_origin = tf.reshape(x, [-1, 256, 256, 1])
y_origin = tf.reshape(y, [-1, 256, 256, 1])

# conv1 256
conv_1_1 = conv2d_layer(x_origin, [5, 5, 1, 64], 64, "conv_1_1", padding='SAME')
conv_1_2 = conv2d_layer(conv_1_1, [5, 5, 64, 64], 64, "conv_1_2", padding='SAME')
# pool1 256 > 128
pool_1 = max_pool_2x2_layer(conv_1_2)

# conv2 128
conv_2_1 = conv2d_layer(pool_1, [5, 5, 64, 128], 128, "conv_2_1", padding='SAME')
conv_2_2 = conv2d_layer(conv_2_1, [5, 5, 128, 128], 128, "conv_2_2", padding='SAME')
# pool2 128> 64
pool_2 = max_pool_2x2_layer(conv_2_2)

# conv3 64
conv_3_1 = conv2d_layer(pool_2, [5, 5, 128, 256], 256, "conv_3_1", padding='SAME')
conv_3_2 = conv2d_layer(conv_3_1, [5, 5, 256, 256], 256, "conv_3_2", padding='SAME')
# pool3 64 > 32
pool_3 = max_pool_2x2_layer(conv_3_2)

# code 16,512
code_layer = pool_3

# deconv3 32
deconv_3_2 = deconv_layer(code_layer, [5, 5, 256, 256], 256, 'deconv_3_2', padding='SAME')
deconv_3_1 = deconv_layer(deconv_3_2, [5, 5, 128, 256], 128, 'deconv_3_1', padding='SAME')
# unpool3 32 > 64
unpool_3 = max_unpool_2x2_layer(deconv_3_1, [-1, 32, 32, 128])    

# deconv2 64
deconv_2_2 = deconv_layer(unpool_3, [5, 5, 128, 128], 128, 'deconv_2_2', padding='SAME')
deconv_2_1 = deconv_layer(deconv_2_2, [5, 5, 64, 128], 64, 'deconv_2_1', padding='SAME')
# unpool2 64 > 128
unpool_2 = max_unpool_2x2_layer(deconv_2_1, [-1, 64, 64, 64])

# deconv1 128
deconv_1_2 = deconv_layer(unpool_2, [5, 5, 64, 64], 64, 'deconv_1_2', padding='SAME')
deconv_1_1 = deconv_layer(deconv_1_2, [5, 5, 1, 64], 1, 'deconv_1_1', padding='SAME')
# unpool1 128 > 256
unpool_1 = max_unpool_2x2_layer(deconv_1_1, [-1, 128, 128, 1])

x_reconstruct = unpool_1

result = tf.sigmoid(x_reconstruct, name='result')
result_round = tf.round(x_reconstruct, name='result_round')

print("input layer shape : %s" % x_origin.get_shape())
print("code layer shape : %s" % code_layer.get_shape())
print("output layer shape : %s" % result.get_shape())

# optimizer
with tf.name_scope('loss'):
    #cost = tf.reduce_mean(tf.pow(y_origin - result, 2))
    cost = tf.sqrt(tf.reduce_mean(tf.square(y_origin - result)))
    tf.summary.scalar('loss', cost)
optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)

with tf.name_scope('accuracy'):
    argmax_probs = tf.round(result)  # 0x1
    correct_pred = tf.cast(tf.equal(argmax_probs, y_origin), tf.float32)
    accuracy = tf.reduce_mean(correct_pred)
    tf.summary.scalar('accuracy', accuracy)


# In[6]:


# GPU config
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
#config = tf.ConfigProto(gpu_options=gpu_options)

#sess = tf.Session(config = config)
#sess = tf.InteractiveSession()
w1 = tf.placeholder("float", name="w1")

batch_size = 10

with tf.Session() as sess:
    # logs
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/", sess.graph)
    
    sess.run(tf.global_variables_initializer())
    
    # saver
    saver = tf.train.Saver()
    
    for i in range(5000):
        batch_x, batch_y = next_batch(batch_size)
        if i%50 == 0: # loss logs
            rs = sess.run(merged,feed_dict={x:batch_x, y:batch_y})
            writer.add_summary(rs, i)
        if i%100 == 0: # print loss
            print("step %d, loss %g, accuracy %g"%(i, cost.eval(feed_dict={x:batch_x, y:batch_y}), accuracy.eval(feed_dict={x:batch_x, y:batch_y})))
        if (i+1)%1000 == 0: # save
            saver.save(sess, 'save/model.ckpt')
            print('model saved')

        optimizer.run(feed_dict={x:batch_x, y:batch_y})

