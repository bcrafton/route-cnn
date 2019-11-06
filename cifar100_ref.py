
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='4'

####################################

import numpy as np
import tensorflow as tf
import keras

from bc_utils.init_tensor import init_filters
from bc_utils.init_tensor import init_matrix

####################################

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

assert(np.shape(x_train) == (50000, 32, 32, 3))
x_train = x_train - np.mean(x_train, axis=0, keepdims=True)
x_train = x_train / np.std(x_train, axis=0, keepdims=True)
y_train = keras.utils.to_categorical(y_train, 100)

assert(np.shape(x_test) == (10000, 32, 32, 3))
x_test = x_test - np.mean(x_test, axis=0, keepdims=True)
x_test = x_test / np.std(x_test, axis=0, keepdims=True)
y_test = keras.utils.to_categorical(y_test, 100)

####################################

epochs = 25
batch_size = 50
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 100])

####################################

def batch_norm(x, f):
    gamma = tf.Variable(np.ones(shape=f), dtype=tf.float32)
    beta = tf.Variable(np.zeros(shape=f), dtype=tf.float32)

    mean = tf.Variable(np.zeros(shape=f), trainable=False, dtype=tf.float32)
    var = tf.Variable(np.zeros(shape=f), trainable=False, dtype=tf.float32)
    count = tf.Variable(0, trainable=False, dtype=tf.int64)

    next_mean = tf.reduce_mean(x, axis=[0,1,2])
    _, next_var = tf.nn.moments(x - next_mean, axes=[0,1,2])

    new_count = count + 1
    new_mean = ((tf.cast(count, tf.float32) * mean) + next_mean) / tf.cast(new_count, tf.float32)
    new_var = ((tf.cast(count, tf.float32) * var) + next_var) / tf.cast(new_count, tf.float32)

    update_mean = mean.assign(new_mean)
    update_var = var.assign(new_var)
    update_count = count.assign(new_count)

    bn = tf.nn.batch_normalization(x=x, mean=new_mean, variance=new_var, offset=beta, scale=gamma, variance_epsilon=1e-3)
    return bn, (update_mean, update_var, update_count)

def block(x, f1, f2, p):
    filters = tf.Variable(init_filters(size=[3,3,f1,f2], init='alexnet'), dtype=tf.float32)

    conv = tf.nn.conv2d(x, filters, [1,1,1,1], 'SAME')
    bn, update = batch_norm(conv, f2)
    relu = tf.nn.relu(bn)

    pool = tf.nn.avg_pool(relu, ksize=[1,p,p,1], strides=[1,p,p,1], padding='SAME')

    return pool, update

def dense(x, size):
    input_size, output_size = size
    w = tf.Variable(init_matrix(size=size, init='alexnet'), dtype=tf.float32)
    b  = tf.Variable(np.zeros(shape=output_size), dtype=tf.float32)
    fc = tf.matmul(x, w) + b
    return fc

####################################

block1, update1 = block(x,        3, 64,  1) # 32 -> 32
block2, update2 = block(block1,  64, 128, 2) # 32 -> 16

block3, update3 = block(block2, 128, 256, 1) # 16 -> 16
block4, update4 = block(block3, 256, 256, 2) # 16 ->  8

block5, update5 = block(block4, 256, 512, 1) #  8 ->  8
block6, update6 = block(block5, 512, 512, 2) #  8 ->  4

pool   = tf.nn.avg_pool(block6, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')  # 4 -> 1
flat   = tf.reshape(pool, [batch_size, 512])
out    = dense(flat, [512, 100])

# this only works with variables...
# ema = tf.train.ExponentialMovingAverage(decay=0.9999)
# avg = ema.apply([block1, block2])

####################################

predict = tf.argmax(out, axis=1)
correct = tf.equal(predict, tf.argmax(y, 1))
sum_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=out)
params = tf.trainable_variables()
grads = tf.gradients(loss, params)
grads_and_vars = zip(grads, params)
train = tf.train.AdamOptimizer(learning_rate=1e-2, epsilon=1.).apply_gradients(grads_and_vars)

####################################

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

####################################

for ii in range(epochs):
    for jj in range(0, 50000, batch_size):
        s = jj
        e = jj + batch_size
        xs = np.reshape(x_train[s:e], (batch_size,32,32,3))
        ys = np.reshape(y_train[s:e], (batch_size,100))
        sess.run([train, update1, update2, update3, update4, update5, update6], feed_dict={x: xs, y: ys})

    total_correct = 0

    for jj in range(0, 10000, batch_size):
        s = jj
        e = jj + batch_size
        xs = np.reshape(x_test[s:e], (batch_size,32,32,3))
        ys = np.reshape(y_test[s:e], (batch_size,100))
        _sum_correct = sess.run(sum_correct, feed_dict={x: xs, y: ys})
        total_correct += _sum_correct
  
    print ("acc: " + str(total_correct * 1.0 / 10000))
        
        
####################################








