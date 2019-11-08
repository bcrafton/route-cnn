
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

x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 100])
t = tf.placeholder(tf.bool, ())
b = tf.placeholder(tf.int32, ())

####################################

'''
def batch_norm(data, name):
    shape_param = data.get_shape()[-1]

    beta = tf.get_variable(name=name+'_beta', shape=shape_param, dtype=tf.float32, initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable(name=name+'_gamma', shape=shape_param, dtype=tf.float32, initializer=tf.constant_initializer(1.0, tf.float32))

    if FLAGS.train_mode:
        mean_param, variance_param = tf.nn.moments(x=data, axes=[0, 1, 2], name=name+'_moments')
        moving_mean = tf.get_variable(name=name+'_moving_mean', shape=shape_param, dtype=tf.float32, initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
        moving_variance = tf.get_variable(name=name+'_moving_variance', shape=shape_param, dtype=tf.float32, initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)

        mean = moving_averages.assign_moving_average(variable=moving_mean, value=mean_param, decay=0.9)
        variance = moving_averages.assign_moving_average(variable=moving_variance, value=variance_param, decay=0.9)
    else:
        mean = tf.get_variable(name=name+'_moving_mean', shape=shape_param, dtype=tf.float32, initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
        variance = tf.get_variable(name=name+'_moving_variance', shape=shape_param, dtype=tf.float32, initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)
        tf.summary.scalar(mean.op.name, mean)
        tf.summary.scalar(variance.op.name, variance)

    b_norm = tf.nn.batch_normalization(x=data, mean=mean, variance=variance, offset=beta, scale=gamma, variance_epsilon=0.001, name=name)
    return b_norm 
'''

def batch_norm(x, f, train, momentum=0.9):

    # definitely is the init for mean and var I think.
    # or atleast that is having negative effect.
    # well actually tbf the values end up being pretty similar.

    gamma = tf.Variable(np.ones(shape=f), dtype=tf.float32)
    beta = tf.Variable(np.zeros(shape=f), dtype=tf.float32)
    mean = tf.Variable(np.zeros(shape=f), trainable=False, dtype=tf.float32)
    var = tf.Variable(np.ones(shape=f), trainable=False, dtype=tf.float32)

    # next_mean = tf.reduce_mean(x, axis=[0,1,2])
    # _, next_var = tf.nn.moments(x - next_mean, axes=[0,1,2])

    next_mean, next_var = tf.nn.moments(x, axes=[0,1,2])

    # new_mean = next_mean 
    # new_var = next_var 
    new_mean = momentum * mean + (1. - momentum) * next_mean
    new_var = momentum * var + (1. - momentum) * next_var

    # new_var = tf.Print(new_var, [tf.reduce_mean(new_var), tf.reduce_mean(next_var)], message='', summarize=1000)
    # next_var = tf.Print(next_var, [tf.reduce_mean(new_var), tf.reduce_mean(next_var)], message='', summarize=1000)

    def batch_norm_train():
        update_mean = mean.assign(new_mean)
        update_var = var.assign(new_var)
        bn = tf.nn.batch_normalization(x=x, mean=new_mean, variance=new_var, offset=beta, scale=gamma, variance_epsilon=1e-3)
        return bn, (update_mean, update_var)

    def batch_norm_inference():
        update_mean = mean.assign(mean)
        update_var = var.assign(var)
        bn = tf.nn.batch_normalization(x=x, mean=next_mean, variance=next_var, offset=beta, scale=gamma, variance_epsilon=1e-3)
        return bn, (update_mean, update_var)

    return tf.cond(train, lambda: batch_norm_train(), lambda: batch_norm_inference())

def block(x, f1, f2, p, train):
    filters = tf.Variable(init_filters(size=[3,3,f1,f2], init='alexnet'), dtype=tf.float32)

    conv = tf.nn.conv2d(x, filters, [1,1,1,1], 'SAME')
    bn, update = batch_norm(conv, f2, train=train)
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

block1, update1 = block(x,        3, 64,  1, t) # 32 -> 32
block2, update2 = block(block1,  64, 128, 2, t) # 32 -> 16

block3, update3 = block(block2, 128, 256, 1, t) # 16 -> 16
block4, update4 = block(block3, 256, 256, 2, t) # 16 ->  8

block5, update5 = block(block4, 256, 512, 1, t) #  8 ->  8
block6, update6 = block(block5, 512, 512, 2, t) #  8 ->  4

pool   = tf.nn.avg_pool(block6, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')  # 4 -> 1
flat   = tf.reshape(pool, [b, 512])
out    = dense(flat, [512, 100])

# this only works with variables...
# ema = tf.train.ExponentialMovingAverage(decay=0.9999)
# avg = ema.apply([block1, block2])

####################################

predict = tf.argmax(out, axis=1)
tf_correct = tf.reduce_sum(tf.cast(tf.equal(predict, tf.argmax(y, 1)), tf.float32))

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

epochs = 25
train_batch_size = 50
test_batch_size = 500

for ii in range(epochs):
    for jj in range(0, 50000, train_batch_size):
        s = jj
        e = jj + train_batch_size
        xs = np.reshape(x_train[s:e], (train_batch_size,32,32,3))
        ys = np.reshape(y_train[s:e], (train_batch_size,100))
        sess.run([train, update1, update2, update3, update4, update5, update6], feed_dict={x: xs, y: ys, b: train_batch_size, t:True})

    total_correct = 0

    for jj in range(0, 10000, test_batch_size):
        s = jj
        e = jj + test_batch_size
        xs = np.reshape(x_test[s:e], (test_batch_size,32,32,3))
        ys = np.reshape(y_test[s:e], (test_batch_size,100))
        [np_correct, _, _, _, _, _, _] = sess.run([tf_correct, update1, update2, update3, update4, update5, update6], feed_dict={x: xs, y: ys, b: test_batch_size, t:False})
        total_correct += np_correct
  
    print ("acc: " + str(total_correct * 1.0 / 10000))
        
        
####################################








