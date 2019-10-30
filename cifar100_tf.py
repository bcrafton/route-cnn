
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'

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

epochs = 10
batch_size = 1
x = tf.placeholder(tf.float32, [None, 32 , 32 , 3])
y = tf.placeholder(tf.float32, [None, 100])

####################################

def batch_norm(x, f):
    gamma = tf.Variable(np.ones(shape=f), dtype=tf.float32)
    beta = tf.Variable(np.zeros(shape=f), dtype=tf.float32)

    mean = tf.reduce_mean(x, axis=[0,1,2])
    _, var = tf.nn.moments(x - mean, axes=[0,1,2])

    bn = tf.nn.batch_normalization(x=x, mean=mean, variance=var, offset=beta, scale=gamma, variance_epsilon=1e-3)
    return bn

def block(x, f1, f2, p):
    filters = tf.Variable(init_filters(size=[3,3,f1,f2], init='alexnet'), dtype=tf.float32)

    conv = tf.nn.conv2d(x, filters, [1,1,1,1], 'SAME')
    bn   = batch_norm(conv, f2)
    relu = tf.nn.relu(bn)
    pool = tf.nn.avg_pool(relu, ksize=[1,p,p,1], strides=[1,p,p,1], padding='SAME')
    return pool

def dense(x, size):
    input_size, output_size = size

    w = tf.Variable(init_matrix(size=size, init='alexnet'), dtype=tf.float32)
    b  = tf.Variable(np.zeros(shape=output_size), dtype=tf.float32)

    fc = tf.matmul(x, w) + b
    return fc

####################################

block1 = block(x,       3, 32, 2) # 32 -> 16
block2 = block(block1, 32, 64, 2) # 16 -> 8
block3 = block(block2, 64, 64, 1) #  8 -> 8

block4 = block(block3, 64, 64, 2) #  8 -> 4
pool   = tf.nn.avg_pool(block4, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME') # 4 -> 1
flat   = tf.reshape(pool, [1, 64])
route  = tf.squeeze(dense(flat, [64, 20]))

train_idx = tf.squeeze(tf.distributions.Categorical(logits=route).sample(1))
test_idx = tf.cast(tf.argmax(route), dtype=tf.int32)

####################################

experts = []
for e in range(20):
    expert_block1 = block(block3,        64, 64, 1) # 8 -> 8
    expert_block2 = block(expert_block1, 64, 64, 2) # 8 -> 4
    expert_block3 = block(expert_block2, 64, 64, 1) # 4 -> 4

    pool  = tf.nn.avg_pool(expert_block3, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME') # 4 -> 1
    flat  = tf.reshape(pool, [1, 64])
    pred  = dense(flat, [64, 100])
    
    experts.append(pred)

####################################

branch_fns = {}
for e in range(20):
    branch_fns[e] = lambda: experts[e]

train_out = tf.switch_case(branch_index=train_idx, branch_fns=branch_fns)
test_out = tf.switch_case(branch_index=test_idx, branch_fns=branch_fns)

####################################

correct = tf.equal(tf.argmax(test_out, axis=1), tf.argmax(y, 1))
sum_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=train_out)
params = tf.trainable_variables()
grads = tf.gradients(loss, params)
grads_and_vars = zip(grads, params)
train = tf.train.AdamOptimizer(learning_rate=1e-1, epsilon=1.).apply_gradients(grads_and_vars)

####################################

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

####################################

for ii in range(epochs):

    idxs = [0] * 20
    for jj in range(0, 50000, batch_size):
        xs = np.reshape(x_train[jj], (1, 32, 32, 3))
        ys = np.reshape(y_train[jj], (1, 100))
        [i, _] = sess.run([train_idx, train], feed_dict={x: xs, y: ys})     
        idxs[i] += 1 
        if ((jj+1) % 1000 == 0):
            print ('%d/%d' % (jj+1, 50000))    
            print (idxs)

    total_correct = 0
    for jj in range(0, 10000, batch_size):
        xs = np.reshape(x_test[jj], (1, 32, 32, 3))
        ys = np.reshape(y_test[jj], (1, 100))
        _sum_correct = sess.run(sum_correct, feed_dict={x: xs, y: ys})
        total_correct += _sum_correct
        if ((jj+1) % 1000 == 0):
            print ('%d/%d' % (jj+1, 10000))

    '''
    param = sess.run(params, feed_dict={})

    for p in param:
      print (np.shape(p))

    np.save('cifar10_weights', param)       
    '''
  
    print ("acc: " + str(total_correct * 1.0 / 10000))
        
####################################








