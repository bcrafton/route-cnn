
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=1000)

from lib.Model import Model

from lib.Layer import Layer 
from lib.ConvToFullyConnected import ConvToFullyConnected
from lib.FullyConnected import FullyConnected
from lib.Convolution import Convolution
from lib.MaxPool import MaxPool
from lib.BatchNorm import BatchNorm

from lib.Activation import Activation
from lib.Activation import Relu
from lib.Activation import Linear

class PPOModel:
    def __init__(self, sess, nbatch, nclass, epsilon, decay_max, lr=2.5e-4, eps=1e-2, alg='bp', restore=None, train=1):

        self.sess = sess
        self.nbatch = nbatch
        self.nclass = nclass
        self.epsilon = epsilon
        self.decay_max = decay_max
        self.lr = lr
        self.eps = eps
        self.alg = alg
        self.restore = restore
        self.train_flag = train

        ##############################################

        self.states = tf.placeholder("float", [None, 84, 84, 4])
        self.advantages = tf.placeholder("float", [None])
        self.rewards = tf.placeholder("float", [None]) 
        
        self.old_actions = tf.placeholder("int32", [None])
        self.old_values = tf.placeholder("float", [None]) 
        self.old_nlps = tf.placeholder("float", [None])

        ##############################################
        with tf.variable_scope('l1'):
            conv1 = tf.layers.conv2d(self.states, 32, 8, 4, activation=tf.nn.relu, name='conv1')
            flat1 = tf.layers.flatten(conv1)
            self.value1 = tf.squeeze(tf.layers.dense(flat1, 1), axis=-1)
        
        with tf.variable_scope('l2'):
            conv2 = tf.layers.conv2d(conv1, 64, 4, 2, activation=tf.nn.relu, name='conv2')
            flat2 = tf.layers.flatten(conv2)
            self.value2 = tf.squeeze(tf.layers.dense(flat2, 1), axis=-1)

        with tf.variable_scope('l3'):
            conv3 = tf.layers.conv2d(conv2, 64, 3, 1, activation=tf.nn.relu, name='conv3')
            flat3 = tf.layers.flatten(conv3)
            self.value3 = tf.squeeze(tf.layers.dense(flat3, 1), axis=-1)

        with tf.variable_scope('l4'):
            flattened = tf.layers.flatten(conv3)
            fc = tf.layers.dense(flattened, 512, activation=tf.nn.relu, name='fc1')
            self.values = tf.squeeze(tf.layers.dense(fc, 1, name='values'), axis=-1)
            self.action_logits = tf.layers.dense(fc, 4, name='actions')

        self.action_dists = tf.distributions.Categorical(logits=self.action_logits)
        self.pi = self.action_dists
        
        ##############################################
        
        if self.train_flag:
            self.actions = tf.squeeze(self.pi.sample(1), axis=0)
        else:
            self.actions = self.pi.mode()

        self.nlps1 = self.pi.log_prob(self.actions)
        self.nlps2 = self.pi.log_prob(self.old_actions)

        ##############################################

        global_step = tf.train.get_or_create_global_step()
        epsilon_decay = tf.train.polynomial_decay(self.epsilon, global_step, self.decay_max, 0.001)

        ##############################################

        ratio = tf.exp(self.nlps2 - self.old_nlps)
        ratio = tf.clip_by_value(ratio, 0, 10)
        surr1 = self.advantages * ratio
        surr2 = self.advantages * tf.clip_by_value(ratio, 1 - epsilon_decay, 1 + epsilon_decay)
        policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

        entropy_loss = -tf.reduce_mean(self.pi.entropy())

        clipped_value_estimate = self.old_values + tf.clip_by_value(self.values - self.old_values, -epsilon_decay, epsilon_decay)
        value_loss_1 = tf.squared_difference(clipped_value_estimate, self.rewards)
        value_loss_2 = tf.squared_difference(self.values, self.rewards)
        value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss_1, value_loss_2))

        ##############################################

        clipped_value_estimate = self.old_values + tf.clip_by_value(self.value1 - self.old_values, -epsilon_decay, epsilon_decay)
        value_loss_1 = tf.squared_difference(clipped_value_estimate, self.rewards)
        value_loss_2 = tf.squared_difference(self.value1, self.rewards)
        value_loss1 = 0.5 * tf.reduce_mean(tf.maximum(value_loss_1, value_loss_2))                

        clipped_value_estimate = self.old_values + tf.clip_by_value(self.value2 - self.old_values, -epsilon_decay, epsilon_decay)
        value_loss_1 = tf.squared_difference(clipped_value_estimate, self.rewards)
        value_loss_2 = tf.squared_difference(self.value2, self.rewards)
        value_loss2 = 0.5 * tf.reduce_mean(tf.maximum(value_loss_1, value_loss_2))
        
        clipped_value_estimate = self.old_values + tf.clip_by_value(self.value3 - self.old_values, -epsilon_decay, epsilon_decay)
        value_loss_1 = tf.squared_difference(clipped_value_estimate, self.rewards)
        value_loss_2 = tf.squared_difference(self.value3, self.rewards)
        value_loss3 = 0.5 * tf.reduce_mean(tf.maximum(value_loss_1, value_loss_2))
        
        ##############################################

        self.loss = policy_loss + 0.01 * entropy_loss + 1. * value_loss
        self.loss1 = value_loss1
        self.loss2 = value_loss2
        self.loss3 = value_loss3

        ##############################################

        if self.alg == 'bp':
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=self.eps).minimize(self.loss)
        elif self.alg == 'lel':
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=self.eps).minimize(self.loss, var_list=tf.trainable_variables('l4'))
            self.train_op1 = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=self.eps).minimize(self.loss1, var_list=tf.trainable_variables('l1'))
            self.train_op2 = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=self.eps).minimize(self.loss2, var_list=tf.trainable_variables('l2'))
            self.train_op3 = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=self.eps).minimize(self.loss3, var_list=tf.trainable_variables('l3'))
        else:
            assert (False)

        ##############################################

        global_step = tf.train.get_or_create_global_step()
        self.global_step_op = global_step.assign_add(1)
        
        ##############################################
        
        self.saver = tf.train.Saver()
        
        if self.restore:
            self.saver.restore(sess=self.sess, save_path=self.restore)
        
        '''
        self.get_conv1 = tf.get_variable('conv1')
        self.get_conv2 = tf.get_variable('conv1')
        self.get_conv3 = tf.get_variable('conv1')
        self.get_fc1 = tf.get_variable('fc1')
        self.get_values = tf.get_variable('values')
        self.get_actions = tf.get_variable('actions')
        '''
        ##############################################
        
    def save_weights(self, filename):
        # [conv1, conv2, conv3, fc1, values, actions] = self.sess.run([self.get_conv1, self.get_conv2, self.get_conv3, self.get_fc1, self.get_values, self.get_actions])
        # weights = {}
        
        self.saver.save(sess=self.sess, save_path=filename)
        
    def set_weights(self):
        self.sess.run(self.global_step_op, feed_dict={})

    ##############################################

    def predict(self, state):
        action, value, nlp = self.sess.run([self.actions, self.values, self.nlps1], {self.states:[state]})

        action = np.squeeze(action)
        value = np.squeeze(value)
        nlp = np.squeeze(nlp)
        
        return action, value, nlp
        
    ##############################################

    def train(self, states, rewards, advantages, old_actions, old_values, old_nlps):
        if self.alg == 'bp':
            self.sess.run([self.train_op], feed_dict={self.states:states, self.rewards:rewards, self.advantages:advantages, self.old_actions:old_actions, self.old_values:old_values, self.old_nlps:old_nlps})
        elif self.alg == 'lel':
            self.sess.run([self.train_op, self.train_op1, self.train_op2, self.train_op3], feed_dict={self.states:states, self.rewards:rewards, self.advantages:advantages, self.old_actions:old_actions, self.old_values:old_values, self.old_nlps:old_nlps})
        else:
            assert (False)

    ##############################################
        
        
        
        
