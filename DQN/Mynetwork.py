'''
This is a 2-layer Deep Q-learning network, equipped with technique like Double Q learning, 
prioritized experience replay, aimed at solve open ai gym problems.

reference:
[1] Human-level control through deep reinforcementlearning. http://www.readcube.com/articles/10.1038/nature14236
[2] Deep Reinforcement Learning with Double Q-learning. https://arxiv.org/abs/1509.06461
[3] Prioritized Experience Replay. https://arxiv.org/abs/1511.05952

Author: Ziyang Zhang
'''

import tensorflow as tf
import numpy as np
from PrioritizedMemory import *


class DeepQNetwork:
    def __init__(self,
                 actionNum,
                 featureNum,
                 networkSize=20,
                 learningRate=0.01,
                 discountRate=0.99,
                 epsilon=0.8,
                 epsilonMin=0,
                 epsilonDecay=0.005,
                 memorySize=2048,
                 batchSize=100,
                 replayPeriod=5,
                 undatePeriod=50,
                 tensorboard=False):
        
        self.nA = actionNum
        self.nF = featureNum
        self.lr = learningRate
        self.netS = networkSize
        self.gamma = discountRate
        self.epsilonMax = epsilon
        self.epsilonMin = epsilonMin
        self.epsilon = epsilon
        self.epsilonD = epsilonDecay # control the speed of epsilon decay
        self.replayPeriod = replayPeriod # frequncy of learning with respect to action and observation
        self.batchS = batchSize # then every transition data is studied (batchS/updatePeriod) times 
                                # in average, if sampled uniformally
        self.memoryS = memorySize # must be power of two in this network    
        self.updateP = undatePeriod # of target network
        self.tensorB = tensorboard
        self.memory = Memory(self.memoryS, self.nF)        
        self.alpha = 0.6 # prioritized experience replay params, for priority utilitiy, [3]
        self.beta = 0.4 # prioritized experience replay params, for importance sampling weight,[3]
        self.learningCounter = 0
        self.sess = tf.Session()
        self._bulid_networks() 
        self.sess.run(tf.global_variables_initializer())
          
        if self.tensorB:
            # tensorboard --logdir=logs
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter("logs/", self.sess.graph)
            
    def _bulid_networks(self):
        self.states = tf.placeholder(tf.float32, [None, self.nF], name='state') 
        self.statesNext = tf.placeholder(tf.float32, [None, self.nF], name='stateNext')
        self.targetsHolder = tf.placeholder(tf.float32, [None, self.nA], name='targetsHolder')
        self.ISweights = tf.placeholder(tf.float32, [None, 1], name='ISweights')
        # online network
        self.onlineOutputs, self.onlineC = \
        self._initial_network(self.states, 'onlineOutputs', 'onlineParams')
        # target network
        self.targetOutputs, self.targetC = \
        self._initial_network(self.statesNext, 'targetOutputs', 'targetParams')
        # define training op
        with tf.name_scope('loss'):         
            self.loss = tf.reduce_mean(self.ISweights*tf.squared_difference(self.targetsHolder, self.onlineOutputs))
            tf.summary.scalar('loss', self.loss)           
        with tf.name_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
        # define target network update operation
        onlineParams = tf.get_collection('onlineParams')
        targetParams = tf.get_collection('targetParams')     
        self.update_target_op = [tf.assign(t, l) for t, l in zip(targetParams, onlineParams)]
                                             
    def _initial_network(self, inputs, variable_scope, collection_name):
        ''' build a two layer network.
            return, output tensor and collection of parameters
        '''
        w_initializer = tf.random_normal_initializer(0, 0.1)
        b_initializer = tf.constant_initializer(0.1)          
        with tf.variable_scope(variable_scope):
            collection = [collection_name, tf.GraphKeys.GLOBAL_VARIABLES]
            with tf.variable_scope('l1'):        
                W1 = tf.get_variable('w1', [self.nF, self.netS], tf.float32, 
                                     initializer=w_initializer, collections = collection)
                b1 = tf.get_variable('b1', [1, self.netS], tf.float32,
                                     initializer=b_initializer, collections = collection)               
                l1 = tf.nn.leaky_relu(tf.matmul(inputs, W1) + b1)
                tf.summary.histogram('w1', W1)
                tf.summary.histogram('b1', b1)
            with tf.variable_scope('l2'):                
                W2 = tf.get_variable('w2', [self.netS, self.nA], tf.float32,
                                     initializer=w_initializer, collections = collection)
                b2 = tf.get_variable('b2', [1, self.nA], tf.float32, 
                                     initializer=b_initializer, collections = collection)             
                outputs = tf.matmul(l1, W2) + b2
                tf.summary.histogram('w2', W2)
                tf.summary.histogram('b2', b2)
                # I'm not sure if regularization is needed, 
                # but it doesn't seem to be helpful when I use it.
#            reg = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) 
        return outputs, collection
               
    def choose_action(self, observation):  
        observation = observation[np.newaxis,:]   
        # epsilon greedy
        if np.random.rand() > self.epsilon:
            actionValues = self.sess.run(self.onlineOutputs, feed_dict={self.states: observation})
            action = np.argmax(actionValues)
        else:
            action = np.random.randint(0, self.nA)      
        return action

    def store(self, s, a, r, s_, done):
        data = []
        data.append(np.hstack((s, [a, r], s_)).reshape(-1, 2*self.nF+2))
        data.append([done])
        predict = self.sess.run(self.onlineOutputs, feed_dict={self.states: data[0][:, :self.nF].reshape(-1,self.nF)})
        target = self._targets(data)
        best_action = np.argmax(predict, axis=1)[0]
        priority = (np.abs(target[0,best_action]-predict[0,a]) + 0.01) ** self.alpha
        data.append([priority])
        self.memory.store_data(data)

    def _targets(self, batch):
        batchS = len(batch[1])
        targetOutputs, onlineOutputs = \
        self.sess.run([self.targetOutputs, self.onlineOutputs],
                      feed_dict={self.statesNext: batch[0][:, -self.nF:], self.states: batch[0][:, :self.nF]})
        # set future reward of the final state to be 0
        for i in range(batchS):
            if batch[1][i] == True:
                targetOutputs[i, :] = np.zeros(self.nA)
        # double Q learning target,[2]
        targets = onlineOutputs.copy() # want to change the actions chosen and keep other actions unchanged
                                       # those unchanged will be subtracted and their position will be 0 in loss
        actionsToUpdate = batch[0][:, self.nF].astype(int)
        actionsTargets = np.argmax(onlineOutputs, axis=1)
        rewards = batch[0][:,self.nF+1] 
        batchIdx = np.arange(batchS)
        targets[batchIdx, actionsToUpdate] = \
                        self.gamma * targetOutputs[batchIdx, actionsTargets] + rewards    
        return targets
             
    def learn(self):
        # update target network to be same as online network
        if self.learningCounter % self.updateP == 0:
            self.sess.run(self.update_target_op)
            print('target network updated')
        # update priorities,[3]
        batch, batchRandomI = self.memory.sample_batch(self.batchS)
        targets = self._targets(batch)
        predictions = self.sess.run(self.onlineOutputs, feed_dict={self.states: batch[0][:, :self.nF]})
        priorities = (np.sum(np.abs(targets-predictions), axis=1) + 0.01) ** self.alpha
        self.memory.update_priorities(batchRandomI, priorities)
        # importance sampling weight, [3]
        ISW = ((self.memory.minPriority / priorities) ** self.beta).reshape(-1, 1)
        # train
        self.sess.run(self.train_op, \
            feed_dict = {self.targetsHolder: targets, 
                         self.states: batch[0][:, :self.nF], 
                         self.ISweights: ISW})
        self.learningCounter += 1
        self._reduce_epsilon()
        # log to tensorboard
        if self.tensorB and self.learningCounter % 3 == 0:
            s = self.sess.run(self.merged, feed_dict={self.targetsHolder: targets, 
                                                      self.states: batch[0][:, :self.nF], 
                                                      self.ISweights: ISW})
            self.writer.add_summary(s, self.learningCounter)
            
    def _reduce_epsilon(self):
        self.epsilon = self.epsilonMin + (self.epsilonMax-self.epsilonMin) * np.exp(-self.epsilonD*self.learningCounter)