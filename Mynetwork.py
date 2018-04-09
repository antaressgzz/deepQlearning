import tensorflow as tf
import numpy as np

class DeepQNetwork:
    def __init__(self,
                 actionNum,
                 featureNum,
                 networkSize=20,
                 learningRate=0.01,
                 discountRate=0.99,
                 epsilon=0.8,
                 memorySize=2000,
                 batchSize=50,
                 undatePeriod=300,
                 tensorboard=False):
        
        self.nA = actionNum
        self.nF = featureNum
        self.lr = learningRate
        self.netS = networkSize
        self.gamma = discountRate
        self.epsilon = epsilon
        self.updateP = undatePeriod 
        self.memoryS = memorySize
        self.batchS = batchSize
        self.tensorB = tensorboard
        
        self.memory = [np.zeros((self.memoryS, 2*self.nF+2)), np.array([False]*self.memoryS)]
        self.memoryCounter = 0
        self.learningCounter = 0
        self.sess = tf.Session()
        self._bulid_networks() 
        self.sess.run(tf.global_variables_initializer())
          
        if self.tensorB:
            # $ tensorboard --logdir=logs
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter("logs/", self.sess.graph)
            
    def _bulid_networks(self):
        self.states = tf.placeholder(tf.float32, [None, self.nF], name='state') 
        self.statesNext = tf.placeholder(tf.float32, [None, self.nF], name='stateNext')
        self.targetsHolder = tf.placeholder(tf.float32, [None, self.nA], name='targetsHolder')
        # online network
        self.onlineOutputs, self.onlineC = \
        self._initial_network(self.states, 'onlineOutputs', 'onlineParams')
        # target network
        self.targetOutputs, self.targetC = \
        self._initial_network(self.statesNext, 'targetOutputs', 'targetParams')
        tf.summary.tensor_summary('onlineOutputs', self.onlineOutputs)
        # define training op
        with tf.name_scope('loss'):         
            self.loss = tf.reduce_mean(tf.squared_difference(self.targetsHolder, self.onlineOutputs))
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
        w_initializer = tf.random_normal_initializer(0, 0.5)
        b_initializer = tf.constant_initializer(0.1)          
        with tf.variable_scope(variable_scope):
            collection = [collection_name, tf.GraphKeys.GLOBAL_VARIABLES]
            with tf.variable_scope('l1'):        
                W1 = tf.get_variable('w1', [self.nF, self.netS], tf.float32, 
                                     initializer=w_initializer, collections = collection)
                b1 = tf.get_variable('b1', [1, self.netS], tf.float32,
                                     initializer=b_initializer, collections = collection)               
                l1 = tf.nn.relu(tf.matmul(inputs, W1) + b1)
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
        
    def sampleBatch(self):
        if self.memoryCounter > self.memoryS:
            batch_random = np.random.choice(self.memoryS, size=self.batchS).astype(int)
        else:
            batch_random = np.random.choice(self.memoryCounter, size=self.batchS).astype(int)           
        batch = [self.memory[0][batch_random,:], self.memory[1][batch_random]]       
        return batch
        
    def store_data(self, s, a, r, s_, done):
        data = np.hstack((s, [a, r], s_))     
        update_index = self.memoryCounter % self.memoryS
        self.memory[0][update_index, :] = data
        self.memory[1][update_index] = done
        self.memoryCounter += 1
        
    def reduce_epsilon(self, difference):
        self.epsilon -= difference
             
    def learn(self):   
        # update target network to be same as online network
        if self.learningCounter % self.updateP == 0:
            self.sess.run(self.update_target_op)
            print('target network updated')
        # targets and predictions
        batch = self.sampleBatch()
        targetOutputs, onlineOutputs = \
        self.sess.run([self.targetOutputs, self.onlineOutputs],
                      feed_dict={self.statesNext: batch[0][:, -self.nF:], self.states: batch[0][:, :self.nF]})
        # set future reward of the final state to be 0
        for i in range(self.batchS):
            if batch[1][i] == True:
                targetOutputs[i, :] = np.zeros(self.nA)
        # double Q learning target
        targets = onlineOutputs.copy()
        actionsToUpdate = batch[0][:, self.nF].astype(int)
        actionsTargets = np.argmax(onlineOutputs, axis=1)
        rewards = batch[0][:,self.nF+1] 
        batchIdx = np.arange(self.batchS)
        targets[batchIdx, actionsToUpdate] = \
                        self.gamma * targetOutputs[batchIdx, actionsTargets] + rewards
        # train
        _ = self.sess.run(self.train_op, \
            feed_dict = {self.targetsHolder: targets, self.states: batch[0][:, :self.nF]})
        self.learningCounter += 1
        # log to tensorboard
        if self.tensorB and self.learningCounter % 5 == 0:
            s = self.sess.run(self.merged, \
                              feed_dict={self.targetsHolder: targets, self.states: batch[0][:, :self.nF]})
            self.writer.add_summary(s, self.learningCounter)
