'''
This tree data only structure stores the priority in the position corresponding to the 
outside memory, which is indicated by self.pointer. The actual data information is stored
in in outside memory.

reference:
[1] Prioritized Experience Replay. https://arxiv.org/abs/1511.05952
    
Author: Ziyang Zhang
'''

import numpy as np

__all__ = ['Memory']
        
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2 * self.capacity ) # discard the 0 position and the tree rooted 
                                                  # at position 1, for convinent of reference
        self.pointer = 0 # next position in memory to update new data, 
                         # self.pointer + capacity = next position in tree to update priority
        
    def sample_batch(self, batchSize):
        batchList = []
        boundaries = np.linspace(0, self.tree[1], batchSize+1)
        # divide the sum into intervals of equal length
        intervalLength = self.tree[1] / batchSize
        for i in range(batchSize):
            sample = self._sample_one(boundaries[i], intervalLength)
            batchList.append(sample)
        return batchList
        
    def _sample_one(self, start, intervalLength):
        # s is uniformally sampled on the interval
        s = np.random.rand() * intervalLength + start
        pointer = 1     
        while pointer < self.capacity:
            if s <= self.tree[2*pointer]:
                pointer *= 2
            else:
                s -= self.tree[2*pointer]
                pointer = 2 * pointer + 1
        pointer -= self.capacity
        return pointer
        
    def add_new(self, priority):
        treePointer = self.pointer + self.capacity
        difference = priority - self.tree[treePointer]
        self.tree[treePointer] = priority
        while treePointer > 1:    
            treePointer //= 2
            self.tree[treePointer] += difference      

    def update(self, batchList, priorityList):
        # counter for priorityList
        counter = 0
        for i in batchList:
            treePointer = i + self.capacity
            difference = priorityList[counter] - self.tree[treePointer]
            self.tree[treePointer] = priorityList[counter]
            counter += 1
            while treePointer > 1:
                treePointer //= 2
                self.tree[treePointer] += difference

class Memory:
    def __init__(self, memorySize, featuresNumber):       
        self.memoryS = memorySize
        self.nF = featuresNumber
        self.memory = [np.zeros((self.memoryS, 2*self.nF+2)), # trainsitions
                       np.array([False]*self.memoryS)] # if done
        self.tree = SumTree(self.memoryS)
        self.memoryCounter = 0
        self.minPriority = 0
        
    def sample_batch(self, batchSize):
        batchRandomI = self.tree.sample_batch(batchSize)
        batch = [self.memory[0][batchRandomI,:], self.memory[1][batchRandomI]]       
        return batch, batchRandomI
        
    def store_data(self, data):    
        self.memory[0][self.memoryCounter, :] = data[0]
        self.memory[1][self.memoryCounter] = data[1][0]
        self.tree.add_new(data[2][0])
        self.memoryCounter += 1
        self.tree.pointer = self.memoryCounter
        if self.memoryCounter == self.memoryS:
            # for calculating importance sampling weight, updated every round of update
            self.minPriority = np.amin(self.tree.tree[-self.memoryS:]) # not exactly but almost,
                                                                       # for reducing computation expense
            self.memoryCounter = 0
            self.tree.pointer = self.memoryCounter
            
    def update_priorities(self, batchList, priorityList):
        self.tree.update(batchList, priorityList)
        
    # for test purpose   
    def print_test(self, trueprobs):
        print('-----------------data memory-----------------')
        print(self.memory)
        print('-----------------SumTree---------------------')
        print(self.tree.tree)
        print('-----------------stored priority---------------------')
        print(self.tree.tree[-8:])
        b, bId = self.sample_batch(1000)
        print('-----------------sampled batch---------------------')
        random = np.random.choice(1000, 10)
        print(b[0][random])
        print('-----------------sampled batch idx---------------------')
        for i in random:
            print(bId[i], end=' ')
        print()
        unique, counts = np.unique(bId, return_counts=True)
        print('-----------------sampled probs---------------------')
        print(dict(zip(unique, counts/1000)))
        print('---------------true probs---------------------------')
        print('%.3f, '*8 % trueprobs)

# some test code
if __name__ == '__main__':
    m = Memory(8, 1)
    for i in range(8):
        m.store_data([[i, 0, 0, i], [True], [i+1]])
    p = (1/36,2/36,3/36,4/36,5/36,6/36,7/36,8/36)
    m.print_test(p)
    m.store_data([[10, 0, 0, 10], [False], [5]]) # after this call, the total priority should be 40
    p = (5/40,2/40,3/40,4/40,5/40,6/40,7/40,8/40)
    m.print_test(p)
    m.update_priorities([1,2,3,4,5,6,7], [5]*7) # now all data has same priority, 5
    p = (1/8,)*8
    m.print_test(p)