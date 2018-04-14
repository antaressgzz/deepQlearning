"""
@author: antaressgzz
"""

import gym
from Mynetwork import DeepQNetwork
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('MountainCar-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

net = DeepQNetwork(actionNum=env.action_space.n,
                   featureNum=env.observation_space.shape[0],
                   learningRate=0.1,
                   networkSize=10,
                   discountRate=0.99,
                   epsilon=0.5,
                   epsilonDecay=0.0005,
                   replayPeriod=5,
                   undatePeriod=500,
                   batchSize=100,
                   tensorboard=True,
                   memorySize=4096)

totalEpisode = 50
rewardsRecords = []
memoryTotal = 0

for episode in range(totalEpisode):
    observation = env.reset()
    episodeRewards = 0
    while True:
        env.render()      
        action = net.choose_action(observation)
        observation_, reward, done, _ = env.step(action)    
        if done:
            reward = 10
        episodeRewards += reward
        net.store(observation, action, reward, observation_, done)      
        memoryTotal += 1
        if memoryTotal >= net.memoryS and memoryTotal % net.replayPeriod == 0:
            net.learn()
            if net.learningCounter % net.updateP == 0:
                print('reward: ', episodeRewards, 'epsilon: ', net.epsilon)
        if done:           
            print('episode: ', episode,'reward: ', episodeRewards)
            break    
        observation = observation_
    rewardsRecords.append(episodeRewards)

plt.plot(np.arange(len(rewardsRecords)), rewardsRecords)
plt.ylabel('Rewards')
plt.xlabel('episode')
plt.show()