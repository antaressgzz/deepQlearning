import gym
from Mynetwork import DeepQNetwork
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('CartPole-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

net = DeepQNetwork(actionNum=env.action_space.n,
                   featureNum=env.observation_space.shape[0],
                   networkSize=20,
                   discountRate=0.95,
                   epsilon=0.3,
                   epsilonDecay=0.001,
                   replayPeriod=10,
                   undatePeriod=50,
                   batchSize=200,
                   tensorboard=True,
                   memorySize=2048)

totalEpisode = 500
rewardsRecords = []
memoryTotal = 0

for episode in range(totalEpisode):
    observation = env.reset()
    episodeRewards = 0
    episodeLength = 0
    counter = 0 # stop learn if get high score for several successive episodes
    while episodeLength < 1000:
        env.render()      
        action = net.choose_action(observation)
        observation_, reward, done, _ = env.step(action)        
        episodeRewards += reward
        net.store(observation, action, reward, observation_, done)      
        memoryTotal += 1
        if memoryTotal >= net.memoryS and memoryTotal % net.replayPeriod == 0 and counter <= 5:
            net.learn()      
        if done:
            print('episode: ', episode, 'reward: ', episodeRewards, 'epsilon: ', net.epsilon)
            break    
        observation = observation_
        episodeLength += 1
    if episodeRewards > 250:
        counter += 1
    else:
        counter = 0
    rewardsRecords.append(episodeRewards)

plt.plot(np.arange(len(rewardsRecords)), rewardsRecords)
plt.ylabel('Cost')
plt.xlabel('episode')
plt.show()