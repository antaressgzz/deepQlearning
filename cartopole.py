import gym
from Mynetwork import DeepQNetwork
import numpy as np

env = gym.make('CartPole-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

#  end of episode indicator
finalIndicator = -100
net = DeepQNetwork(actionNum=env.action_space.n,
                   featureNum=env.observation_space.shape[0],
                   finalIndicator=finalIndicator,
                   networkSize=20,
                   tensorboard=True)

totalEpisode = 150
minMemorySize = 500
exploreEpisode = 50

for episode in range(totalEpisode):
    observation = env.reset()
    episode_r = 0
    
    while True:
        env.render()      
        action = net.choose_action(observation)
        observation_, reward, done, _ = env.step(action)
        episode_r += reward

        if done:
            observation_ = np.zeros(env.observation_space.shape[0])
            observation_[0] = finalIndicator
        net.store_data(observation, action, reward, observation_)      
        episode_r += reward
        
        if net.memoryCounter > 1000:
            net.learn()           
        if done:
            print('episode: ', episode, 'reward: ', episode_r, 'epsilon: ', net.epsilon)
            break      
        observation = observation_     

    if episode > exploreEpisode:
        net.reduce_epsilon((net.epsilonMax-net.epsilonMin) / (totalEpisode-exploreEpisode))

        