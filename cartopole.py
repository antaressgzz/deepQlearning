import gym
from Mynetwork import DeepQNetwork
import numpy as np

env = gym.make('CartPole-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)


net = DeepQNetwork(actionNum=env.action_space.n,
                   featureNum=env.observation_space.shape[0],
                   tensorboard=True)

totalEpisode = 150
minMemorySize = 1000
exploreEpisode = 50
epsilon = net.epsilon

for episode in range(totalEpisode):
    observation = env.reset()
    episode_r = 0
    
    while True:
        env.render()      
        action = net.choose_action(observation)
        observation_, reward, done, _ = env.step(action)
        episode_r += reward
        net.store_data(observation, action, reward, observation_, done)        
        if net.memoryCounter > 1000:
            net.learn()           
        if done:
            print('episode: ', episode, 'reward: ', episode_r, 'epsilon: ', net.epsilon)
            break      
        observation = observation_     
    if episode >= exploreEpisode:
        net.reduce_epsilon((epsilon-0.01) / (totalEpisode-exploreEpisode))