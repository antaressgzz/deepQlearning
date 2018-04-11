
import gym
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
    
def learning(env , episode_num, alpha, discount_rate, epsilon, 
               learning_method='Q_learning'):
    
    nA = env.action_space.n
    q_table = defaultdict(lambda: np.zeros(nA))
    # Some data which will be used to analysis the result
    step_num_list = np.zeros(episode_num)
    step_reward_list = np.zeros(episode_num)
    
    for i in range(episode_num):
       
        state = env.reset()
        isdone = False
        # Deducing exploration
        epsilon /= i+1
        while not isdone:
            # Take next action, using epsilon greedy policy  
            if np.random.uniform() > epsilon:
                action = np.argmax(q_table[state])
            else:
                action = np.random.choice(np.arange(nA))
            next_state, r, isdone, _ = env.step(action)
            #Update the data
            step_num_list[i] += 1
            step_reward_list[i] += r
            # Update the q_table
            if learning_method == 'Q_learning':      
                best_next_action = np.argmax(q_table[next_state])
                target = r + discount_rate * q_table[next_state][best_next_action]
                delta = target - q_table[state][action]
            elif learning_method == 'SARSA':
                probs = epsilon * np.ones(nA) / nA
                probs[np.argmax(q_table[next_state])] += 1 - epsilon
                target = r + discount_rate * np.sum(q_table[next_state]*probs)
                delta = target - q_table[state][action]
            else: print('Cannot understand method.')
            # Update table
            q_table[state][action] +=  alpha * delta
            # Update state
            state = next_state
    
    return q_table, step_num_list, step_reward_list
                
env = gym.make('CliffWalking-v0')
episode_num = 500
alpha = 0.1
discount_rate = 0.9
epsilon = 0.2
q_table, step_num_list, step_reward_list = learning(env, episode_num, alpha, 
                                                    discount_rate, epsilon, 'SARSA')

plt.figure()
plt.subplot(211)
plt.plot(np.arange(episode_num), step_reward_list)
plt.xlabel('number of eqisode')
plt.ylabel('episode reward')

plt.subplot(212)
plt.plot(np.arange(episode_num), step_num_list)
plt.xlabel('number of eqisode')
plt.ylabel('episode length')

# Showing the learned best strategy from start
state = env.reset()
isdone = False
while not isdone:
    next_action = np.argmax(q_table[state])
    env.render()   
    state, r, isdone, _ = env.step(next_action)