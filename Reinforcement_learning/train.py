#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append("../package")
from environment import Env
import numpy as np
import matplotlib.pyplot as plt
from ddpg import Agent, Buffer
import pandas as pd
import csv
import os
import time
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main():

    buffer_size = 100000
    total_reward = []
    search_region = 2      #length of a side of search region (Å)
    max_step = 200         #max step in one episord
    inverse_atom_num = 1   #number of inverse designed atom
    action_high = pd.read_csv('action_limit_high.csv', header=None)
    action_low = pd.read_csv('action_limit_low.csv', header=None)
    action_high = np.array(action_high)[0][:inverse_atom_num*2].reshape((1,-1))
    action_low = np.array(action_low)[0][:inverse_atom_num*2].reshape((1,-1))

    loss_min = 0.122    #if the error is less than 'loss_min',episord is terminated 
    total_atom_num = 12
    model_name = 'Cd6Se6/model_Cd6Se6.pkl'    #spectra predict model
    target_data = 'Cd6Se6/Cd6Se6_abs.csv'              #target structure and spectra data



    env = Env(model_name, target_data, max_step, loss_min, total_atom_num, inverse_atom_num,search_region)
    state_size = inverse_atom_num*3
    action_size = inverse_atom_num*2
    agent = Agent(state_size, action_size, action_high, action_low)
    buffer = Buffer(buffer_size)
    start = time.time()
    ###train DDPG agent##########
    for i_episode in range(5000):
        print("episode: %d" % i_episode)
        state = env.reset()
        total_reward_in_episode = 0
        done = False
        while not done:
            action = agent.choose_action(state, (1+i_episode*0.005))
            next_state, reward, done = env.step(action, state)
            total_reward_in_episode += reward
            transition = np.array([state, action, next_state, reward, done],dtype=object)
            state = next_state
            buffer.store(transition)
            agent.train(buffer.transitions)
        print("Episode finish---time steps: %d" % env.t)
        print("total reward: %d" % total_reward_in_episode)
        total_reward.append(total_reward_in_episode)
    print(time.time() - start)
    ################################

#    save_name = 'trained_parameter' 
#    agent.save(save_name)

    ###plot reward curve##########
    episode = 5000
    x_episode = np.linspace(1, episode, episode)

    plt.plot(x_episode, total_reward)
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.tick_params(top=1, right=1, direction='in')
    plt.show()

    rewards_mean = []
    rewards_std = []
    moving_size = 20
    for i in range(episode-moving_size):
        rewards_mean.append(np.mean(total_reward[i:(i+moving_size)]))
        rewards_std.append(np.std(total_reward[i:(i+moving_size)]))

    rewards_mean = np.array(rewards_mean)
    rewards_std = np.array(rewards_std)

    x_episode = np.linspace(1, episode-moving_size, episode-moving_size)

    fig, ax = plt.subplots()
    ax.plot(x_episode, rewards_mean)
    ax.fill_between(x_episode, rewards_mean+rewards_std, rewards_mean-rewards_std, facecolor='blue', alpha=0.2)
    ax.set_ylabel('Reward')
    ax.set_xlabel('Episode')
    ax.tick_params(top=1, right=1, direction='in')
    plt.show()
    ################################





if __name__ == '__main__':
    main()
