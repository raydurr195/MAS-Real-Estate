
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ToyMarket import ToyMarket
from pettingzoo.test import parallel_api_test

env = ToyMarket()

#parallel_api_test(env, num_cycles= 1000)
observations, infos = env.reset()


num_episodes = 10
num_step = 100
info = []
for episode in range(num_episodes):
    for step in range(num_step):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        info.append(infos)
        print(f"Step {step}:")
        for agent in env.agents:
            #if 'buyer' in agent:
                #print(f"  {agent} - action: {actions[agent]}, sum: {sum(actions[agent])}")
            print(f"  {agent} - Reward: {rewards[agent]}, Terminated: {terminations[agent]}, Truncated: {truncations[agent]}, action: {actions[agent]}")
        # Check if all agents are done
        if all(terminations.values()) or all(truncations.values()):
            print("All agents are done. Resetting environment.")
            observations, infos = env.reset()
        print(f'bids: {env.bid}')
        print(f'price: {env.price}')
