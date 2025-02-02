import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ToyMarket import ToyMarket

env = ToyMarket()

observations, infos = env.reset()

def random_policy(agent, observation):
    return env.action_space(agent).sample()

num_steps = 10
info = []
for step in range(num_steps):
    actions = {agent: random_policy(agent, observations[agent]) for agent in env.agents}

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
