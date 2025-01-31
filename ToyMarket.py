import gymnasium
from gymnasium import spaces

import numpy as np

from pettingzoo import ParallelEnv

class ToyMarket(ParallelEnv):
    metadata = {'name': 'toy_market'}

    def __init__(self,num_buyer = 2, num_seller = 2, money = [1000,1000,1000,1000]):
        self.num_buyer = num_buyer
        self.num_seller = num_seller
        #Defines possible agents(not actual agents in the enviornment)
        self.possible_agents = [f'buyer_{i}' for i in range(self.num_buyer)]
        self.possible_agents = self.possible_agents + [f'seller_{i}' for i in range(self.num_seller)]
        self.money = {}
        for i in len(money):
            self.money.update({self.possible_agents[i] : money[i]})

        self.render_mode = None
    def action_space(self, agent):
        #action space is the how we want our actions to look like. Currently the actions take the shape of a 1 by n box with the lowest
        #value being a 0 and the highest being the amount of money currently within that individual buyer's account(for buyers) or infinity(for sellers)
        #therefore an agent is able to offer or request a certain amount of money from each buyer/seller within the enviornment
        #change to tuple if we want to restrict the number of people an agent can communicate with(ie first element is the box, second element specifies the target(s))
        return {agent: spaces.Box(low = 0, high = np.inf, shape = (self.num_buyer,)) if 'seller' in agent #logic for seller action space
         else spaces.Box(low = 0, high = self.state[agent]['private'][0], shape = (self.num_buyer,))  #logic for buyer action space
        }
    def observation_space(self, agent):
        if 'buyer' in agent:
            obs = {agent: spaces.Dict({'private': spaces.Box(low = -np.inf, high = np.inf, shape = (1,5)), #currently represents the money(index 0), number of days on the market(index 1)
                                                                                #num buyers(index 2), num sellers(index 3), and number of houses owned(index 4)
                                                                                #note that money for a seller is the value of the house and for a buyer it is the
                                                                                #amount of money in their bank account
                                'public': spaces.Box(low = -np.inf, high = np.inf, shape = (self.num_seller,))
                                #public shows the latest offer by all sellers
        })
        }
        else:
            obs= {agent: spaces.Dict({'private': spaces.Box(low = -np.inf, high = np.inf, shape = (1,5)), 
                                    'public': spaces.Box(low = -np.inf, high = np.inf, shape = (self.num_buyer,))
                                    #shows the latest offer by all buyers                      
        })
        }
        return obs


    def reset(self):
        self.agents = self.possible_agents

        self.houses = np.ones((1,self.num_seller)) #there are as many houses as there are sellers

        observations = {agent:
                      np.array([self.money[0], 0, self.num_buyer, self.num_seller, 0]) if 'buyer' in agent #buyers start with no house
                      else np.array([self.money[0], 0, self.num_buyer, self.num_seller, 1]) #sellers start with one house
                      for agent in self.agents}
        self.state = observations
        self.bid = np.zeros((self.num_buyer, self.num_seller)) #matrix such that the ij element specifies the ith buyer's offer to the jth seller
        self.price = np.zeros((self.num_buyer, self.num_seller)) #matrix such that the ij element specifies the jth sellers's offer to the ith buyer
        infos = {agent: {} for agent in self.agents}
        return observations, infos
    
    def step(self, actions):
        rewards = {agent: 0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {} #might want to track number of bids for each seller/buyer, number of "improper" bids(bids that are more than a buyer has)
        
        for agent, action in actions.items():
            agent_id = int(agent.split('_')[1])
            if 'buyer' in agent: #buyer logic
                if sum(bids) >= self.state[agent]['private'][0]: #if a buyer offers more in total than it has
                    if self.state[agent][4] == 0: #if the buyer does not own a house
                        rewards[agent] += -10 #then reduce rewards of the buyer
                    bids = np.zeros((self.num_buyer,)) #then make bids equal to 0
                self.bid[agent_id,] = bids #update self.bid matrix
            else: #seller logic
                self.price[:,agent_id] = action #update self.price matrix
        
        