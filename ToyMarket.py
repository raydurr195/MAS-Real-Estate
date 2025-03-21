import gymnasium
from gymnasium import spaces

import numpy as np

from pettingzoo import ParallelEnv
from pettingzoo.utils.wrappers import BaseParallelWrapper

class ToyMarket(ParallelEnv):
    metadata = {'name': 'toy_market'}

    def __init__(self, num_buyer = 2, num_seller = 2, money = [1500,1500,1000,1000], t = 5, max_day = 25):
        super().__init__()
        self.num_buyer = num_buyer
        self.num_seller = num_seller

        self.max_day = max_day
        #Defines possible agents(not actual agents in the enviornment)
        self.buyers = [f'buyer_{i}' for i in range(self.num_buyer)]
        self.sellers = [f'seller_{i}' for i in range(self.num_seller)]
        self.possible_agents = self.buyers + self.sellers
        self.t = t #the number of bids/prices sellers/buyers can see in the past
        self.money = {}
        self.iv = []
        for i in range(len(money)):
            self.money.update({self.possible_agents[i] : money[i]})
            if 'seller' in self.possible_agents[i]:
                self.iv.append(money[i])
        self.iv = np.array(self.iv).reshape(1, self.num_seller)
        self.render_mode = None

        self.buyer_obs_space =spaces.Dict({'private': spaces.Box(low = -np.inf, high = np.inf, shape = (1,4)), #currently represents the money(index 0), number of houses owned,
                                                                                #number of days w/o desired number of houses, desired number of houses
                                'public': spaces.Box(low = -np.inf, high = np.inf, shape = (2,self.t,self.num_seller))  
                                #public shows the latest t offer by all sellers, and the latest t bids to each seller
        })
        self.seller_obs_space = spaces.Dict({'private': spaces.Box(low = -np.inf, high = np.inf, shape = (1,3)), #currently reps the internal value of a house(index 0)
                                                                                                        #number of houses owned, number of days on the market,
                                    'public': spaces.Box(low = -np.inf, high = np.inf, shape = (2,self.t,self.num_buyer))
                                    #shows the latest offer t by all buyers, and the latest t prices sent to each seller           
        })
        self.buyer_act_space = spaces.Box(low = 0, high = money[0], shape = (self.num_seller,))
        self.seller_act_space = spaces.Box(low = 0, high = np.inf, shape = (self.num_buyer + 1,))


        

    def action_space(self, agent):
        #action space is the how we want our actions to look like. Currently the actions take the shape of a 1 by n box with the lowest
        #value being a 0 and the highest being the amount of money currently within that individual buyer's account(for buyers) or infinity(for sellers)
        #therefore an agent is able to offer or request a certain amount of money from each buyer/seller within the enviornment
        #change to tuple if we want to restrict the number of people an agent can communicate with(ie first element is the box, second element specifies the target(s))
        if "buyer" in agent:
            return spaces.flatten_space(self.buyer_act_space)
        else:
            return spaces.flatten_space(self.seller_act_space)

        
    def observation_space(self, agent):
        if "buyer" in agent:
            return spaces.flatten_space(self.buyer_obs_space)
        else:
            return spaces.flatten_space(self.seller_obs_space)


    def reset(self, *, seed = None, options = None):
        self.agents = self.possible_agents[:]
        self.houses = self.num_seller
        buyer_price_bid_mat = np.zeros((2,self.t,self.num_seller)) #first matrix is the prices of each house, second shows the latest offers(where there are t rows and self.num_seller columns)
        buyer_price_bid_mat[0,0,:] = 100000
        seller_bid_price_mat = np.zeros((2,self.t, self.num_buyer))
        seller_bid_price_mat[1,0,:] = 100000
        observations = {agent:
                      spaces.flatten(self.buyer_obs_space, {'private' : np.array([self.money[agent], 0, 0, 1]),
                       'public': buyer_price_bid_mat })
                      if 'buyer' in agent #buyers start with no house
                      else spaces.flatten(self.seller_obs_space,{'private': np.array([self.money[agent], 1, 0]), #sellers start with one house
                            'public' : seller_bid_price_mat
                      })
                      for agent in self.agents}
        self.state = observations
        self.bid = np.zeros((self.num_buyer, self.num_seller)) #matrix such that the ij element specifies the ith buyer's offer to the jth seller
        self.price = np.zeros((self.num_buyer, self.num_seller)) #matrix such that the ij element specifies the jth sellers's offer to the ith buyer
        self.improp = {agent: 0 for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        self.day = 1
        return observations, infos
    
    def step(self, actions):
        rewards = {agent: 0 for agent in self.agents}
        observations = self.state
        observations = {agent: spaces.unflatten(self.buyer_obs_space,obs) if 'buyer' in agent
                        else spaces.unflatten(self.seller_obs_space, obs)
                        for agent,obs in observations.items()}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {} #might want to track number of bids for each seller/buyer, number of "improper" bids(bids that are more than a buyer has)

        #collecting bids and prices
        for agent, act in actions.items():
            action = [spaces.unflatten(self.buyer_act_space, act) if 'buyer' in agent 
                      else spaces.unflatten(self.seller_act_space,act)][0]
            agent_id = int(agent.split('_')[1])
            if 'buyer' in agent: #buyer logic
                if np.sum(action) >= observations[agent]['private'][0,0]: #if a buyer offers more in total than it has
                    self.improp[agent] += 1
                    if observations[agent]['private'][0,1] == 0: #if the buyer does not own a house
                        rewards[agent] += -10 #then reduce rewards of the buyer
                    action = np.zeros((self.num_buyer,)) #then make bids equal to 0
                self.bid[agent_id,:] = action #update self.bid matrix
                observations[agent]['public'][1,:,:] = np.vstack([action,observations[agent]['public'][1,0:self.t-1,:]]) #might need to reshape action here
                #for seller in self.sellers:
                #    observations[seller]['public'][0,]
            else: #seller logic
                if (observations[agent]['private'][0,1] == 0) and np.any(action[0:self.num_buyer-1] != 0): #if the seller does not own a house and they've created a price
                    self.improp[agent] += 1
                    rewards[agent] += -10 #penalize them
                prices = action[0:len(action) - 1].reshape((self.num_buyer,)) #the -1 removes the desperation factor
                self.price[:,agent_id] = prices
                observations[agent]['public'][1,:,:] = np.vstack([prices,observations[agent]['public'][1,0:self.t-1,:]]) #might need to reshape action here
        #Finding potential bids for each seller
        good_bid = self.bid >= self.iv #marks True if a bid is greaater than or equal to the internal value of the house
        filt = [self.bid[good_bid[:,seller],seller] for seller in range(self.bid.shape[1])] #gets a list such that each element is an array of good bids for that sellers house
        #Loops through each seller and collection of potential bids
        for seller_id, bids in enumerate(filt):
            if bids.size > 0: #only perform function if there are some potential bids
                seller = self.sellers[seller_id] #note we used self.sellers since that is indexed at 0 for all sellers and we are here indexing at 0 for seller
                #self.agents indexes at 0 but includes buyers as well
                if observations[seller]['private'][0,1] > 0: #making sure sellers only sell houses if they have a house
                    buyer_id, accept = self.decide_sell(bids,seller_id,actions[seller]) #gets the action of the seller
                    buyer = self.buyers[buyer_id]
                    if accept == 1: #if the seller accepts the offer
                        observations[buyer]['private'][0,1] += 1 #add a house to the buyer
                        observations[buyer]['private'][0,2] = 0 #resets days w/o desired house
                        observations[seller]['private'][0,1] -= 1 #remove a house from the seller
                        observations[seller]['private'][0,2] = 0 #resets days on market
                        rewards[buyer] += 2*self.iv[0,seller_id] - np.max(bids)
                        rewards[seller] += np.max(bids) - self.iv[0,seller_id]
                        self.houses += -1
                        #if observations[buyer]['private'][0,1] == observations[buyer]['private'][0,3]: #if the number of houses == desired number of houses
                            #terminations[buyer] = True #terminate buyer
                        #if observations[seller]['private'][0,1] == 0: #if the seller has no more houses
                            #terminations[seller] = True #terminate seller


        infos = {agent: {'num days' : observations[agent]['private'][0,2],
                 'num improper bids': self.improp[agent],
                 'num house': observations[agent]['private'][0,1]}
                 for agent in self.agents}    
        
        #updating seller/buyer price/bid matrices and final rewards for not having desired number of houses/not selling a house
        for agent in self.agents:
            agent_id = int(agent.split('_')[1])
            if 'buyer' in agent:
                new_prices = self.price[agent_id,:]
                observations[agent]['public'][0,:,:] = np.vstack([new_prices, observations[agent]['public'][0,0:self.t-1,:]]) #updates prices index
                if observations[agent]['private'][0,1] < observations[agent]['private'][0,3]: #if num of houses is less than num of deisred houses
                    observations[agent]['private'][0,2] += 1 #adds another day without the number of desired houses
                    rewards[agent] += ((observations[agent]['private'][0,3]+1) / (observations[agent]['private'][0,1]+1))*(-5)*observations[agent]['private'][0,2] #scales punishment by num days w/o desired house and proportion of houses to desired houses              
                observations[agent] = spaces.flatten(self.buyer_obs_space, observations[agent])
            else: #seller  logic
                new_bids = self.price[:,agent_id].reshape(1,self.num_buyer)
                observations[agent]['public'][0,:,:] = np.vstack([new_bids,observations[agent]['public'][0,0:self.t-1,:]]) #updates bid index
                if observations[agent]['private'][0,1] > 0: #if the seller still owns a house
                    rewards[agent] += (-5)*(observations[agent]['private'][0,2]) #punish seller with scaled factor of days on market
                    observations[agent]['private'][0,2] += 1 #increase the number of days on the market
                observations[agent] = spaces.flatten(self.seller_obs_space,observations[agent])
        
        self.day += 1
        if self.day >= self.max_day or self.houses <= 0:
            truncations = {agent: True for agent in self.agents}
        return observations, rewards, terminations, truncations, infos




        
    def decide_sell(self, bids,seller_id, actions):
        #Takes in a bids which should by a np array, seller_id, and that sellers' actions
        num_buyer = len(bids)
        max_bid = np.max(bids)
        desp = actions[self.num_buyer]
        iv = self.iv[0,seller_id]
        buyers = np.where(self.bid[:,seller_id] == max_bid)[0] #gets buyer ids(to be used in self.buyers) of people who have max bid
        if len(buyers) > 1:
            buyer_id = np.random.choice(buyers)
        else: 
            buyer_id = buyers[0]

        accept = 1 #stochastic function here(returns 0 if not accept and 1 if accept)
        return buyer_id, accept



        
        
        





        
        
        