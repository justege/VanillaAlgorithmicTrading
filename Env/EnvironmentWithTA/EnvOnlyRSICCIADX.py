from __future__ import division
import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
import csv


matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
# initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE = 1

# [100,-50,100]

# total number of stocks in our portfolio
STOCK_DIM = 3
# transaction fee: 1/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.0025
REWARD_SCALING = 10000


#[-0.7*HMAX_NORMALIZE, 0.5*HMAX_NORMALIZE,0.3*HMAX_NORMALIZE]
# w1, w2, w3,
class StockENVCCIRSIADX(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, day=0, initial=True, previous_state=[], model_name='', iteration=''):
        # super(StockEnv, self).__init__()
        # money = 10 , scope = 1
        self.day = day
        self.df = df

        # action_space normalization and shape is STOCK_DIM
        self.action_space = spaces.Box(low=0, high=1, shape=(STOCK_DIM + 1,))
        # Shape = 181: [Current Balance]+[prices 1-30]+[owned shares 1-30]
        # +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(13,))
        # load data from a pandas dataframe
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        # initalize state
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                     [0] * STOCK_DIM + \
                     self.data.rsi.values.tolist() + \
                     self.data.cci.values.tolist() + \
                     self.data.adx.values.tolist()

        self.PricesList = self.data.adjcp.values.tolist()
        # initialize reward
        self.reward = 0
        self.cost = 0
        # memorize all the total balance change
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self.trades = 0
        self.P_t_0 = 0
        # self.reset()
        self._seed()
        self.W_t_1 = [1,0,0,0]
        self.W_t =   [1,0,0,0]
        self.Yt = self.data.adjcp.values.tolist()
        self.P_t_1 =  1
        self.previous_state = previous_state
        self.initial = initial
        self.model_name = model_name
        self.iteration = iteration



    def make_actions(self, index, action):
        available_amount = (1 -  sum(np.array(self.state[:(index)])))
        if available_amount > 0:
            self.trades += 1
            self.state[index] = min(available_amount, action)



    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        # print(actions)

        if self.terminal:

            return self.state, self.reward, self.terminal, {}

        else:
            # print(np.array(self.state[1:29]))
            actions = actions

            # actions = (actions.astype(int))
            # print("actions {}".format(actions))

            # print("state before buying actions {}".format(self.state))
            v_t_1 = np.array([1] + self.data.adjcp.values.tolist())


            """
            for index in range(1,STOCK_DIM+1):
                # print('take buy action: {}'.format(actions[index]))
                #print("while buying actions {}".format(self.state))
                self.make_actions(index, actions[index])
            """
            numerator = np.exp(actions)
            denominator = np.sum(np.exp(actions))
            softmax_output = numerator / denominator

            # print(actions)
            # print(softmax_output)

            self.day += 1
            self.data = self.df.loc[self.day, :]

            self.state[0], self.state[1], self.state[2], self.state[3] = softmax_output



            # print("stock_shares:{}".format(self.state[29:]))
            self.state = [self.state[0]] + \
                         list(self.state[(1):(STOCK_DIM + 1)]) + \
                         self.data.rsi.values.tolist() + \
                         self.data.cci.values.tolist() + \
                         self.data.adx.values.tolist()

            v_t_0 =  np.array([1] + self.data.adjcp.values.tolist())

            #print("state after buying actions {}".format(self.state))

            self.W_t = np.array(self.state[:(STOCK_DIM + 1)])

            Y_t = np.divide(v_t_0,v_t_1)

            #print("-------------------")
            #print("Y_t:{}".format(Y_t))
            #print("P_t_1:{}".format(self.P_t_1))

            #print("self.P_t_1 * Y_t:{}".format(self.P_t_1 * Y_t))
            #print("W_t_1:{}".format(self.W_t_1))


            """
            Compare
            """
            self.cost = TRANSACTION_FEE_PERCENT * (
                np.abs(self.W_t_1[1:] - self.W_t[1:])).sum()


            self.P_t_0 = self.P_t_1 * (1 - self.cost) * np.dot(Y_t,self.W_t_1)


            #print("P_t_0:{}".format(self.P_t_0))

            self.W_t_1 = self.W_t

            self.P_t_0 = np.clip(self.P_t_0, 0, np.inf)



            self.asset_memory.append(self.P_t_0)

            self.reward = np.log((self.P_t_0 + (1e-7))/ (self.P_t_1 + (1e-7)))

            self.P_t_1 = self.P_t_0

            self.rewards_memory.append(self.reward)

            info = {'mean reward': np.mean(self.rewards_memory), 'value_portfolio': self.P_t_0, 'reward': self.reward, 'weights': self.W_t}





        return self.state, self.reward, self.terminal, info

    def reset(self):
        self.P_t_1 = 1
        self.P_t_0 = 0
        self.W_t_1 = [1, 0, 0, 0]
        self.W_t = [1, 0, 0, 0]
        self.Yt = self.data.adjcp.values.tolist()

        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []

        if self.initial:
            self.asset_memory = [INITIAL_ACCOUNT_BALANCE]

            # initiate state
            self.state = [INITIAL_ACCOUNT_BALANCE] + \
                         [0] * STOCK_DIM + \
                         self.data.rsi.values.tolist() + \
                         self.data.cci.values.tolist() + \
                         self.data.adx.values.tolist()

        else:
            previous_total_asset = self.previous_state[0] + \
                                   sum(np.array(self.previous_state[1:(STOCK_DIM + 1)]) * np.array(
                                       self.previous_state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))

            print(previous_total_asset)

            self.asset_memory = [previous_total_asset]
            self.state = [self.previous_state[0]] + \
                         [0] * STOCK_DIM + \
                         self.data.rsi.values.tolist() + \
                         self.data.cci.values.tolist() + \
                         self.data.adx.values.tolist()


        # iteration += 1
        return self.state

    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]