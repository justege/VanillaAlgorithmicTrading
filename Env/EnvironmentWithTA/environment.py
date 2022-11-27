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
class StockEnvTrainWithTA(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, day=0):
        # super(StockEnv, self).__init__()
        # money = 10 , scope = 1
        self.day = day
        self.df = df

        # action_space normalization and shape is STOCK_DIM
        self.action_space = spaces.Box(low=0, high=1, shape=(STOCK_DIM + 1,))
        # Shape = 181: [Current Balance]+[prices 1-30]+[owned shares 1-30]
        # +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(19,))
        # load data from a pandas dataframe
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        # initalize state
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                     [0] * STOCK_DIM + \
                     self.data.adjcp.values.tolist() + \
                     self.data.macd.values.tolist() + \
                     self.data.rsi.values.tolist() + \
                     self.data.cci.values.tolist() + \
                     self.data.adx.values.tolist()
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



    def make_actions(self, index, action):
        available_amount = (1 -  sum(np.array(self.state[1:(index)])))
        if available_amount > 0:
            self.trades += 1
            self.state[index] = min(available_amount, action)



    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        # print(actions)

        if self.terminal:
            plt.plot(self.asset_memory, 'r')
            plt.savefig('results/account_value_train.png')
            plt.close()

            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv('results/account_value_train.csv')
            df_total_value.columns = ['account_value']
            df_total_value['daily_return'] = df_total_value.pct_change(1)
            sharpe = (252 ** 0.5) * df_total_value['daily_return'].mean() / \
                     df_total_value['daily_return'].std()

            #print("-------------------------------------------------FINISH---------------------")
            #print("Portfolio Value:{}".format(self.P_t_0))
            #print("Sharpe: ",sharpe)

            #column_name = ["Sharpe", "PortfolioValue", "AmountOfTrades"]  # The name of the columns


            pd.DataFrame({'sharpe':[sharpe],'PortfolioValue':[self.P_t_0],'trades':[self.trades]}).to_csv("Results_Train.csv",index=False, mode='a', header=False)


            # print("=================================")
            #df_rewards = pd.DataFrame(self.rewards_memory)
            #df_rewards.to_csv('results/account_rewards_train.csv')
            #print('self.reward: {}'.format(np.mean(self.rewards_memory)))

            return self.state, self.reward, self.terminal, {}

        else:
            # print(np.array(self.state[1:29]))
            actions = actions

            # actions = (actions.astype(int))
            #print("actions {}".format(actions))
            self.state[0] = actions[0]

            #print("state before buying actions {}".format(self.state))
            v_t_1 = np.array([1] + self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)])


            for index in range(1,STOCK_DIM+1):
                # print('take buy action: {}'.format(actions[index]))
                #print("while buying actions {}".format(self.state))
                self.make_actions(index, actions[index])

            self.day += 1
            self.data = self.df.loc[self.day, :]
            # load next state



            self.state[0] = (1 - self.state[1] - self.state[2] - self.state[3])


            # print("stock_shares:{}".format(self.state[29:]))
            self.state = [self.state[0]] + \
                         list(self.state[(1):(STOCK_DIM+ 1)]) + \
                         self.data.adjcp.values.tolist() + \
                         self.data.macd.values.tolist() + \
                         self.data.rsi.values.tolist() + \
                         self.data.cci.values.tolist() + \
                         self.data.adx.values.tolist()

            v_t_0 = np.array( [1] + self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)])

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


            if self.P_t_0 == 0:
                self.reward = -1
                self.rewards_memory.append(0)

            elif self.state[0]==1:
                self.reward = -0.5
                self.rewards_memory.append(0)
            else:
                self.reward = self.P_t_0/self.P_t_1 # or this: (self.P_t_0/self.P_t_1)
                self.rewards_memory.append(self.reward)

            self.P_t_1 = self.P_t_0

            #print("step_reward:{}".format(self.reward))
            self.reward = self.reward * REWARD_SCALING




        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.P_t_1 =  1
        self.P_t_0 = 0
        self.W_t_1 = [1, 0 ,0 ,0]
        self.W_t   = [1, 0 ,0 ,0]
        self.Yt = self.data.adjcp.values.tolist()
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        # initiate state
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                     [0] * STOCK_DIM + \
                     self.data.adjcp.values.tolist() + \
                     self.data.macd.values.tolist() + \
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