from __future__ import division
import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib


matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
# initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE = 1

# [100,-50,100]

# total number of stocks in our portfolio
STOCK_DIM = 3
# transaction fee: 1/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.00125
REWARD_SCALING = 1e-4


#[-0.7*HMAX_NORMALIZE, 0.5*HMAX_NORMALIZE,0.3*HMAX_NORMALIZE]
# w1, w2, w3,
class StockEnvTrain(gym.Env):
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
        # self.reset()
        self._seed()



    def make_actions(self, index, action):
        # perform buy action based on the sign of the action

        #print("np.array(self.state[(0):(index-1)]))) {}".format(self.state[0:index]))
        #print("self.state[index + 1 + STOCK_DIM]  {}".format(self.state[index + 1 + STOCK_DIM] ))

        available_amount = (1 -  sum(np.array(self.state[:(index)])))  # 1 -sum( idx1, idx2)


        #print("available_amount {}".format(float(available_amount)))

        previous_action = self.state[index  + 1]

        #print("previous_action {}".format(previous_action))

        #print("min(available_amount, action) {}".format(min(available_amount, action)))

        self.state[index  + 1] = min(available_amount, action)

        #print("self.state[index  + 1] {}".format(self.state[index  + 1]))

        self.cost += (abs(self.state[index + STOCK_DIM +1] * min(available_amount, action) - previous_action) * TRANSACTION_FEE_PERCENT)
        self.trades += 1

    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        # print(actions)

        if self.terminal:
            plt.plot(self.asset_memory, 'r')
            plt.savefig('results/account_value_train.png')
            plt.close()
            end_total_asset = self.state[0] + \
                              sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(
                                  self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))

            # print("end_total_asset:{}".format(end_total_asset))
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv('results/account_value_train.csv')
            # print("total_reward:{}".format(self.state[0]+sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):61]))- INITIAL_ACCOUNT_BALANCE ))
            # print("total_cost: ", self.cost)
            # print("total_trades: ", self.trades)
            df_total_value.columns = ['account_value']
            df_total_value['daily_return'] = df_total_value.pct_change(1)
            sharpe = (252 ** 0.5) * df_total_value['daily_return'].mean() / \
                     df_total_value['daily_return'].std()
            # print("Sharpe: ",sharpe)
            # print("=================================")
            df_rewards = pd.DataFrame(self.rewards_memory)
            # df_rewards.to_csv('results/account_rewards_train.csv')

            # print('total asset: {}'.format(self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))))
            # with open('obs.pkl', 'wb') as f:
            #    pickle.dump(self.state, f)

            return self.state, self.reward, self.terminal, {}

        else:
            # print(np.array(self.state[1:29]))

            actions = actions

            # actions = (actions.astype(int))
            #print("actions {}".format(actions))


            self.state[0] = actions[0]

            #print("state before buying actions {}".format(self.state))


            begin_total_asset = self.state[0] + \
                                sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(
                                    self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))
            # print("begin_total_asset:{}".format(begin_total_asset))

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

            #print("state after buying actions {}".format(self.state))

            end_total_asset = self.state[0] + \
                              sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(
                                  self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))

            self.asset_memory.append(end_total_asset)
            # print("end_total_asset:{}".format(end_total_asset))

            self.reward = end_total_asset - begin_total_asset - self.cost
            # print("step_reward:{}".format(self.reward))
            self.rewards_memory.append(self.reward)

            self.reward = self.reward * REWARD_SCALING

        return self.state, self.reward, self.terminal, {}

    def reset(self):
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