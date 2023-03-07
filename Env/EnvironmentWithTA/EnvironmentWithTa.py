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
REWARD_SCALING = 1


#[-0.7*HMAX_NORMALIZE, 0.5*HMAX_NORMALIZE,0.3*HMAX_NORMALIZE]
# w1, w2, w3,
class StockEnvTestingWithPV(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, modelNumber, tauValue=1, testOrTrain='valid', extraInformation='Vanilla', day=0):
        # super(StockEnv, self).__init__()
        # money = 10 , scope = 1
        self.day = day
        self.df = df
        self.modelNumber = modelNumber
        self.tauValue = tauValue
        self.testOrTrain = testOrTrain
        self.extraInformation = extraInformation

        # action_space normalization and shape is STOCK_DIM
        self.action_space = spaces.Box(low=0, high=1, shape=(STOCK_DIM + 1,))
        # Shape = 181: [Current Balance]+[prices 1-30]+[owned shares 1-30]
        # +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(20,))
        # load data from a pandas dataframe
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        # initalize state
        self.state = [1] + \
                     [0] * STOCK_DIM + \
                     self.data.adjcp.values.tolist() + \
                     self.data.macd.values.tolist() + \
                     self.data.rsi.values.tolist() + \
                     self.data.cci.values.tolist() + \
                     self.data.adx.values.tolist() + \
                    [INITIAL_ACCOUNT_BALANCE]
        # initialize reward
        self.cash_states = 0
        self.reward = 0
        self.cost = 0
        self.total_cost = 0
        # memorize all the total balance change
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.weight_memory = []
        self.rewards_memory = []
        self.trades = 0
        self.P_t_0 = 0
        self.P_t_1 =  1
        # self.reset()
        self._seed()
        self.equal_weights = [0.25,0.25,0.25,0.25]
        self.W_t_1 = [1,0,0,0]
        self.W_t =   [1,0,0,0]
        self.Yt = self.data.adjcp.values.tolist()
        self.P_equalWeights_t_1 = 1
        self.P_equalWeights_t_0 = 0
        self.asset_memory_equal_weights = [INITIAL_ACCOUNT_BALANCE]



    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        # print(actions)

        if self.terminal:
            plt.plot(self.asset_memory, 'r')
            plt.savefig('runs/account_value_train.png')
            plt.close()

            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv('runs/account_value_train.csv')
            df_total_value.columns = ['account_value']
            df_total_value['daily_return'] = df_total_value.pct_change(1)
            sharpe = (252 ** 0.5) * df_total_value['daily_return'].mean() / \
                     df_total_value['daily_return'].std()
            #CVAR
            #Alpha
            #Beta

            total_return = df_total_value['daily_return'].cumsum()
            drawdown = total_return - total_return.cummax()
            maxDD = drawdown.min()


            # 'totalReturn': [total_return], 'drawdown': [drawdown], 'maxDD': [maxDD]

            pd.DataFrame({'extraInformation:': self.extraInformation,'tauValue':self.tauValue,'model_number': self.modelNumber,'sharpe':[sharpe],'value_portfolio':[self.P_t_0],'trades':[self.trades],'total_cost':[self.total_cost], 'variance': [df_total_value['daily_return'].std()], 'maxDD': [maxDD], 'mean_return':[df_total_value['daily_return'].mean()],'riskless_state': [self.cash_states]}).to_csv("runs/" + self.testOrTrain  + '/' + "resultsPortfolioValue_" + self.testOrTrain +  ".csv",index=False, mode='a', header=False)
            pd.DataFrame({'weight_memory':self.weight_memory}).to_csv("runs/" +  self.testOrTrain + '/' + str(self.modelNumber) + "_resultsWeights_" + self.testOrTrain  + '_' + str(self.modelNumber) + "_" + str(self.tauValue) +".csv",index=True, mode='a', header=False)
#            'asset_memory': self.asset_memory, 'asset_memory_equal_weights': self.asset_memory_equal_weights
            info = {'extraInformation:': self.extraInformation,'tauValue':self.tauValue,'model_number': self.modelNumber,'riskless_state': [self.cash_states],'sharpe':[sharpe],'value_portfolio':[self.P_t_0],'trades':[self.trades],'total_cost':[self.total_cost], 'variance': [df_total_value['daily_return'].std()], 'maxDD': [maxDD], 'mean_return':[df_total_value['daily_return'].mean()]}

            return self.state, self.reward, self.terminal, info

        else:
            actions = actions

            if actions[0]>(actions[1] and actions[2] and actions[3]):
                self.cash_states = self.cash_states + 1

            v_t_1 = np.array([1] + self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)])

            numerator = np.exp(actions)
            denominator = np.sum(np.exp(actions))
            softmax_output = numerator / denominator

            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.state[0], self.state[1], self.state[2], self.state[3] = softmax_output

            # print("stock_shares:{}".format(self.state[29:]))
            self.state = [self.state[0]] + \
                         list(self.state[(1):(STOCK_DIM+ 1)]) + \
                         self.data.adjcp.values.tolist() + \
                         self.data.macd.values.tolist() + \
                         self.data.rsi.values.tolist() + \
                         self.data.cci.values.tolist() + \
                         self.data.adx.values.tolist() + \
                         [self.P_t_1]

            v_t_0 = np.array([1] + self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)])


            self.W_t = np.array(self.state[:(STOCK_DIM + 1)])

            Y_t = np.divide(v_t_0,v_t_1)

            self.cost = TRANSACTION_FEE_PERCENT * (
                np.abs(self.W_t_1[1:] - self.W_t[1:])).sum()

            if self.cost>0:
                self.trades = self.trades + 1

            self.total_cost = self.total_cost + self.cost

            self.P_t_0 = self.P_t_1 * (1 - self.cost) * np.dot(Y_t,self.W_t_1)
            self.P_equalWeights_t_0 = self.P_equalWeights_t_1 * (1 - self.cost) * np.dot(Y_t,self.equal_weights)

            self.W_t_1 = self.W_t

            self.P_t_0 = np.clip(self.P_t_0, 0, np.inf)

            self.asset_memory.append(self.P_t_0)

            self.asset_memory_equal_weights.append(self.P_equalWeights_t_0)

            self.weight_memory.append(self.W_t)

            self.reward = np.log((self.P_t_0 + (1e-7))/ (self.P_t_1 + (1e-7)))

            self.P_t_1 = self.P_t_0

            self.state = [self.state[0]] + \
                         list(self.state[(1):(STOCK_DIM+ 1)]) + \
                         self.data.adjcp.values.tolist() + \
                         self.data.macd.values.tolist() + \
                         self.data.rsi.values.tolist() + \
                         self.data.cci.values.tolist() + \
                         self.data.adx.values.tolist() + \
                         [self.P_t_1]

            self.P_equalWeights_t_1 = self.P_equalWeights_t_0

            self.rewards_memory.append(self.reward)

            info = {'extraInformation:': self.extraInformation,'tauValue':self.tauValue,'model_number': self.modelNumber,'riskless_state': [self.cash_states]}

        return self.state, self.reward, self.terminal, info

    def reset(self):
        self.P_t_1 =  1
        self.P_t_0 = 0
        self.P_equalWeights_t_1 = 1
        self.P_equalWeights_t_0 = 0
        self.W_t_1 = [1, 0 ,0 ,0]
        self.W_t   = [1, 0 ,0 ,0]
        self.cash_states = 0
        self.Yt = self.data.adjcp.values.tolist()
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.weight_memory = []
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.cost = 0
        self.trades = 0
        self.total_cost = 0
        self.terminal = False
        self.rewards_memory = []
        # initalize state
        self.state = [1] + \
                     [0] * STOCK_DIM + \
                     self.data.adjcp.values.tolist() + \
                     self.data.macd.values.tolist() + \
                     self.data.rsi.values.tolist() + \
                     self.data.cci.values.tolist() + \
                     self.data.adx.values.tolist() + \
                    [INITIAL_ACCOUNT_BALANCE]
        # iteration += 1
        return self.state

    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]