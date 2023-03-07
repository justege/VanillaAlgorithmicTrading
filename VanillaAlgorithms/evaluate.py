import gym
import time
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, DDPG, A2C
import os
from stable_baselines3.common.evaluation import evaluate_policy
import additional
from Env.EnvironmentWithoutTA.EnvironmentWithoutTA import StockEnvTrainWithoutTA
from Env.EnvironmentWithoutTA.EnvironmentWithoutTA_Trade import StockEnvTradeWithoutTA
from Env.EnvironmentWithoutTA.EnvironmentWithoutTA_validation import StockEnvValidationWithTA
from Env.EnvironmentWithTA.EnvironmentWithTa import StockEnvTestingWithPV
from Env.EnvironmentWithTA.environment_validation import StockEnvValidationWithTA
from Env.EnvironmentWithTA.environment_Trade import StockEnvTradeWithoutTA
from stable_baselines3.common.env_util import make_vec_env
import pandas as pd
import numpy as np
from additional import *
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# Parallel environments

def data_split(df, start, end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.datadate >= start) & (df.datadate < end)]
    data = data.sort_values(['datadate', 'tic'], ignore_index=True)

    # data  = data[final_columns]
    data.index = data.datadate.factorize()[0]

    return data


preprocessed_path = "/Users/egemenokur/PycharmProjects/VanillaAlgorithmicTrading/model/0001_test.csv"
data = pd.read_csv(preprocessed_path, index_col=0)
data = data.drop(columns=["datadate_full"])


data = data[["datadate", "tic", "adjcp", "open", "high", "low", "volume", "macd", "rsi", "cci", "adx"]]

data['adjcp'] = round(data['adjcp'], 1)
data['macd'] = round(data['macd'], 1)
data['rsi'] = round(data['rsi'], 1)
data['cci'] = round(data['cci'], 1)
data['adx'] = round(data['adx'], 1)
#print(data.to_string())

validate = data_split(data, start=20210101, end=20210601)
test = data_split(data, start=20210601, end=20220101)






BATCHES = 48
TIMESTEPS = 12501


batch_number = []
batch_rewardsl = []
t_batch_number = []
t_batch_rewardsl = []
train_batch_rewardsl = []


batch_sharpeL_test = []
batch_FassetL_test = []

batch_sharpeL_validate = []
batch_FassetL_validate = []

batch_sharpeL_train = []
batch_FassetL_train = []

FIRSTMODEL = 2





for batch in range(FIRSTMODEL,BATCHES):

    vali_env = StockEnvValidationWithTA(validate, modelNumber=batch)
    test_env = StockEnvTestingWithPV(test, modelNumber=batch)

    seed = 3
    test_env.seed(seed)
    vali_env.seed(seed)

    vali_env.reset()
    test_env.reset()
    models_dir = f"models/{int(time.time())}/"
    logdir = f"logs/{int(time.time())}/"

    n_actions = vali_env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))
    #model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=0, learning_rate=0.005, batch_size=32)
    #model = A2C("MlpPolicy", vali_env)
    model = PPO("MlpPolicy", vali_env)

    model.load(
        '/Users/egemenokur/PycharmProjects/VanillaAlgorithmicTrading/runs/PPO_' + str(TIMESTEPS) + '_' + str(batch - 1) +'_Vanilla')

    print('loading Model' + str(batch-1))

    evaluate_policy(model, vali_env, n_eval_episodes=1, render=False, return_episode_rewards= True)
    evaluate_policy(model, test_env, n_eval_episodes=1, render=False, return_episode_rewards= True)


