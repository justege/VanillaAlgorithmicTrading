import gym
import time
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
import os
import additional
from Env.EnvironmentVersion3.environmentVersion3 import StockEnvTrainVersion3
from Env.EnvironmentVersion3.ValidateEnvironmentVersion3 import StockEnvValidateVersion3

from Env.EnvironmentWithoutTA.EnvironmentWithoutTA_Trade import StockEnvTradeWithoutTA
from Env.EnvironmentWithTA.environment_validation import StockEnvValidationWithTA
from stable_baselines3.common.evaluation import evaluate_policy
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


train = data_split(data, start=20160101, end=20200101)
validate_data = data_split(data, start=20200101, end=20200301)
test_data = data_split(data, start=20200301, end=20210301)

TIMESTEPS = 10000
env = DummyVecEnv([lambda: StockEnvTrainVersion3(train)])
validate_env = DummyVecEnv([lambda: StockEnvValidateVersion3(validate_data)])

BATCHES = 200

seed = 3
env.seed(seed)
validate_env.seed(seed)

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


FIRSTMODEL = 62



for batch in range(FIRSTMODEL,FIRSTMODEL+BATCHES):
    env.reset()
    models_dir = f"models/{int(time.time())}/"
    logdir = f"logs/{int(time.time())}/"

    tensorboard_log = '/tmp/A2C' + str(TIMESTEPS) + '_' + str(batch-1)

    batch_number.append(batch)

    if FIRSTMODEL == 0:
        print('First Model')
        model = A2C("MlpPolicy", env , tensorboard_log=tensorboard_log, verbose=1, n_steps=256, ent_coef=0.005,
                    learning_rate=0.0007)

        model.learn(total_timesteps=int(TIMESTEPS))
        FIRSTMODEL = 1

        print('Model Finish')

    else:
        print('loading Model' + str(batch-1))
        model = A2C.load("runs/A2C_" + str(TIMESTEPS) + '_' + str(batch-1))
        model.set_env(env)
        model.learn(total_timesteps=int(TIMESTEPS))
        print('Model Finish')

    vali_env = DummyVecEnv([lambda: StockEnvValidateVersion3(test_data, modelNumber=batch, tauValue=1, testOrTrain='train', extraInformation='A2C')])

    vali_env.seed(seed)
    model.save("runs/A2C_" + str(TIMESTEPS) + '_' + str(batch))
    #evaluate_policy(model, env, n_eval_episodes=1, render=False, return_episode_rewards=True)
    evaluate_policy(model, vali_env, n_eval_episodes=1, render=False, return_episode_rewards=True)

