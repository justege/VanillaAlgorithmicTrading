import gym
import time
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
import os
import additional
from Env.EnvironmentWithTA.EnvironmentWithTa import StockEnvTrainWithTA
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


train = data_split(data, start=20180101, end=20210101)
validate_train = data_split(data, start=20180101, end=20210101)
validate = data_split(data, start=20210101, end=20210601)
trade = data_split(data, start=20210601, end=20220601)



TIMESTEPS = 12501
env = DummyVecEnv([lambda: StockEnvTrainWithTA(train)])
test_env = DummyVecEnv([lambda: StockEnvValidationWithTA(validate_train, modelNumber=TIMESTEPS)])
vali_env = DummyVecEnv([lambda: StockEnvValidationWithTA(validate, modelNumber=TIMESTEPS)])

print(train)
print(validate)



BATCHES = 200


seed = 3
env.seed(seed)
test_env.seed(seed)
vali_env.seed(seed)


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


FIRSTMODEL = 0

COMMENT = 'Vanilla'

for batch in range(FIRSTMODEL,FIRSTMODEL+BATCHES):
    env.reset()
    models_dir = f"models/{int(time.time())}/"
    logdir = f"logs/{int(time.time())}/"

    tensorboard_log = '/tmp/PPO' + str(TIMESTEPS) + '_' + str(batch-1) + '_' +COMMENT
    tmp_path = "/tmp/"
    # set up logger

    batch_number.append(batch)

    if FIRSTMODEL == 0:
        print('First Model')
        model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=tensorboard_log, n_steps=512, ent_coef=0.005,
                    learning_rate=0.0001)
        model.learn(total_timesteps=int(TIMESTEPS))
        FIRSTMODEL = 1
        print('Model Finish')

    else:
        print('loading Model' + str(batch-1))
        model = PPO.load("/Users/egemenokur/PycharmProjects/VanillaAlgorithmicTrading/runs/PPO_" + str(TIMESTEPS) + '_' + str(batch-1) + '_' +COMMENT)


        model.set_env(env)
        model.learn(total_timesteps=int(TIMESTEPS))
        print('Model Finish')

    model.save("runs/PPO_" + str(TIMESTEPS) + '_' + str(batch)+ '_' +COMMENT)

    evaluate_policy(model, test_env, n_eval_episodes=1, render=False, return_episode_rewards=True)
    evaluate_policy(model, vali_env, n_eval_episodes=1, render=False, return_episode_rewards=True)