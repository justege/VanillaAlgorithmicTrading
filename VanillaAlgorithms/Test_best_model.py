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
from Env.EnvironmentWithTA.EnvironmentWithTa import StockEnvTrainWithTA
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


train = data_split(data, start=20180101, end=20220101)
validate = data_split(data, start=20220101, end=20220701)


env = StockEnvTrainWithTA(train)
vali_env = StockEnvTrainWithTA(validate)

print(train)



BATCHES = 92
TIMESTEPS = 25000

seed = 3
env.seed(seed)
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

FIRSTMODEL = 1

COMMENT = 'TACCIRSIADX'

for batch in range(FIRSTMODEL,BATCHES):
    env.reset()
    models_dir = f"models/{int(time.time())}/"
    logdir = f"logs/{int(time.time())}/"

    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))
    #model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=0, learning_rate=0.005, batch_size=32)
    #model = A2C("MlpPolicy", env)
    model = PPO("MlpPolicy", env)
    model.load(
        '/Users/egemenokur/PycharmProjects/VanillaAlgorithmicTrading/runs/PPO_' + str(TIMESTEPS) + '_' + str(batch - 1) + '_' +COMMENT)



    print('loading Model' + str(batch-1))


    rewardsl_tra = list()
    rewardsl_v = list()
    t_rewardsl = list()

    score = 0
    print('-----training----')
    obs = env.reset()
    train_batch_rewardsl = []
    for i in range(15):

        done = 0
        score = 0
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            score = score + rewards
            if done:
                train_batch_rewardsl.append(info['value_portfolio'])
                env.reset()
                obs = env.reset()

    mean_result = np.mean(train_batch_rewardsl)

    batch_FassetL_train.append(mean_result)

    obs = vali_env.reset()
    print('-----validating----')

    batch_rewardsl = []
    for i in range(15):
        done = 0
        score = 0
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = vali_env.step(action)

            score = score + rewards
            if done:
                batch_rewardsl.append(info['value_portfolio'])
                vali_env.reset()
                obs = vali_env.reset()

    mean_result = np.mean(batch_rewardsl)
    batch_FassetL_validate.append(mean_result)


    print(np.stack(batch_FassetL_train).T)
    print(np.stack(batch_FassetL_validate).T)
    batch_number.append(batch)
    print('-----finish validating----')


df_scores = pd.DataFrame(list(zip(batch_number,batch_FassetL_train, batch_FassetL_validate)))
df_scores.to_csv('/Users/egemenokur/PycharmProjects/VanillaAlgorithmicTrading/CSVs/PPO_10001_final_results_eval_mean_with_TA_differentENV.csv', mode='a', encoding='utf-8', index=True)
print('mean of scores:{}'.format(np.mean(df_scores)))
rewardsl = np.array(t_batch_rewardsl).mean()
print(rewardsl)


