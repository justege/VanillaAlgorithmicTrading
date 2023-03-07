import gym
import time
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DDPG

import additional
from Env.EnvironmentWithoutTA.EnvironmentWithTA import StockEnvValidationWithTA
from Env.EnvironmentWithoutTA.EnvironmentWithoutTA_validation import StockEnvValidationWithTA
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
data = data[["datadate","tic","adjcp"]]
#print(data.to_string())


train = data_split(data, start=20211010, end=20220101)
validate = data_split(data, start=20220101, end=20220601)
test = data_split(data, start=20220601, end=20221011)


env = StockEnvTrainWithTA(train)
test_env = StockEnvTradeWithTA(test)
vali_env = StockEnvValidationWithTA(validate)

print(train)
print(test)



BATCHES = 50
TIMESTEPS = 100000

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

for batch in range(FIRSTMODEL,BATCHES):
    env.reset()
    models_dir = f"models/{int(time.time())}/"
    logdir = f"logs/{int(time.time())}/"
    batch_number.append(batch)

    if FIRSTMODEL == 0:
        print('First Model')
        n_actions = env.action_space.shape[-1]
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))
        model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, learning_rate=0.001, batch_size=64)
        model.learn(total_timesteps=int(TIMESTEPS))
        FIRSTMODEL = 1
        print('Model Finish')
    else:
        print('loading Model' + str(batch-1))
        model = DDPG.load("runs/DDPG_" + str(TIMESTEPS) + '_' + str(batch-1))
        print('Model Finish')
        model.set_env(env)
        model.learn(total_timesteps=int(TIMESTEPS))

    score = 0
    model.save("runs/DDPG_" + str(TIMESTEPS) + '_' + str(batch))

