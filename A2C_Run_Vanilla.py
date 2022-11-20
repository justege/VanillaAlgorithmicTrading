import gym
import time
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C

import additional
from environment import StockEnvTrain
from environment_validation import StockEnvValidation
from environment_Trade import StockEnvTrade
from basicTradingEnv import BasicStockEnvTrain
from stable_baselines3.common.env_util import make_vec_env
import pandas as pd
import numpy as np
from additional import *

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


preprocessed_path = "/Users/egemenokur/PycharmProjects/VanillaAlgorithmicTrading/synthetic_portolio_ready.csv"
data = pd.read_csv(preprocessed_path, index_col=0)
#print(data.to_string())


train = data_split(data, start=20200101, end=20200201)
validate = data_split(data, start=20200201, end=20200301)
test = data_split(data, start=20200201, end=20200301)

print(data.daily_return.values.tolist())

print(train)
print(test)


env = DummyVecEnv([lambda: BasicStockEnvTrain(train)])
vali_env = DummyVecEnv([lambda: BasicStockEnvTrain(validate)])
test_env = DummyVecEnv([lambda: BasicStockEnvTrain(test)])


BATCHES = 1
TIMESTEPS = 10000

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
        model = A2C('MlpPolicy', env=env, verbose=0, tensorboard_log=logdir, learning_rate=10e-5, gamma=0.99)
        FIRSTMODEL = 1
        print('Model Finish')
    else:
        print('loading Model' + str(batch-1))
        model = A2C.load("runs/A2C_" + str(TIMESTEPS) + '_' + str(batch-1) + '.pth')
        print('Model Finish')
        model.set_env(env)
        model.learn(total_timesteps=int(TIMESTEPS))
    """
    rewardsl_train = []
    rewardsl_v = []
    t_rewardsl = []
    sharpel_train = []
    sharpel_v = []
    t_sharpel = []
    cumretl_train = []
    cumretl_v = []
    t_cumretl = []
    """


    score = 0

    model.save("runs/A2C_" + str(TIMESTEPS) + '_' + str(batch) + '.pth')
    print('-----testing period validating----')
    obs = env.reset()
    for i in range(10):
        done = 0
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            if done:
                env.reset()
    print('-----testing period done----')
    #train_batch_rewardsl.append(np.array(rewardsl_train).mean())
    #batch_sharpeL_train.append(np.array(sharpel_train).mean())
    #batch_FassetL_train.append(np.array(cumretl_train).mean())
    print('-----begin validating----')
    obs = vali_env.reset()
    for i in range(10):
        done = 0
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = vali_env.step(action)
            if done:
                #rewardsl_v.append(score)
                #sharpel_v.append(vali_env.sharpe)
                #cumretl_v.append(vali_env.final_asset_value)
                score = 0
                vali_env.reset()

    #batch_sharpeL_validate.append(np.array(sharpel_v).mean())
    #batch_FassetL_validate.append(np.array(cumretl_v).mean())
    #batch_rewardsl.append(np.array(rewardsl_v).mean())
    print('-----finish validating----')
    print('-------Evaluating------')
    obs = test_env.reset()
    for i in range(10):
        done = 0
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = test_env.step(action)
            score = score + rewards
            if done:
                #rewardsl_v.append(score)
                #sharpel_v.append(vali_env.sharpe)
                #cumretl_v.append(vali_env.final_asset_value)
                score = 0
                test_env.reset()

    #t_batch_rewardsl.append(np.array(t_rewardsl).mean())
    #batch_sharpeL_test.append(np.array(sharpel_train).mean())
    #batch_FassetL_test.append(np.array(cumretl_train).mean())

    print('-------Finished Evaluating------')


"""
df_scores = pd.DataFrame(list(zip(batch_number,train_batch_rewardsl,batch_rewardsl, t_batch_rewardsl,batch_sharpeL_test,batch_FassetL_test, batch_sharpeL_validate, batch_FassetL_validate, batch_sharpeL_train,batch_FassetL_train)))
df_scores.to_csv('CSVs/PPO_results_eval_mean.csv', mode='a', encoding='utf-8', index=True)
print('mean of scores:{}'.format(np.mean(df_scores)))
rewardsl = np.array(t_batch_rewardsl).mean()
print(rewardsl)

"""

