import gym
import time
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
from environment import StockEnvTrain
from environment_validation import StockEnvTest
from environment_Trade import StockEnvValid
from stable_baselines3.common.env_util import make_vec_env
import pandas as pd
import numpy as np

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


preprocessed_path = "0001_test.csv"
data = pd.read_csv(preprocessed_path, index_col=0)
data = data.drop(columns=["datadate_full"])
data = data[["datadate","tic","Close","open","high","low","volume","macd","rsi","cci","adx"]]
#print(data.to_string())

data.Close = data.Close.apply(np.int64)
data.macd = data.macd.apply(np.int64)
data.rsi = data.rsi.apply(np.int64)
data.cci = data.cci.apply(np.int64)
data.adx = data.adx.apply(np.int64)


train = data_split(data, start=20180101, end=20210101)
validate = data_split(data, start=20210101, end=20220101)
test_d = data_split(data, start=20220101, end=20221101)


env = DummyVecEnv([lambda: StockEnvTrain(train)])
test_env = DummyVecEnv([lambda: StockEnvTest(test_d)])
vali_env = DummyVecEnv([lambda: StockEnvValid(validate)])

BATCHES = 200
TIMESTEPS = 20000

seed = 2
env.seed(seed)
test_env.seed(seed)
vali_env.seed(seed)


batch_number = []
batch_rewardsl = []
t_batch_number = []
t_batch_rewardsl = []

obs = vali_env.reset()

FIRSTMODEL = 0

for batch in range(FIRSTMODEL,BATCHES):
    env.reset()
    models_dir = f"models/{int(time.time())}/"
    logdir = f"logs/{int(time.time())}/"

    if FIRSTMODEL == 0:
        print('First Model')
        model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=logdir, ent_coef = 0.005)
        FIRSTMODEL = 1
    else:
        print('loading Model' + str(batch-1))
        model = A2C.load("runs/A2C_" + str(TIMESTEPS) + '_' + str(batch - 1) + '.pth')
        model.set_env(env)


    model.learn(total_timesteps=TIMESTEPS)
    model.save("runs/A2C_"+str(TIMESTEPS)+'_'+str(batch)+'.pth')

    rewardsl_v = []
    t_rewardsl = []

    score = 0
    print('-----begin validating----')
    for i in range(10):
        done = 0
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = vali_env.step(action)
            score = score + rewards
            if done:
                rewardsl_v.append(score)
                score = 0
                vali_env.reset()
    batch_rewardsl.append(np.array(rewardsl_v).mean())
    batch_number.append(batch)
    print('-----finish validating----')
    print('-------Evaluating------')
    for i in range(10):
        done = 1
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = test_env.step(action)
            score = score + rewards
            if done:
                t_rewardsl.append(score)
                score = 0
                test_env.reset()
                break
    t_batch_rewardsl.append(np.array(t_rewardsl).mean())
    print('-------Finished Evaluating------')

df_scores = pd.DataFrame(list(zip(batch_number,batch_rewardsl, t_batch_rewardsl)))
df_scores.to_csv('CSVs/A2C_results_eval_mean.csv', mode='a', encoding='utf-8', index=True)
print('mean of scores:{}'.format(np.mean(df_scores)))
rewardsl = np.array(t_batch_rewardsl).mean()
print(rewardsl)


