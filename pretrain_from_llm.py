import gymnasium as gym
import numpy as np
import d3rlpy
import pickle
import pandas as pd

from env.atari.represented_atari_game import GymCompatWrapper2

hyperparams = {
        "env": "MountainCar-v0", #"CartPole-v0", # "Acrobot-v0", "MountainCar-v0", "FrozenLake-v1", "CliffWalking-v0", "Taxi-v3", "RepresentedPong-v0"
        "seed": 42069,
        "n_episodes": 10,#5000,
        "max_episode_len": 200, # Around 10h per 100k steps in Leviathan server
        "eps": 0.1,  # epsilon for exploration
        "n_exp": 5,
        "n_pretrain_eps": 10,
        "n_online_eps": 90,
        "gpu": True, # True if use GPU to train with d3rlpy
        "buffer_size": 100000, #Test with 100k, 200k, 500k. 1M might be too much
        "data_path": None,#'data/CartPole_Qwen2.5-7B-Instruct_Neps_10_20250406040150.pkl',
        "model_path": None,#'d3rlpy_loss/DoubleDQN_online_20250331153346/model_600000.d3',
        "batch_size":256, #Test smaller batch size: 32, 64. May be noisier
        "learning_rate":5e-5,
        "gamma":0.99,
        "target_update_interval":1000, #Test with 1k, 2k, 5k
        "n_steps_per_epoch": 200,
        "n_pretrain_steps": 3000
    }

# class InputParams:
#     def __init__(self):
#         self.env="CartPole-v0" #"CartPole-v0", # "Acrobot-v0", "MountainCar-v0", "FrozenLake-v1", "CliffWalking-v0", "Taxi-v3", "RepresentedPong-v0"
#         self.seed=1
#         self.gpu=True

# args=InputParams()
n_pretrain_eps = 10
n_online_eps = 90#90
n_exp = 5

# d3rlpy supports both Gym and Gymnasium
if "Represented" in hyperparams["env"]:
    env = GymCompatWrapper2(gym.make(hyperparams["env"]))
    eval_env = GymCompatWrapper2(gym.make(hyperparams["env"]))
else:
    env = gym.make(hyperparams["env"])
    eval_env = gym.make(hyperparams["env"])

# fix seed
d3rlpy.seed(hyperparams["seed"])
d3rlpy.envs.seed_env(env, hyperparams["seed"])
d3rlpy.envs.seed_env(eval_env, hyperparams["seed"])

with open('data/MountainCar_Qwen2.5-32B-Instruct_Neps_10_20250407125034.pkl', 'rb') as file:
    Qwen_32B_dataset = pickle.load(file)
    
with open('data/MountainCar_Qwen2.5-7B-Instruct_Neps_10_20250407113942.pkl', 'rb') as file:
    Qwen_7B_dataset = pickle.load(file)

Qwen_32B_rewards = []
for i in range(n_pretrain_eps):
    Qwen_32B_rewards.append(Qwen_32B_dataset.episodes[i].compute_return())
Qwen_7B_rewards = []
for i in range(n_pretrain_eps):
    Qwen_7B_rewards.append(Qwen_7B_dataset.episodes[i].compute_return())

Qwen_32B_avg = np.ones(n_pretrain_eps + n_online_eps) * np.mean(Qwen_32B_rewards)
Qwen_7B_avg = np.ones(n_pretrain_eps + n_online_eps) * np.mean(Qwen_7B_rewards)

pretrain_7b_dqn = d3rlpy.algos.DoubleDQNConfig(
    batch_size=hyperparams['batch_size'], #Test smaller batch size: 32, 64. May be noisier
    learning_rate=hyperparams['learning_rate'],
    gamma=hyperparams['gamma'],
    target_update_interval=hyperparams['target_update_interval'] #Test with 1k, 2k, 5k
    ).create(device=hyperparams['gpu'])
pretrain_32b_dqn = d3rlpy.algos.DoubleDQNConfig(
    batch_size=hyperparams['batch_size'], #Test smaller batch size: 32, 64. May be noisier
    learning_rate=hyperparams['learning_rate'],
    gamma=hyperparams['gamma'],
    target_update_interval=hyperparams['target_update_interval'] #Test with 1k, 2k, 5k
    ).create(device=hyperparams['gpu'])

hyperparams["target_update_interval"] = 200
# start offline training
pretrain_7b_dqn.fit(Qwen_7B_dataset, n_steps=hyperparams["n_pretrain_steps"], n_steps_per_epoch=hyperparams['n_steps_per_epoch'])
with open(f'models/{hyperparams["env"].split("-")[0]}_ddqn_pretrain_7b_{hyperparams["n_pretrain_steps"]}_steps.pkl', 'wb') as file:
    pickle.dump(pretrain_7b_dqn, file)

pretrain_32b_dqn.fit(Qwen_32B_dataset, n_steps=hyperparams["n_pretrain_steps"], n_steps_per_epoch=hyperparams['n_steps_per_epoch'])
with open(f'models/{hyperparams["env"].split("-")[0]}_ddqn_pretrain_32b_{hyperparams["n_pretrain_steps"]}_steps.pkl', 'wb') as file:
    pickle.dump(pretrain_32b_dqn, file)