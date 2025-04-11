import gymnasium as gym
import numpy as np
import d3rlpy
import pickle
import pandas as pd

from env.atari.represented_atari_game import GymCompatWrapper2

hyperparams = {
        "env": "FrozenLake-v1", #"CartPole-v0", # "Acrobot-v0", "MountainCar-v0", "FrozenLake-v1", "CliffWalking-v0", "Taxi-v3", "RepresentedPong-v0"
        "seed": 42069,
        "n_episodes": 10,#5000,
        "max_episode_len": 200, # Around 10h per 100k steps in Leviathan server
        "eps": 0.1,  # epsilon for exploration
        "n_exp": 5,
        "n_pretrain_eps": 30,
        "n_online_eps": 270,
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

n_pretrain_eps = hyperparams["n_pretrain_eps"] #10
n_online_eps = hyperparams["n_online_eps"]
n_exp = hyperparams["n_exp"]
hyperparams["target_update_interval"] = 200 # For pretraining, leave the original value of 1000 for online training

# # d3rlpy supports both Gym and Gymnasium
# if "Represented" in hyperparams["env"]:
#     env = GymCompatWrapper2(gym.make(hyperparams["env"]))
#     eval_env = GymCompatWrapper2(gym.make(hyperparams["env"]))
# else:
#     env = gym.make(hyperparams["env"])
#     eval_env = gym.make(hyperparams["env"])

# fix seed
d3rlpy.seed(hyperparams["seed"])
# d3rlpy.envs.seed_env(env, hyperparams["seed"])
# d3rlpy.envs.seed_env(eval_env, hyperparams["seed"])

with open(f"data/{hyperparams['env'].split('-')[0]}_Qwen2.5-32B-Instruct_Neps_30_20250411030422.pkl", 'rb') as file: #FrozenLake
# with open(f"data/{hyperparams['env'].split('-')[0]}_Qwen2.5-32B-Instruct_Neps_30_20250409124533.pkl", 'rb') as file: #CartPole with Eps
    Qwen_32B_dataset = pickle.load(file)

with open(f"data/{hyperparams['env'].split('-')[0]}_Qwen2.5-7B-Instruct_Neps_30_20250411000858.pkl", 'rb') as file: #FrozenLake
# with open(f"data/{hyperparams['env'].split('-')[0]}_Qwen2.5-7B-Instruct_Neps_30_20250409023954.pkl", 'rb') as file: #CartPole with Eps
    Qwen_7B_dataset = pickle.load(file)

def ensure_numpy_array(x, hyperparams=hyperparams):
    if isinstance(gym.make(hyperparams["env"]).observation_space, gym.spaces.Discrete):
        # Convert discrete observations to one-hot encoded arrays
        return np.eye(gym.make(hyperparams["env"]).observation_space.n)[x]
    if not isinstance(x, np.ndarray):
        return np.array([x])
    return x
def get_new_dataset(dataset, n_eps):
    observations, actions, rewards, terminals = [], [], [], []
    # Extract observations, actions, rewards, and terminals from the original dataset
    for episode in dataset.episodes[:n_eps]:
        observations += [ensure_numpy_array(o) for o in episode.observations]  # Ensure observations are numpy arrays
        actions += [a for a in episode.actions]
        rewards += [r for r in episode.rewards]
        terminals += [0] * (len(episode.rewards) - 1) + [1] # Add terminal flag for the last step
    dataset_new = d3rlpy.dataset.MDPDataset(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            terminals=np.array(terminals),
        )
    # Verify the lengths of the extracted lists
    print(f"Number of episodes: {len(dataset_new.episodes)}")
    return dataset_new

Qwen_32B_dataset_new = get_new_dataset(Qwen_32B_dataset, n_pretrain_eps)
Qwen_7B_dataset_new = get_new_dataset(Qwen_7B_dataset, n_pretrain_eps)

pretrain_7b_dqn = d3rlpy.algos.DoubleDQNConfig(
    batch_size=hyperparams['batch_size'],
    learning_rate=hyperparams['learning_rate'],
    gamma=hyperparams['gamma'],
    target_update_interval=hyperparams['target_update_interval']
    ).create(device=hyperparams['gpu'])
pretrain_32b_dqn = d3rlpy.algos.DoubleDQNConfig(
    batch_size=hyperparams['batch_size'], 
    learning_rate=hyperparams['learning_rate'],
    gamma=hyperparams['gamma'],
    target_update_interval=hyperparams['target_update_interval']
    ).create(device=hyperparams['gpu'])

# start offline training
pretrain_7b_dqn.fit(Qwen_7B_dataset_new, n_steps=hyperparams["n_pretrain_steps"], n_steps_per_epoch=hyperparams['n_steps_per_epoch'])
with open(f'models/{hyperparams["env"].split("-")[0]}_ddqn_pretrain_7b_{hyperparams["n_pretrain_steps"]}_steps_{hyperparams["n_pretrain_eps"]}.pkl', 'wb') as file:
    pickle.dump(pretrain_7b_dqn, file)

pretrain_32b_dqn.fit(Qwen_32B_dataset_new, n_steps=hyperparams["n_pretrain_steps"], n_steps_per_epoch=hyperparams['n_steps_per_epoch'])
with open(f'models/{hyperparams["env"].split("-")[0]}_ddqn_pretrain_32b_{hyperparams["n_pretrain_steps"]}_steps_{hyperparams["n_pretrain_eps"]}.pkl', 'wb') as file:
    pickle.dump(pretrain_32b_dqn, file)