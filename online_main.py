import gymnasium as gym
import numpy as np
import d3rlpy
import pickle

import matplotlib.pyplot as plt
from env.atari.represented_atari_game import GymCompatWrapper2


def online_training(
    env,
    eval_env,
    hyperparams,
    explorer=None,
    model=None
):
    # Load model with proper validation
    if hyperparams['model_path']:
        # with open(hyperparams['model_path'], 'rb') as file:
        #     dqn = pickle.load(file)
        dqn = d3rlpy.load_learnable(hyperparams['model_path'])
    elif model:
        dqn = model
    else:
        # dqn = d3rlpy.algos.DQNConfig(
        dqn = d3rlpy.algos.DoubleDQNConfig(
            batch_size=hyperparams['batch_size'], #Test smaller batch size: 32, 64. May be noisier
            learning_rate=hyperparams['learning_rate'],
            gamma=hyperparams['gamma'],
            target_update_interval=hyperparams['target_update_interval'] #Test with 1k, 2k, 5k
            ).create(device=hyperparams['gpu'])

    # Initialize empty FIFO buffer
    buffer = d3rlpy.dataset.ReplayBuffer(
        buffer=d3rlpy.dataset.FIFOBuffer(limit=hyperparams['buffer_size']),
        env=env,
    )

    # Load and merge offline data with type-checking
    if hyperparams['data_path']:
        try:
            # Load dataset with proper validation
            with open(hyperparams['data_path'], 'rb') as f:
                dataset = pickle.load(f)

            # Verify dataset structure
            if hasattr(dataset, 'episodes'):
                # Calculate safe episode count to load
                valid_episodes = min(hyperparams['n_pretrain_eps'], len(dataset.episodes))

                # Append episodes with transition validation
                for episode in dataset.episodes[:valid_episodes]:
                    if len(episode) > 0 and hasattr(episode, 'rewards'):
                        buffer.append_episode(episode)
                    else:
                        print(f"Skipping invalid episode: {episode}")
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")

    # Configure training with safety checks
    if buffer.transition_count == 0:
        print("Empty buffer (just Online training)!")
    dqn.fit_online(
        env=env,
        buffer=buffer,
        explorer=explorer,
        eval_env=eval_env,
        n_steps=hyperparams['n_steps'],
        n_steps_per_epoch=hyperparams['n_steps_per_epoch'],
        # update_interval=update_interval,
        # experiment_name="online_training",
    )

    # Extract rewards safely
    # rewards = []
    # for episode in buffer.episodes: # Only collect the online data
    #     rewards.append(episode.compute_return())

    # if hyperparams['cut_off_threshold']:
    #     start, end = hyperparams['cut_off_threshold']
    #     rewards = rewards[start:end]
    # return rewards
    return buffer.episodes

if __name__ == "__main__":
    hyperparams = {
        "env": "CartPole-v0", #"CartPole-v0", # "Acrobot-v0", "MountainCar-v0", "FrozenLake-v1", "CliffWalking-v0", "Taxi-v3", "RepresentedPong-v0"
        "seed": 42069,
        "n_episodes": 200,#5000,
        "max_episode_len": 50, # Around 10h per 100k steps in Leviathan server
        "eps": 0.1,  # epsilon for exploration
        "n_exp": 1,
        "n_pretrain_eps": 10,
        "n_online_eps": 490,
        "gpu": True, # True if use GPU to train with d3rlpy
        "buffer_size": 100000, #Test with 100k, 200k, 500k. 1M might be too much
        "data_path": None,#'data/RepresentedPong_Qwen2.5-7B-Instruct_Neps_500.pkl',
        "model_path": None,#'d3rlpy_loss/DoubleDQN_online_20250331153346/model_600000.d3',
        "batch_size":2048, #Test smaller batch size: 32, 64. May be noisier
        "learning_rate":1e-4,
        "gamma":0.999,
        "target_update_interval":1000 #Test with 1k, 2k, 5k
    }

    # d3rlpy supports both Gym and Gymnasium
    env = GymCompatWrapper2(gym.make(hyperparams['env']))
    eval_env = GymCompatWrapper2(gym.make(hyperparams['env']))
    # fix seed
    d3rlpy.seed(hyperparams['seed'])
    d3rlpy.envs.seed_env(env, hyperparams['seed'])
    d3rlpy.envs.seed_env(eval_env, hyperparams['seed'])
    np.random.seed(hyperparams['seed'])

    # onl_rewards_eps_decay = np.zeros((hyperparams['n_episodes'], hyperparams['n_exp']))

    # setup explorers
    # explorer = d3rlpy.algos.ConstantEpsilonGreedy(hyperparams['eps'])
    explorer = d3rlpy.algos.LinearDecayEpsilonGreedy(
        start_epsilon=1,
        end_epsilon=0.1,
        duration=20000,
    )

    hyperparams['n_steps'] = int(hyperparams['n_episodes']*hyperparams['max_episode_len']) # rough calculation
    hyperparams['n_steps_per_epoch'] = int(max(1, hyperparams['n_steps']//50))
    hyperparams['cut_off_threshold'] = (0,hyperparams['n_episodes'])

    for i in range(hyperparams['n_exp']):
        # onl_rewards_eps_decay[:,i]=online_training(env, eval_env, hyperparams, explorer)
        episodes = online_training(env, eval_env, hyperparams, explorer) # return buffer.episodes
        with open('data/finetune_'+hyperparams["env"].split('-')[0]+'_exp_'+str(i)+'.pkl', 'wb') as file:
            pickle.dump(episodes, file)

    # with open('data/finetune_RepresentedPong_Neps_500.pkl', 'rb') as file:
    #     onl_rewards_eps_decay = pickle.load(file)