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
        with open(hyperparams['model_path'], 'rb') as file:
            dqn = pickle.load(file)
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
    rewards = []
    for episode in buffer.episodes: # Only collect the online data
        # if hasattr(episode, 'rewards') and episode.rewards.size > 0:
        #     rewards.extend(episode.rewards.flatten().tolist())
        rewards.append(episode.returns)

    if hyperparams['cut_off_threshold']:
        start, end = hyperparams['cut_off_threshold']
        rewards = rewards[start:end]
    return rewards

if __name__ == "__main__":
    hyperparams = {
        "env": "RepresentedPong-v0",
        "batch_size": 4,
        "seed": 42069,
        "n_episodes": 500,#5000,
        "max_episode_len": 500, # Around 10h per 100k steps in Leviathan server
        "eps": 0.3,  # epsilon for exploration
        "n_exp": 1,
        "n_pretrain_eps": 500,
        "n_online_eps": 500,
        "gpu": True, # True if use GPU to train with d3rlpy
        "buffer_size": 100000, #Test with 100k, 200k, 500k. 1M might be too much
        "data_path": 'data/RepresentedPong_Qwen2.5-7B-Instruct_Neps_500.pkl',
        "model_path": None,
        "batch_size":512, #Test smaller batch size: 32, 64. May be noisier
        "learning_rate":1e-4,
        "gamma":0.999,
        "target_update_interval":1000 #Test with 1k, 2k, 5k
    }

    # d3rlpy supports both Gym and Gymnasium
    env = GymCompatWrapper2(gym.make(hyperparams['env']))
    eval_env = GymCompatWrapper2(gym.make(hyperparams['env']))
    # n_exp = 3
    # llama_LORO_rewards = np.zeros((n_pretrain_eps + n_online_eps, n_exp))
    # Qwen_LORO_rewards = np.zeros((n_pretrain_eps + n_online_eps, n_exp))
    # rand_LORO_rewards = np.zeros((n_pretrain_eps + n_online_eps, n_exp))
    # onl_rewards = np.zeros((n_pretrain_eps + n_online_eps, n_exp))
    # onl_rewards_eps = np.zeros((n_pretrain_eps + n_online_eps, n_exp))
    # onl_rewards_eps_decay = np.zeros((n_pretrain_eps + n_online_eps, n_exp))


    # onl_rewards = np.zeros((hyperparams['n_episodes'], hyperparams['n_exp']))
    # onl_rewards_eps = np.zeros((hyperparams['n_episodes'], hyperparams['n_exp']))
    onl_rewards_eps_decay = np.zeros((hyperparams['n_episodes'], hyperparams['n_exp']))

    # setup explorers
    # const_explorer = d3rlpy.algos.ConstantEpsilonGreedy(hyperparams['eps'])
    decay_explorer = d3rlpy.algos.LinearDecayEpsilonGreedy(
        start_epsilon=0.5,
        end_epsilon=0.1,
        duration=250000,
    )

    hyperparams['n_steps'] = int(hyperparams['n_episodes']*hyperparams['max_episode_len']*1.2) # rough calculation
    hyperparams['n_steps_per_epoch'] = int(max(1, hyperparams['n_steps']//100))
    hyperparams['cut_off_threshold'] = (0,hyperparams['n_episodes'])

    for i in range(hyperparams['n_exp']):
        # onl_rewards[:,i] = online_training(env, eval_env, n_steps=n_steps)
        # onl_rewards_eps[:,i] = online_training(env, eval_env, const_explorer, n_steps=n_steps)
        onl_rewards_eps_decay[:,i]=online_training(env, eval_env, hyperparams, decay_explorer)

    with open('finetune_'+hyperparams["env"].split('-')[0]+'_Neps_'+str(hyperparams['n_episodes'])+'.pkl', 'wb') as file:
    # with open(hyperparams["env"].split('-')[0]+'_Neps_'+str(hyperparams['n_episodes'])+'.pkl', 'wb') as file:
        pickle.dump(onl_rewards_eps_decay, file)

    # with open('data/RepresentedPong_Neps_1.pkl', 'rb') as file:
    #     onl_rewards_eps_decay = pickle.load(file)