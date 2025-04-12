import gymnasium as gym
import numpy as np
import d3rlpy
import pickle
from tqdm import trange
from datetime import datetime

import matplotlib.pyplot as plt
from env.atari.represented_atari_game import GymCompatWrapper2
from d3rlpy.metrics import EnvironmentEvaluator
import gymnasium.spaces

class OneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete), "Only Discrete observation spaces are supported."
        self.n = env.observation_space.n
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.n,), dtype=np.float32
        )

    def observation(self, obs):
        one_hot = np.zeros((self.n,), dtype=np.float32)
        one_hot[obs] = 1.0
        return one_hot

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

    n_pretrain_eps = hyperparams['n_pretrain_eps']
    # Load and merge offline data with type-checking
    if hyperparams['data_path']:
        try:
            # Load dataset with proper validation
            with open(hyperparams['data_path'], 'rb') as f:
                dataset = pickle.load(f)

            # Append episodes with transition validation
            for episode in dataset.episodes:
                if len(episode) > 0 and hasattr(episode, 'rewards'):
                    buffer.append_episode(episode)
                    # n_pretrain_steps += len(episode)
                    n_pretrain_eps -= 1
                else:
                    print(f"Skipping invalid episode: {episode}")
                if n_pretrain_eps <= 0:
                    break
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")

    # Configure training with safety checks
    if buffer.transition_count == 0:
        print("Empty buffer (just Online training)!")

    rewards = []
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    for _ in trange(n_pretrain_eps+hyperparams['n_online_eps']):
        # Ensure we have enough transitions in buffer before training
        dqn.fit_online(
            env=env,
            buffer=buffer,
            explorer=explorer,
            # eval_env=eval_env,
            n_steps=hyperparams['max_episode_len'],
            # n_steps_per_epoch=hyperparams['n_steps_per_epoch'],
            # update_interval=update_interval,
            experiment_name=f"{timestamp}_online_training",
        )
        env_evaluator = EnvironmentEvaluator(eval_env)
        rewards.append(env_evaluator(dqn, dataset=None))
    return rewards

if __name__ == "__main__":
    hyperparams = {
        "env": "FrozenLake-v1", #"CartPole-v0", # "Acrobot-v0", "MountainCar-v0", "FrozenLake-v1", "CliffWalking-v0", "Taxi-v3", "RepresentedPong-v0"
        "seed": 42069,
        "n_episodes": 200,#5000,
        "max_episode_len": 200, # Around 10h per 100k steps in Leviathan server
        "eps": 0.1,  # epsilon for exploration
        "n_exp": 5,
        "n_pretrain_eps": 10,
        "n_online_eps": 290, #10-290 for mountainCar, 30-170 for CartPole, 30-270 for FrozenLake
        "gpu": True, # True if use GPU to train with d3rlpy
        "buffer_size": 100000, #Test with 100k, 200k, 500k. 1M might be too much
        "data_path": None,#'data/CartPole_Qwen2.5-7B-Instruct_Neps_10_20250406040150.pkl',
        "model_path": None,#'d3rlpy_loss/DoubleDQN_online_20250331153346/model_600000.d3',
        "batch_size":256, #Test smaller batch size: 32, 64. May be noisier
        "learning_rate":5e-5,
        "gamma":0.99,
        "target_update_interval":1000 #Test with 1k, 2k, 5k
    }

    # d3rlpy supports both Gym and Gymnasium
    if "Represented" in hyperparams["env"]:
        env = GymCompatWrapper2(gym.make(hyperparams['env']))
        eval_env = GymCompatWrapper2(gym.make(hyperparams['env']))
    elif isinstance(gym.make(hyperparams["env"]).observation_space, gym.spaces.Discrete):
        env = OneHotWrapper(gym.make(hyperparams["env"]))
        eval_env = OneHotWrapper(gym.make(hyperparams["env"]))
    else:
        env = gym.make(hyperparams["env"])
        eval_env = gym.make(hyperparams["env"])
    # fix seed
    d3rlpy.seed(hyperparams['seed'])
    d3rlpy.envs.seed_env(env, hyperparams['seed'])
    d3rlpy.envs.seed_env(eval_env, hyperparams['seed'])
    np.random.seed(hyperparams['seed'])

    # setup explorers
    explorer = d3rlpy.algos.ConstantEpsilonGreedy(hyperparams['eps'])
    # explorer = d3rlpy.algos.LinearDecayEpsilonGreedy(
    #     start_epsilon=1,
    #     end_epsilon=0.1,
    #     duration=5000,
    # )

    cache = {}

    with open(f'models/{hyperparams["env"].split("-")[0]}_ddqn_pretrain_32b_1000_steps_{hyperparams["n_pretrain_eps"]}.pkl', 'rb') as file:
        pretrain_32b_1000_dqn = pickle.load(file)
    with open(f'models/{hyperparams["env"].split("-")[0]}_ddqn_pretrain_7b_1000_steps_{hyperparams["n_pretrain_eps"]}.pkl', 'rb') as file:
        pretrain_7b_1000_dqn = pickle.load(file)
    with open(f'models/{hyperparams["env"].split("-")[0]}_ddqn_pretrain_32b_3000_steps_{hyperparams["n_pretrain_eps"]}.pkl', 'rb') as file:
        pretrain_32b_3000_dqn = pickle.load(file)
    with open(f'models/{hyperparams["env"].split("-")[0]}_ddqn_pretrain_7b_3000_steps_{hyperparams["n_pretrain_eps"]}.pkl', 'rb') as file:
        pretrain_7b_3000_dqn = pickle.load(file)

    tmp_n_pretrain_eps = hyperparams['n_pretrain_eps']
    hyperparams['data_path'] =  None
    hyperparams['n_pretrain_eps'] = 0 # Set to 0 to avoid pretraining when using pre-trained models
    for i in range(hyperparams['n_exp']):
        cache[f'pretrain_7b_1000_{i}'] = online_training(env, eval_env, hyperparams, explorer, pretrain_7b_1000_dqn)
    for i in range(hyperparams['n_exp']):
        cache[f'pretrain_32b_1000_{i}'] = online_training(env, eval_env, hyperparams, explorer, pretrain_32b_1000_dqn)
    for i in range(hyperparams['n_exp']):
        cache[f'pretrain_7b_3000_{i}'] = online_training(env, eval_env, hyperparams, explorer, pretrain_7b_3000_dqn)
    for i in range(hyperparams['n_exp']):
        cache[f'pretrain_32b_3000_{i}'] = online_training(env, eval_env, hyperparams, explorer, pretrain_32b_3000_dqn)

    hyperparams['n_pretrain_eps'] = tmp_n_pretrain_eps # restore n_pretrain_eps for subsequent runs
    for i in range(hyperparams['n_exp']):
        cache[f'online_{i}'] = online_training(env, eval_env, hyperparams, explorer)

    hyperparams['data_path'] = f"data/{hyperparams['env'].split('-')[0]}_Qwen2.5-7B-Instruct_Neps_{hyperparams['n_pretrain_eps']}_20250411000858.pkl" #FrozenLake
    # hyperparams['data_path'] = f"data/{hyperparams['env'].split('-')[0]}_Qwen2.5-7B-Instruct_Neps_{hyperparams['n_pretrain_eps']}_20250408032240_withEps.pkl" #CartPole with Eps
    for i in range(hyperparams['n_exp']):
        cache[f'finetune_7b_{i}'] = online_training(env, eval_env, hyperparams, explorer)
        
    hyperparams['data_path'] = f"data/{hyperparams['env'].split('-')[0]}_Qwen2.5-32B-Instruct_Neps_{hyperparams['n_pretrain_eps']}_20250411030422.pkl" #FrozenLake
    # hyperparams['data_path'] = f"data/{hyperparams['env'].split('-')[0]}_Qwen2.5-32B-Instruct_Neps_{hyperparams['n_pretrain_eps']}_20250408120007_withEps.pkl" #CartPole with Eps
    for i in range(hyperparams['n_exp']):
        cache[f'finetune_32b_{i}'] = online_training(env, eval_env, hyperparams, explorer)

    with open(f'data/cache_{hyperparams["env"].split("-")[0]}_Neps_{hyperparams["n_pretrain_eps"]}.pkl', 'wb') as file:
        pickle.dump(cache, file)