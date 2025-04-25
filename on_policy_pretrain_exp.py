import gymnasium as gym
import numpy as np
import d3rlpy
import pickle
from tqdm import trange
from datetime import datetime

# import matplotlib.pyplot as plt
from env.atari.represented_atari_game import GymCompatWrapper2
from d3rlpy.metrics import EnvironmentEvaluator
# import gymnasium.spaces
# from pretrain_from_llm import get_new_dataset
from online_main import OneHotWrapper, evaluate_qlearning_with_environment

def online_training_split(
    env,
    eval_env,
    hyperparams,
    explorer=None
):
    if isinstance(env.action_space, gym.spaces.Box):  # Continuous action space
        algo_config = d3rlpy.algos.SACConfig(
            batch_size=hyperparams['batch_size'],
            # learning_rate=hyperparams['learning_rate'],
            gamma=hyperparams['gamma'],
            # target_update_interval=hyperparams['target_update_interval']
        )
    else:  # Discrete action space
        algo_config = d3rlpy.algos.DoubleDQNConfig(
            batch_size=hyperparams['batch_size'],
            learning_rate=hyperparams['learning_rate'],
            gamma=hyperparams['gamma'],
            target_update_interval=hyperparams['target_update_interval']
        )
    dqn = algo_config.create(device=hyperparams['gpu'])

    # Initialize empty FIFO buffer
    buffer = d3rlpy.dataset.ReplayBuffer(
        buffer=d3rlpy.dataset.FIFOBuffer(limit=hyperparams['buffer_size']),
        env=env,
    )

    rewards = []
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    for _ in trange(hyperparams['n_pretrain_eps']):
        dqn.fit_online(
            env=env,
            buffer=buffer,
            explorer=explorer,
            n_steps=hyperparams['max_episode_len'],
            experiment_name=f"{timestamp}_online_training",
        )
        if hyperparams['env'] == "CliffWalking-v0":
            r=evaluate_qlearning_with_environment(dqn, eval_env, hyperparams["max_episode_len"])
        else:
            env_evaluator = EnvironmentEvaluator(env, n_trials=1)
            r = env_evaluator(dqn, dataset=None)
        rewards.append(r)

    dqn.fit(buffer, n_steps=hyperparams["n_pretrain_steps"], n_steps_per_epoch=hyperparams['n_steps_per_epoch'])

    for _ in trange(hyperparams['n_online_eps']):
        dqn.fit_online(
            env=env,
            buffer=buffer,
            explorer=explorer,
            n_steps=hyperparams['max_episode_len'],
            experiment_name=f"{timestamp}_online_training",
        )
        if hyperparams['env'] == "CliffWalking-v0":
            r=evaluate_qlearning_with_environment(dqn, eval_env, hyperparams["max_episode_len"])
        else:
            env_evaluator = EnvironmentEvaluator(env, n_trials=1)
            r = env_evaluator(dqn, dataset=None)
        rewards.append(r)
    return rewards

def online_training_rand(
    env,
    eval_env,
    hyperparams,
    explorer=None
):
    """
    Same as online_training_split, but with random actions for the first n_pretrain_eps episodes.
    """
    if isinstance(env.action_space, gym.spaces.Box):  # Continuous action space
        algo_config = d3rlpy.algos.SACConfig(
            batch_size=hyperparams['batch_size'],
            # learning_rate=hyperparams['learning_rate'],
            gamma=hyperparams['gamma'],
            # target_update_interval=hyperparams['target_update_interval']
        )
    else:  # Discrete action space
        algo_config = d3rlpy.algos.DoubleDQNConfig(
            batch_size=hyperparams['batch_size'],
            learning_rate=hyperparams['learning_rate'],
            gamma=hyperparams['gamma'],
            target_update_interval=hyperparams['target_update_interval']
        )
    dqn = algo_config.create(device=hyperparams['gpu'])

    # Initialize empty FIFO buffer
    buffer = d3rlpy.dataset.ReplayBuffer(
        buffer=d3rlpy.dataset.FIFOBuffer(limit=hyperparams['buffer_size']),
        env=env,
    )

    rewards = []
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    observations, actions, pretrain_rewards, terminals = [], [], [], []
    for _ in range(hyperparams["n_pretrain_eps"]):
        env.reset()
        done = False
        eps_reward = 0
        count = 0
        while not done:
            action = env.action_space.sample()
            observation, reward, done, _, info = env.step(action)
            if count >= hyperparams["max_episode_len"]:
                done = True
            observations.append(observation)
            actions.append(action)
            pretrain_rewards.append(reward)
            terminals.append(int(done))
            eps_reward += reward
            count += 1
        rewards.append(eps_reward)
    pretrain_dataset = d3rlpy.dataset.MDPDataset(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(pretrain_rewards),
        terminals=np.array(terminals),
    )
    for episode in pretrain_dataset.episodes:
        if len(episode) > 0 and hasattr(episode, 'rewards'):
            buffer.append_episode(episode)
        else:
            print(f"Skipping invalid episode: {episode}")

    dqn.fit(buffer, n_steps=hyperparams["n_pretrain_steps"], n_steps_per_epoch=hyperparams['n_steps_per_epoch'])

    for _ in trange(hyperparams['n_online_eps']):
        dqn.fit_online(
            env=env,
            buffer=buffer,
            explorer=explorer,
            n_steps=hyperparams['max_episode_len'],
            experiment_name=f"{timestamp}_online_training",
        )
        if hyperparams['env'] == "CliffWalking-v0":
            r=evaluate_qlearning_with_environment(dqn, eval_env, hyperparams["max_episode_len"])
        else:
            env_evaluator = EnvironmentEvaluator(env, n_trials=1)
            r = env_evaluator(dqn, dataset=None)
        rewards.append(r)
    return rewards

if __name__ == "__main__":
    hyperparams = {
        "env": "Pendulum-v1", #"CartPole-v0", # "Acrobot-v0", "MountainCar-v0", "FrozenLake-v1", "CliffWalking-v0", "Taxi-v3", "RepresentedPong-v0"
        "seed": 42069,
        "n_episodes": 200,#5000,
        "max_episode_len": 200, # Around 10h per 100k steps in Leviathan server
        "eps": 0.1,  # epsilon for exploration
        "n_exp": 5,
        "n_pretrain_eps": 30,
        "n_online_eps": 170, #10-5990 for mountainCar, 30-120 for CartPole, 30-120 for FrozenLake
        "gpu": True, # True if use GPU to train with d3rlpy
        "buffer_size": 100000, #Test with 100k, 200k, 500k. 1M might be too much
        "data_path": None,#'data/CartPole_Qwen2.5-7B-Instruct_Neps_10_20250406040150.pkl',
        "model_path": None,#'d3rlpy_loss/DoubleDQN_online_20250331153346/model_600000.d3',
        "batch_size":256, #Test smaller batch size: 32, 64. May be noisier
        "learning_rate":5e-5,
        "gamma":0.99,
        "target_update_interval":1000, #Test with 1k, 2k, 5k
        "n_steps_per_epoch": 200,
        "n_pretrain_steps": 1000
    }
    assert hyperparams["n_episodes"]==hyperparams["n_pretrain_eps"]+hyperparams["n_online_eps"], "Check n_episodes=n_pretrain_eps+n_online_eps"

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

    cache = {}
    def run_exp(n_pretrain_steps, n_pretrain_eps, cache, env, eval_env, hyperparams, explorer):
        hyperparams["n_pretrain_steps"]=n_pretrain_steps
        hyperparams["n_pretrain_eps"]=n_pretrain_eps
        hyperparams["n_online_eps"]=hyperparams["n_episodes"]-hyperparams["n_pretrain_eps"]
        for i in range(hyperparams['n_exp']):
            cache[f'pretrain_{hyperparams["n_pretrain_eps"]}_eps_{hyperparams["n_pretrain_steps"]}_steps_{i}_rand'] = online_training_rand(env, eval_env, hyperparams, explorer)
            # cache[f'pretrain_{hyperparams["n_pretrain_eps"]}_eps_{hyperparams["n_pretrain_steps"]}_steps_{i}'] = online_training_split(env, eval_env, hyperparams, explorer)
        return cache
    cache = run_exp(1000, 30, cache, env, eval_env, hyperparams, explorer)
    cache = run_exp(1000, 20, cache, env, eval_env, hyperparams, explorer)
    cache = run_exp(1000, 10, cache, env, eval_env, hyperparams, explorer)
    cache = run_exp(3000, 30, cache, env, eval_env, hyperparams, explorer)
    cache = run_exp(3000, 20, cache, env, eval_env, hyperparams, explorer)
    cache = run_exp(3000, 10, cache, env, eval_env, hyperparams, explorer)

    with open(f'data/cache_{hyperparams["env"].split("-")[0]}_on_policy_pretrain_exp_rand.pkl', 'wb') as file:
    # with open(f'data/cache_{hyperparams["env"].split("-")[0]}_on_policy_pretrain_exp.pkl', 'wb') as file:
        pickle.dump(cache, file)