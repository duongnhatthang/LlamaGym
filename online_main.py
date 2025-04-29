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
from pretrain_from_llm import get_new_dataset

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

def evaluate_qlearning_with_environment(
    algo,
    env,
    max_episode_len,
    n_trials: int = 10,
    epsilon: float = 0.0,
) -> float:
    """
    From d3rlpy.metrics.EnvironmentEvaluator
    Modified because the original code bugged out on CliffWalking-v0. The episode never end.
    
    Returns average environment score.

    .. code-block:: python

        import gym

        from d3rlpy.algos import DQN
        from d3rlpy.metrics.utility import evaluate_with_environment

        env = gym.make('CartPole-v0')

        cql = CQL()

        mean_episode_return = evaluate_with_environment(cql, env)


    Args:
        alg: algorithm object.
        env: gym-styled environment.
        n_trials: the number of trials.
        epsilon: noise factor for epsilon-greedy policy.

    Returns:
        average score.
    """
    episode_rewards = []
    for _ in range(n_trials):
        observation, _ = env.reset()
        episode_reward = 0.0
        count=0
        while True:
            # take action
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                if isinstance(observation, np.ndarray):
                    observation = np.expand_dims(observation, axis=0)
                elif isinstance(observation, (tuple, list)):
                    observation = [
                        np.expand_dims(o, axis=0) for o in observation
                    ]
                else:
                    raise ValueError(
                        f"Unsupported observation type: {type(observation)}"
                    )
                action = algo.predict(observation)[0]
            observation, reward, done, truncated, _ = env.step(action)
            episode_reward += float(reward)
            count+=1
            if count >= max_episode_len:
                done = True

            if done or truncated:
                break
        episode_rewards.append(episode_reward)
    return float(np.mean(episode_rewards))

def online_training(
    env,
    eval_env,
    hyperparams,
    explorer=None,
    model=None
):
    # Determine the algorithm based on the action space
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

    # Load model with proper validation
    if hyperparams['model_path']:
        dqn = d3rlpy.load_learnable(hyperparams['model_path'])
    elif model:
        dqn = model
    else:
        dqn = algo_config.create(device=hyperparams['gpu'])

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
            dataset_new = get_new_dataset(dataset, n_pretrain_eps)

            # Append episodes with transition validation
            for episode in dataset_new.episodes:
                if len(episode) > 0 and hasattr(episode, 'rewards'):
                    buffer.append_episode(episode)
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
    for _ in trange(n_pretrain_eps + hyperparams['n_online_eps']):
        # Ensure we have enough transitions in buffer before training
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
        "env": "CliffWalking-v1", #"CartPole-v0", # "Acrobot-v0", "MountainCar-v0", "FrozenLake-v1", Pendulum-v1, "CliffWalking-v0", "Taxi-v3", "RepresentedPong-v0"
        "seed": 42069,
        "n_episodes": 200,#5000,
        "max_episode_len": 200, # Around 10h per 100k steps in Leviathan server
        "eps": 0.1,  # epsilon for exploration
        "n_exp": 5,
        "n_pretrain_eps": 30,
        "n_online_eps": 170, #10-290 for mountainCar, 30-120 for CartPole, 30-120 for FrozenLake, 30-570 for Pendulum
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

    # with open(f'models/{hyperparams["env"].split("-")[0]}_ddqn_pretrain_7b_1000_steps_{hyperparams["n_pretrain_eps"]}SFT.pkl', 'rb') as file:
    #     pretrain_7b_1000_dqn = pickle.load(file)
    # with open(f'models/{hyperparams["env"].split("-")[0]}_ddqn_pretrain_7b_3000_steps_{hyperparams["n_pretrain_eps"]}SFT.pkl', 'rb') as file:
    #     pretrain_7b_3000_dqn = pickle.load(file)
    # with open(f'models/{hyperparams["env"].split("-")[0]}_ddqn_pretrain_32b_1000_steps_{hyperparams["n_pretrain_eps"]}DS.pkl', 'rb') as file:
    #     pretrain_32b_1000_dqn = pickle.load(file)
    # with open(f'models/{hyperparams["env"].split("-")[0]}_ddqn_pretrain_7b_1000_steps_{hyperparams["n_pretrain_eps"]}DS.pkl', 'rb') as file:
    #     pretrain_7b_1000_dqn = pickle.load(file)
    # with open(f'models/{hyperparams["env"].split("-")[0]}_ddqn_pretrain_32b_3000_steps_{hyperparams["n_pretrain_eps"]}DS.pkl', 'rb') as file:
    #     pretrain_32b_3000_dqn = pickle.load(file)
    # with open(f'models/{hyperparams["env"].split("-")[0]}_ddqn_pretrain_7b_3000_steps_{hyperparams["n_pretrain_eps"]}DS.pkl', 'rb') as file:
    #     pretrain_7b_3000_dqn = pickle.load(file)
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


    # Mixed pretraining and online data. finetune is a lagacy name.
    hyperparams['data_path'] = f"data/{hyperparams['env'].split('-')[0]}_Qwen2.5-7B-Instruct_Neps_{hyperparams['n_pretrain_eps']}_20250420103044.pkl" #Pong
    # hyperparams['data_path'] = f"data/{hyperparams['env'].split('-')[0]}_Qwen2.5-7B-Instruct_Neps_{hyperparams['n_pretrain_eps']}_20250412171921.pkl" #CliffWalkingTypo
    # hyperparams['data_path'] = f"data/{hyperparams['env'].split('-')[0]}_Qwen2.5-7B-Instruct_Neps_{hyperparams['n_pretrain_eps']}_20250415095844.pkl" #MountainCar
    # hyperparams['data_path'] = f"data/{hyperparams['env'].split('-')[0]}_Qwen2.5-7B-Instruct_Neps_{hyperparams['n_pretrain_eps']}_20250413234248.pkl" #Pendulum
    # hyperparams['data_path'] = f"data/{hyperparams['env'].split('-')[0]}_Qwen2.5-7B-Instruct_Neps_{hyperparams['n_pretrain_eps']}_20250412075104.pkl" #FrozenLakeTypo
    # hyperparams['data_path'] = f"data/{hyperparams['env'].split('-')[0]}_Qwen2.5-7B-Instruct_Neps_{hyperparams['n_pretrain_eps']}_20250410211529.pkl" #CartPole
    # hyperparams['data_path'] = f"data/{hyperparams['env'].split('-')[0]}_Qwen2.5-7B-Instruct_Neps_{hyperparams['n_pretrain_eps']}_20250411000858B.pkl" #FrozenLakeBug
    # hyperparams['data_path'] = f"data/{hyperparams['env'].split('-')[0]}_Qwen2.5-7B-Instruct_Neps_{hyperparams['n_pretrain_eps']}_20250408032240_withEps.pkl" #CartPole with Eps
    for i in range(hyperparams['n_exp']):
        cache[f'finetune_7b_{i}'] = online_training(env, eval_env, hyperparams, explorer)

    hyperparams['data_path'] = f"data/{hyperparams['env'].split('-')[0]}_Qwen2.5-32B-Instruct_Neps_{hyperparams['n_pretrain_eps']}_20250420162547.pkl" #Pong
    # hyperparams['data_path'] = f"data/{hyperparams['env'].split('-')[0]}_Qwen2.5-32B-Instruct_Neps_{hyperparams['n_pretrain_eps']}_20250415065446.pkl" #CliffWalkingTypo
    # hyperparams['data_path'] = f"data/{hyperparams['env'].split('-')[0]}_Qwen2.5-32B-Instruct_Neps_{hyperparams['n_pretrain_eps']}_20250413081613.pkl" #MountainCar
    # hyperparams['data_path'] = f"data/{hyperparams['env'].split('-')[0]}_Qwen2.5-32B-Instruct_Neps_{hyperparams['n_pretrain_eps']}_20250414014508.pkl" #Pendulum
    # hyperparams['data_path'] = f"data/{hyperparams['env'].split('-')[0]}_Qwen2.5-32B-Instruct_Neps_{hyperparams['n_pretrain_eps']}_20250412120230.pkl" #FrozenLakeTypo
    # hyperparams['data_path'] = f"data/{hyperparams['env'].split('-')[0]}_Qwen2.5-32B-Instruct_Neps_{hyperparams['n_pretrain_eps']}_20250412032827.pkl" #CartPole
    # hyperparams['data_path'] = f"data/{hyperparams['env'].split('-')[0]}_Qwen2.5-32B-Instruct_Neps_{hyperparams['n_pretrain_eps']}_20250411030422B.pkl" #FrozenLakeBug
    # hyperparams['data_path'] = f"data/{hyperparams['env'].split('-')[0]}_Qwen2.5-32B-Instruct_Neps_{hyperparams['n_pretrain_eps']}_20250408120007_withEps.pkl" #CartPole with Eps
    for i in range(hyperparams['n_exp']):
        cache[f'finetune_32b_{i}'] = online_training(env, eval_env, hyperparams, explorer)

    # with open(f'data/cache_{hyperparams["env"].split("-")[0]}_Neps_{hyperparams["n_pretrain_eps"]}SFT.pkl', 'wb') as file:
    # with open(f'data/cache_{hyperparams["env"].split("-")[0]}_Neps_{hyperparams["n_pretrain_eps"]}DS.pkl', 'wb') as file:
    with open(f'data/cache_{hyperparams["env"].split("-")[0]}_Neps_{hyperparams["n_pretrain_eps"]}.pkl', 'wb') as file:
        pickle.dump(cache, file)