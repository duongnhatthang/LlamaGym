import argparse
import gymnasium as gym
import d3rlpy
import numpy as np

class BlackjackWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=-1, high=32, shape=(3,), dtype=np.int32)

    def observation(self, obs):
        return np.array(obs, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    # parser.add_argument("--env", type=str, default="Hopper-v2")
    parser.add_argument("--env", type=str, default="Blackjack-v1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    # d3rlpy supports both Gym and Gymnasium
    if args.env=="Blackjack-v1":
        env = BlackjackWrapper(gym.make("Blackjack-v1"))
        eval_env = BlackjackWrapper(gym.make("Blackjack-v1"))
    else:
        env = gym.make(args.env)
        eval_env = gym.make(args.env)

    # fix seed
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)
    d3rlpy.envs.seed_env(eval_env, args.seed)

    # setup algorithm
    dqn = d3rlpy.algos.DQNConfig(
        batch_size=32,
        learning_rate=6.25e-5,
        ).create(device=args.gpu)

    with open('Qwen2.5-0.5B_eps_5000.pkl', 'rb') as file:
        dataset = pickle.load(file)

    # start offline training
    dqn.fit(dataset, n_steps=100000)

    # replay buffer for experience replay
    buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=1000000, env=env)

    # start training
    dqn.fit_online(
        env,
        buffer,
        eval_env=eval_env,
        n_steps=1000000,
        n_steps_per_epoch=10000,
        update_interval=1,
        update_start_step=1000,
    )


if __name__ == "__main__":
    main()
