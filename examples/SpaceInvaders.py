# [Translator classes and functions for Atari Space Invaders environment]

class ObsTranslator:
    def __init__(self):
        pass

    def translate(self, state):
        invaders_left_count, player_score, num_lives, player_x, enemies_x, missiles_y, enemies_y = state
        # breakpoint()
        enemies_positions = [enemies_x, enemies_y]
        missiles_positions = [missiles_y]

        return "\n".join([f"You are at position x={player_x}. ",
               f"Number of invaders left: {invaders_left_count}. ",
               f"Player's score: {player_score}. ",
               f"Number of lives left: {num_lives}. ",
               f"Enemies are at positions: {enemies_positions}. ",
               f"Missiles are at y positions: {missiles_positions}."])


class GameDescriber:
    def __init__(self, args):
        self.is_only_local_obs = args.is_only_local_obs == 1
        self.max_episode_len = args.max_episode_len
        self.args = args
        self.action_desc_dict = {
        }
        self.reward_desc_dict = {
        }

    def describe_goal(self):
        return "The goal is to destroy all the invaders while avoiding their attacks and protecting your lives."

    def translate_terminate_state(self, state, episode_len, max_episode_len):
        return f"Game over. Final score: {state[1]}."

    def translate_potential_next_state(self, state, action):
        enemies_positions = [state[4], state[6]]
        missiles_positions = [state[5]]
        return f"After taking the action, you might move to position x={state[3]} with the enemies at positions {enemies_positions} and missiles at y positions {missiles_positions}."

    def describe_game(self):
        return "\n".join(["In the Space Invaders game, you control a spaceship and aim to destroy all the invaders while avoiding their attacks and protecting your lives. ",
               f"There are {self.args.frameskip} frames per step and the action sticks for the skipping frames." if self.args.frameskip > 0 else "",
               "Scoring Points: Points are scored by destroying invaders. ",
               "You can control the movement and actions of your spaceship to dodge attacks and shoot at invaders to achieve a high score."])

    def describe_action(self):
        return "\n".join(["Type 1 for NOOP (no operation), 2 to FIRE (trigger fire button), 3 to move RIGHT, 4 to move LEFT, 5 to move RIGHT and FIRE, 6 to move LEFT and FIRE. ",
               "Ensure you only provide the action number from the valid action list, i.e., [1, 2, 3, 4, 5, 6]."])


# class TransitionTranslator(ObsTranslator):
#     def translate(self, infos, is_current=False):
#         descriptions = []
#         if is_current:
#             state_desc = ObsTranslator().translate(infos[-1]['state'])
#             return state_desc
#         for i, info in enumerate(infos):
#             assert 'state' in info, "info should contain state information"

#             state_desc = ObsTranslator().translate(info['state'])
#             action_desc = self.get_action_description(info['action'])
#             reward_desc = f"Result: Reward of {info['reward']}, "
#             next_state_desc = ObsTranslator().translate(info['next_state'])
#             descriptions.append(f"{state_desc}.\n {action_desc} \n {reward_desc} \n Transit to {next_state_desc}")
#         return descriptions

#     def get_action_description(self, action):
#         if action == 1:
#             return "Take Action: 'Do nothing (NOOP)'"
#         elif action == 2:
#             return "Take Action: 'FIRE'"
#         elif action == 3:
#             return "Take Action: 'Move RIGHT'"
#         elif action == 4:
#             return "Take Action: 'Move LEFT'"
#         elif action == 5:
#             return "Take Action: 'Move RIGHT and FIRE'"
#         elif action == 6:
#             return "Take Action: 'Move LEFT and FIRE'"
#         else:
#             return "Take Action: 'Invalid action'"

import os
from tqdm import trange
import wandb

from transformers import AutoTokenizer
from peft import LoraConfig
from trl import AutoModelForCausalLMWithValueHead

import re
import gymnasium as gym
from llamagym import Agent
import numpy as np
import d3rlpy
import pickle
import argparse

from env.atari import represented_atari_game
from env.translation_agent import SpaceInvadersAgent


if __name__ == "__main__":
    hyperparams = {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        # "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "env": "RepresentedSpaceInvaders-v0",#"ALE/SpaceInvaders-v5",
        "lora/target_modules": ["q_proj","up_proj","o_proj","k_proj","down_proj","gate_proj","v_proj"],
        "lora/r": 8,
        "lora/lora_alpha": 16,
        "lora/lora_dropout": 0.05,
        "lora/bias": "none",
        "lora/task_type": "CAUSAL_LM",
        "load_in_8bit": True,
        "batch_size": 4,
        "seed": 42069,
        "episodes": 5000,
        "generate/max_new_tokens": 32,
        "generate/do_sample": True,
        "generate/top_p": 0.6,
        "generate/top_k": 0,
        "generate/temperature": 0.9,
    }
    wandb_run = wandb.init(project=os.environ.get("WANDB_PROJECT"), config=hyperparams)
    device = "cuda:0"
    HF_TOKEN = os.environ.get("HF_TOKEN")

    lora_config = LoraConfig(
        **{
            key.split("/")[-1]: value
            for key, value in hyperparams.items()
            if key.startswith("lora/")
        }
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        pretrained_model_name_or_path=hyperparams["model_name"],
        peft_config=lora_config,
        load_in_8bit=hyperparams["load_in_8bit"],
        token=HF_TOKEN,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(hyperparams["model_name"], token=HF_TOKEN)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.pretrained_model.resize_token_embeddings(len(tokenizer))
    
    agent = SpaceInvadersAgent(
        model,
        tokenizer,
        device,
        {
            key: value
            for key, value in hyperparams.items()
            if key.startswith("generate/")
        },
        {
            "batch_size": hyperparams["batch_size"],
            "mini_batch_size": hyperparams["batch_size"],
        },
    )
    env = gym.make(hyperparams["env"])
    observations, actions, rewards, terminals = [], [], [], []
    for episode in trange(hyperparams["episodes"]):
        observation, info = env.reset()
        done = False

        while not done:
            action = agent.act(observation)
            wandb.log({"action": action})
            observation, reward, terminated, truncated, info = env.step(action)
            agent.assign_reward(reward)
            done = terminated or truncated
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            terminals.append(int(terminated))

        episode_stats = {
            "episode": episode,
            "mean_return": np.mean(agent.current_episode_rewards),
            "message_ct": len(agent.current_episode_messages),
            "episode_messages": agent.current_episode_messages,
        }
        train_stats = agent.terminate_episode(train=False)
        episode_stats.update(train_stats)
        wandb.log(episode_stats)

    dataset = d3rlpy.dataset.MDPDataset(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
        terminals=np.array(terminals),
    )
    with open("SpaceInvaders_"+hyperparams["model_name"]+'_eps_'+str(hyperparams['num_episodes'])+'.pkl', 'wb') as file:
        pickle.dump(dataset, file)