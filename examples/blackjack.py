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


class BlackjackAgent(Agent):
    def get_system_prompt(self) -> str:
        return """You are an expert blackjack player. Every turn, you'll see your current sum, the dealer's showing card value, and whether you have a usable ace. Win by exceeding the dealer's hand but not exceeding 21.
Decide whether to stay with your current sum by writing "Action: 0" or accept another card by writing "Action: 1". Accept a card unless very close to 21."""

    def format_observation(self, observation: gym.core.ObsType) -> str:
        return f"You: {observation[0]}. Dealer: {observation[1]}. You have {'an' if bool(observation[2]) else 'no'} ace."

    def extract_action(self, response: str) -> gym.core.ActType:
        match = re.compile(r"Action: (\d)").search(response)
        if match:
            return int(match.group(1))

        digits = [char for char in response if char.isdigit()]
        if len(digits) == 0 or digits[-1] not in ("0", "1"):
            if "stick" in response.lower():
                return 0
            elif "hit" in response.lower():
                return 1

        return 0


if __name__ == "__main__":
    hyperparams = {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        # "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "env": "Blackjack-v1",
        "lora/target_modules": [
            "q_proj",
            "up_proj",
            "o_proj",
            "k_proj",
            "down_proj",
            "gate_proj",
            "v_proj",
        ],
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
        "eps": 0.3,  # epsilon for exploration
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

    agent = BlackjackAgent(
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
    env = gym.make(hyperparams["env"], natural=False, sab=False)
    observations, actions, rewards, terminals = [], [], [], []
    for episode in trange(hyperparams["episodes"]):
        observation, info = env.reset()
        done = False

        while not done:
            rand = bool(np.random.binomial(n=1, p=hyperparams["eps"]))
            if rand:
                action = env.action_space.sample()
            else:
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
    with open(
        hyperparams["model_name"] + "_eps_" + str(hyperparams["episodes"]) + ".pkl",
        "wb",
    ) as file:
        pickle.dump(dataset, file)
