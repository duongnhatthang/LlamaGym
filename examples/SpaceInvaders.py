import os
from tqdm import trange
import wandb

from transformers import AutoTokenizer
from peft import LoraConfig
from trl import AutoModelForCausalLMWithValueHead

import gymnasium as gym
import numpy as np
import d3rlpy
import pickle

from env.translation_agent import SpaceInvadersAgent


if __name__ == "__main__":
    hyperparams = {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        # "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
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
        "episodes": 2,#5000,
        "generate/max_new_tokens": 32,
        "generate/do_sample": True,
        "generate/top_p": 0.6,
        "generate/top_k": 0,
        "generate/temperature": 0.9,
        "max_episode_len": 100000 # Around 10h per 100k steps in Leviathan server
    }
    # wandb_run = wandb.init(project=os.environ.get("WANDB_PROJECT"), config=hyperparams)
    device = "cuda"
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
        n_step = 0
        while not done:
            action = agent.act(observation)
            # wandb.log({"action": action})
            observation, reward, terminated, info = env.step(action)
            agent.assign_reward(reward)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            terminals.append(int(terminated))
            n_step += 1
            if n_step > 0 and n_step % 1000 == 0:
                print(f"Episode {episode}, Step {n_step}, max_episode_len: {hyperparams['max_episode_len']}")
            if n_step >= hyperparams["max_episode_len"]:
                done = True

        episode_stats = {
            "episode": episode,
            "mean_return": np.mean(agent.current_episode_rewards),
            "message_ct": len(agent.current_episode_messages),
            "episode_messages": agent.current_episode_messages,
        }
        train_stats = agent.terminate_episode(train=False)
        episode_stats.update(train_stats)
        # wandb.log(episode_stats)

    dataset = d3rlpy.dataset.MDPDataset(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
        terminals=np.array(terminals),
    )
    with open("SpaceInvaders_no_hist_"+hyperparams["model_name"]+'_eps_'+str(hyperparams['episodes'])+'.pkl', 'wb') as file:
        pickle.dump(dataset, file)