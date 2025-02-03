import os
import re
from collections import Counter
from tqdm import trange
import wandb
import gymnasium as gym
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llamagym.major_agent import MajorityVoteAgent


class BlackjackAgent(MajorityVoteAgent):
    def get_system_prompt(self) -> str:
        return (
            "You are an expert blackjack player. Every turn, you'll see your current sum, "
            "the dealer's showing card value, and whether you have a usable ace. Win by exceeding "
            "the dealer's hand but not exceeding 21.\n"
            'Decide whether to stay with your current sum by writing "Action: 0" or accept another '
            'card by writing "Action: 1". Accept a card unless very close to 21.'
        )

    def format_observation(self, observation: gym.core.ObsType) -> str:
        return f"You: {observation[0]}. Dealer: {observation[1]}. You have {'an' if bool(observation[2]) else 'no'} ace."

    def extract_action(self, response: str) -> gym.core.ActType:
        match = re.search(r"Action: (\d)", response)
        if match:
            return int(match.group(1))
        if "stick" in response.lower():
            return 0
        elif "hit" in response.lower():
            return 1
        print("Invalid response, defaulting to stay.")
        return 0

if __name__ == "__main__":
    hyperparams = {
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "env": "Blackjack-v1",
        "batch_size": 8,
        "seed": 42069,
        "episodes": 5000,
        "generate/max_new_tokens": 32,
        "generate/do_sample": True,
        "generate/top_p": 0.6,
        "generate/top_k": 0,
        "generate/temperature": 0.9,
        "num_votes": 5  # Majority vote count
    }

    # W&B stuff here
    wandb_run = wandb.init(project=os.environ.get("WANDB_PROJECT"), config=hyperparams)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    HF_TOKEN = os.environ.get("HF_TOKEN")

    # Model & Tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=hyperparams["model_name"],
        token=HF_TOKEN,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(hyperparams["model_name"], token=HF_TOKEN)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))

    agent = BlackjackAgent(
        model,
        tokenizer,
        device,
        {key.split("/")[-1]: value for key, value in hyperparams.items() if key.startswith("generate/")},
        num_votes=hyperparams["num_votes"]
    )
    env = gym.make(hyperparams["env"], natural=False, sab=False)


    total_reward = 0
    # Inference Loop (No PPO, No Fine-tuning)
    for episode in trange(hyperparams["episodes"]):
        observation, info = env.reset()
        done = False
        steps = 0

        while not done:
            action = agent.act(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

        episode_stats = {
            "episode": episode,
            "total_return": total_reward,
            "steps": steps,
            "message_count": len(agent.current_episode_messages),
        }

        train_stats = agent.terminate_episode(train=False)  # [TODO] !! reset after each episode. @Thang, plz check the reset message here.
        episode_stats.update(train_stats)

        print("AAA", total_reward)

        wandb.log(episode_stats)