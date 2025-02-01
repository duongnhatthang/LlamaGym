import os
import re
import wandb
from tqdm import trange

from transformers import AutoTokenizer, AutoModelForCausalLM
import gymnasium as gym
from llamagym.inf_agent import InferenceAgent

# examples/bj_noft.py
class BlackjackAgent(InferenceAgent):
    def get_system_prompt(self) -> str:
        return """You are an expert blackjack player. Every turn, you'll see your current sum, the dealer's showing card value, and whether you have a usable ace. Win by exceeding the dealer's hand but not exceeding 21.
Decide whether to stay with your current sum by writing "Action: 0" or accept another card by writing "Action: 1". Accept a card unless very close to 21."""

    def format_observation(self, observation: gym.core.ObsType) -> str:
        return f"You: {observation[0]}. Dealer: {observation[1]}. You have {'an' if bool(observation[2]) else 'no'} ace."

    def extract_action(self, response: str) -> gym.core.ActType:
        match = re.compile(r"Action: (\d)").search(response)
        if match:
            return int(match.group(1))

        if "stick" in response.lower():
            return 0
        elif "hit" in response.lower():
            return 1
        
        digits = [char for char in response if char.isdigit()]
        if len(digits) == 0 or digits[-1] not in ("0", "1"):
            if "stick" in response.lower():
                return 0
            elif "hit" in response.lower():
                return 1

        return 0  # Default to 'stay' if unclear

if __name__ == "__main__":
    hyperparams = {
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "env": "Blackjack-v1",
        "batch_size": 1,
        "episodes": 5000,
        "generate/max_new_tokens": 32,
    }

    wandb_run = wandb.init(project=os.environ.get("WANDB_PROJECT"), config=hyperparams)
    device = "cuda:1"
    HF_TOKEN = os.environ.get("HF_TOKEN")

    model = AutoModelForCausalLM.from_pretrained(
        hyperparams["model_name"],
        token=HF_TOKEN
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(hyperparams["model_name"], token=HF_TOKEN)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))

    # [TODO] MING: Fix the error: Remove unnecessary empty dicts (only pass generate config)
    agent = BlackjackAgent(model, tokenizer, device, {
        "max_new_tokens": 32,
        "do_sample": True,
        "top_p": 0.6,
        "top_k": 0,
        "temperature": 0.9,
    })

    env = gym.make(hyperparams["env"], natural=False, sab=False)

    for episode in trange(hyperparams["episodes"]):
        observation, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        wandb.log({
            "episode": episode,
            "total_return": total_reward
        })

    wandb_run.finish()