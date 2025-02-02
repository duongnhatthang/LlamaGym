from collections import Counter
from llamagym import Agent
import gymnasium as gym
import torch
from trl import (
    PPOTrainer,
    PPOConfig,
    create_reference_model,
)


class MajorityVoteAgent(Agent):
    def __init__(
        self,
        model,
        tokenizer,
        device,
        generate_config_dict=None,
        ppo_config_dict=None,
        num_votes=10
    ):
        super().__init__(model, tokenizer, device, generate_config_dict, ppo_config_dict)
        self.num_votes = num_votes

    def get_system_prompt(self) -> str:
        return "You are an intelligent agent designed to solve tasks optimally."

    def format_observation(self, observation: gym.core.ObsType) -> str:
        return f"Observation: {observation}"

    def extract_action(self, response: str) -> gym.core.ActType:
        if "Action:" in response:
            return response.split("Action:")[-1].strip()
        raise ValueError("No valid action found in response.")

    def act(self, observation):
        message = self.format_observation(observation)
        self.current_episode_messages.append({"role": "user", "content": message})

        responses = []
        actions = []
        for _ in range(self.num_votes):
            response = self.llm(self.current_episode_messages)
            action = self.extract_action(response)
            print(actions)
            responses.append(response)
            actions.append(action)

        # Majority voting
        action_counts = Counter(actions)
        majority_action, _ = action_counts.most_common(1)[0]
        majority_response = next(response for response, action in zip(responses, actions) if action == majority_action)
        self.current_episode_messages.append({"role": "assistant", "content": majority_response})

        return majority_action