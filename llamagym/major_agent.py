from collections import Counter
from llamagym import Agent
import gymnasium as gym
import re
import torch


class MajorityVoteAgent(Agent):
    def __init__(self, model, tokenizer, device, generate_config_dict=None, num_votes=1):
        super().__init__()  # Init
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.generate_config_dict = generate_config_dict or {
            "max_new_tokens": 32,
            "do_sample": True,
            "top_p": 0.6,
            "top_k": 0,
            "temperature": 0.9,
        }
        self.num_votes = num_votes
        self.current_episode_messages = [{"role": "system", "content": self.get_system_prompt()}]

    def act(self, observation):
        message = self.format_observation(observation)
        self.current_episode_messages.append({"role": "user", "content": message})

        responses, actions = [], []
        for _ in range(self.num_votes):
            response = self.llm(self.current_episode_messages)
            action = self.extract_action(response)
            responses.append(response)
            actions.append(action)

        # Majority Vote
        action_counts = Counter(actions)
        majority_action, _ = action_counts.most_common(1)[0]
        majority_response = next(
            response for response, action in zip(responses, actions) if action == majority_action
        )

        self.current_episode_messages.append({"role": "assistant", "content": majority_response})
        return majority_action