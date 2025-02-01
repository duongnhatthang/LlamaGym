from .agent import Agent

class InferenceAgent(Agent):
    def __init__(self, model, tokenizer, device, generate_config_dict=None):
        # Directly initialize only the necessary attributes (bypass PPO-related init)
        if generate_config_dict is None:
            generate_config_dict = {
                "max_new_tokens": 32,
                "do_sample": True,
                "top_p": 0.6,
                "top_k": 0,
                "temperature": 0.9,
            }

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.generate_config_dict = generate_config_dict

        self.current_episode_messages = [
            {"role": "system", "content": self.get_system_prompt()}
        ]
        self.current_episode_rewards = []

    def assign_reward(self, reward):
        pass  # No-op since we're not fine-tuning

    def terminate_episode(self, train=False):
        self.current_episode_messages = [
            {"role": "system", "content": self.get_system_prompt()}
        ]
        self.current_episode_rewards = []
        return {}

    def train_batch(self, batch_queries, batch_responses, batch_rewards):
        raise NotImplementedError("Training is disabled in InferenceAgent.")