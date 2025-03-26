from llamagym import Agent
from env import atari
import gymnasium as gym
import argparse

class TranslationAgent(Agent):
    def __init__(
        self, model, tokenizer, device, generate_config_dict=None, ppo_config_dict=None, obs_translator=None, game_describer=None
    ):
        self.obs_translator = obs_translator
        self.game_describer = game_describer
        super().__init__(model, tokenizer, device, generate_config_dict, ppo_config_dict)
        
    def get_system_prompt(self) -> str:
        return f"You are an expert-level game player. {self.game_describer.describe_game()}. {self.game_describer.describe_goal()}. {self.game_describer.describe_action()}"

    def format_observation(self, observation: gym.core.ObsType) -> str:
        return f"{self.obs_translator.translate(observation)}"

    def extract_action(self, response: str) -> gym.core.ActType:
        # Implement here for each game
        pass

class SpaceInvadersAgent(TranslationAgent):
    def __init__(
        self, model, tokenizer, device, generate_config_dict=None, ppo_config_dict=None, obs_translator=None, game_describer=None
    ):
        if obs_translator is None:
            obs_translator = atari.SpaceInvaders_translator.ObsTranslator()
        if game_describer is None:
            parser = argparse.ArgumentParser(
                description="Place holder args to init stuff."
            )
            parser.add_argument(
                "--is_only_local_obs",
                type=int,
                default=1,
                help="Whether only taking local observations, if is_only_local_obs = 1, only using local obs"
            )
            parser.add_argument(
                "--max_episode_len",
                type=int,
                default=108000//6,
                help="The maximum number of steps in an episode",
            )
            parser.add_argument(
                "--frameskip",
                type=int,
                default=4,
                help="The frameskip for atari environments",
            )
            args = parser.parse_args()
            game_describer = atari.SpaceInvaders_translator.GameDescriber(args)
        super().__init__(model, tokenizer, device, obs_translator, game_describer, generate_config_dict, ppo_config_dict)
    def extract_action(self, response: str) -> gym.core.ActType:
        match = re.compile(r"Take Action: (\d)").search(response)
        if match:
            return int(match.group(1))

        digits = [char for char in response if char.isdigit()]
        if len(digits) == 0 or digits[-1] not in ("0", "1", "2", "3", "4", "5", "6"):
            if "Move LEFT and FIRE" in response.lower():
                return 6
            elif "Move RIGHT and FIRE" in response.lower():
                return 5
            elif "Move LEFT" in response.lower():
                return 4
            elif "Move RIGHT" in response.lower():
                return 3
            elif "FIRE" in response.lower():
                return 2
            elif "NOOP" in response.lower():
                return 1
        else:
            print(f"TranslationAgent.extract_action({response}): cannot extract action. Return 1: Do nothing (NOOP)")
        return 1