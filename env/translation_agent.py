from llamagym import Agent
from env import atari
import gymnasium as gym
import argparse
from collections import Counter

class TranslationAgent(Agent):

    def __init__(self, model, tokenizer, device, generate_config_dict=None, ppo_config_dict=None, obs_translator=None, game_describer=None, 
                 reasoning_mode="BASE", num_votes=3, num_cot_samples=5):
        # We'll store references needed for COT, MVOTE, and BEST
        self.reasoning_mode = reasoning_mode
        self.num_votes = num_votes # MVOTE
        self.num_cot_samples = num_cot_samples # BEST
        self.obs_translator = obs_translator
        if game_describer is None and self.game_describer is None:
            assert False, "game_describer is None. Please provide a game describer."
        elif game_describer:
            self.game_describer = game_describer
        super().__init__(model, tokenizer, device, generate_config_dict, ppo_config_dict)
        
    def get_system_prompt(self) -> str:
        return f"You are an expert-level game player. {self.game_describer.describe_game()} {self.game_describer.describe_goal()} {self.game_describer.describe_action()}"

    def format_observation(self, observation: gym.core.ObsType) -> str:
        return f"{self.obs_translator.translate(observation)}"

    def extract_action(self, response: str) -> gym.core.ActType:
        # Implement here for each game
        pass

    def _inject_think_step_by_step(self):
        """
        For COT, we can simply append 'Think step by step.' to the system prompt
        or use any special token/hint you like.
        """
        self.current_episode_messages[0]["content"] += "\nThink step by step."

    def _generate_single_response(self) -> str:
        """
        Helper: calls self.llm() on current_episode_messages, returns the raw string.
        """
        return self.llm(self.current_episode_messages)

    def _majority_vote(self) -> str:
        """
        1) Sample multiple responses
        2) Extract action from each
        3) Pick majority
        4) Return the response whose action is the majority
        """
        responses = []
        actions = []
        for _ in range(self.num_votes):
            r = self._generate_single_response()
            a = self.extract_action(r)
            responses.append(r)
            actions.append(a)

        # pick the action that occurs the most
        majority_action = Counter(actions).most_common(1)[0][0]

        # find the first response that had that majority action
        for r, a in zip(responses, actions):
            if a == majority_action:
                return r
        return responses[0]

    def _best_of_n(self) -> str:
        """
        1) Generate multiple candidate responses
        2) Use the same model to judge the best response
        3) Return that best response
        """
        # Step 1: Collect N candidate responses
        candidate_responses = []
        for _ in range(self.num_cot_samples):
            r = self._generate_single_response()
            candidate_responses.append(r)

        # Step 2: Build a judge prompt that enumerates them
        # e.g. "Here are multiple answers. Which is best, 1..N? Return only the number."
        judge_prompt = (
            "You are judging multiple possible answers to a game observation.\n"
            f"Here is the game description: {self.get_system_prompt()}\n"
            "Answers:\n"
        )
        for i, resp in enumerate(candidate_responses, start=1):
            judge_prompt += f"{i}. {resp}\n"
        judge_prompt += (
            "\nPick the best answer. Return only the number corresponding to the best."
        )

        # We'll temporarily replace our conversation with a custom judge scenario
        backup_messages = list(self.current_episode_messages)
        self.current_episode_messages = [
            {"role": "system", "content": "System: You are the judge."},
            {"role": "user", "content": judge_prompt},
        ]

        # Step 3: Let model produce "1", "2", etc.
        judge_decision = self._generate_single_response().strip()
        # restore conversation
        self.current_episode_messages = backup_messages

        # parse out the integer
        try:
            chosen_idx = int("".join([c for c in judge_decision if c.isdigit()])) - 1
        except:
            chosen_idx = 0
        if not (0 <= chosen_idx < len(candidate_responses)):
            chosen_idx = 0

        # return that candidate
        return candidate_responses[chosen_idx]

    def act(self, observation):
        """
        Because we want a fresh chain-of-thought or fresh majority vote each turn,
        we rebuild self.current_episode_messages with just the system prompt + user message.
        """
        # Rebuild message for this step
        self.current_episode_messages = [
            {"role": "system", "content": self.get_system_prompt()},
        ]
        obs_text = self.format_observation(observation)
        self.current_episode_messages.append({"role": "user", "content": obs_text})

        # Now pick strategy
        if self.reasoning_mode == "BASE":
            # 1) no special logic; single generation
            response = self._generate_single_response()

        elif self.reasoning_mode == "COT":
            # 2) chain-of-thought style, instruct to "Think step by step"
            self._inject_think_step_by_step()
            response = self._generate_single_response()

        elif self.reasoning_mode == "MVOTE":
            # 3) majority vote among num_votes responses
            response = self._majority_vote()

        elif self.reasoning_mode == "BEST":
            # 4) best-of-N with a judge prompt
            response = self._best_of_n()

        else:
            raise ValueError(f"Unknown reasoning_mode: {self.reasoning_mode}")

        # parse final action from the chosen response
        action = self.extract_action(response)
        # store the final assistant message
        self.current_episode_messages.append({"role": "assistant", "content": response})

        return action
            

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
            self.game_describer = atari.SpaceInvaders_translator.GameDescriber(args)
        super().__init__(model, tokenizer, device, generate_config_dict, ppo_config_dict, obs_translator, game_describer)

    def extract_action(self, response: str) -> gym.core.ActType:
        digits = [char for char in response if char.isdigit()]
        if len(digits) == 0 or digits[-1] not in ("1", "2", "3", "4", "5", "0"):
            if "Move LEFT and FIRE" in response.lower():
                out = 5
            elif "Move RIGHT and FIRE" in response.lower():
                out = 4
            elif "Move LEFT" in response.lower():
                out = 3
            elif "Move RIGHT" in response.lower():
                out = 2
            elif "FIRE" in response.lower():
                out = 1
            elif "NOOP" in response.lower():
                out = 0
        elif digits[-1] in ("1", "2", "3", "4", "5", "0"):
            out = int(digits[-1])
        else:
            print(f"TranslationAgent.extract_action({response}): cannot extract action. Return 0: Do nothing (NOOP)")
            out = 0
        if out not in range(6):
            print(f"TranslationAgent.extract_action({response}): out of bounds. Return 0: Do nothing (NOOP)")
            out = 0
        return out

class PongAgent(TranslationAgent):
    def __init__(
        self, model, tokenizer, device, generate_config_dict=None, ppo_config_dict=None, obs_translator=None, game_describer=None
    ):
        if obs_translator is None:
            obs_translator = atari.Pong_translator.ObsTranslator()
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
            self.game_describer = atari.Pong_translator.GameDescriber(args)
        super().__init__(model, tokenizer, device, generate_config_dict, ppo_config_dict, obs_translator, game_describer)

    def extract_action(self, response: str) -> gym.core.ActType:
        digits = [char for char in response if char.isdigit()]
        out = -1
        if len(digits) == 0 or digits[-1] not in ("1", "2", "3", "4", "5", "0"):
            if "Move left while hiting the ball" in response.lower():
                out = 5
            elif "Move right while hiting the ball" in response.lower():
                out = 4
            elif "Move LEFT" in response.lower():
                out = 3
            elif "Move RIGHT" in response.lower():
                out = 2
            elif "Hit your ball" in response.lower():
                out = 1
            elif "NOOP" in response.lower():
                out = 0
        elif digits[-1] in ("1", "2", "3", "4", "5", "0"):
            out = int(digits[-1])
        else:
            print(f"PongAgent.extract_action({response}): cannot extract action. Return 0: Do nothing (NOOP)")
            out = 0
        if out not in range(6):
            print(f"The extracted actio is {out}, which is out of bounds. Return 0: Do nothing (NOOP)")
            out = 0
        return out
