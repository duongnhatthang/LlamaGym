import argparse
import os
import openai
from dotenv import load_dotenv
import gymnasium as gym
from collections import Counter

from llamagym import Agent
from env import atari


class TranslationAgent(Agent):
    def __init__(
        self,
        openai_api_key=None,
        use_env_api_key=False,       # <--- boolean flag to load from .env
        reasoning_mode="BASE",
        num_votes=3,
        num_cot_samples=5,
        obs_translator=None,
        game_describer=None,
    ):
        """
        If `use_env_api_key` is True, we load from .env, ignoring the passed `openai_api_key`.
        Otherwise, we use the `openai_api_key` passed in.
        """
        if use_env_api_key:
            load_dotenv()  # loads environment variables from .env
            openai_api_key = os.getenv("OPENAI_API_KEY", None)

        if not openai_api_key:
            raise ValueError(
                "No valid OpenAI API key found. Please provide one directly or via .env."
            )

        # Store references needed for CoT, MVOTE, BEST, etc.
        openai.api_key = openai_api_key
        self.reasoning_mode = reasoning_mode
        self.num_votes = num_votes
        self.num_cot_samples = num_cot_samples
        self.obs_translator = obs_translator
        self.game_describer = game_describer

        super().__init__(model=None, tokenizer=None, device=None)

    def get_system_prompt(self) -> str:
        """
        Build the system-level instructions describing the environment or game context.
        """
        if not self.game_describer:
            return "You are an expert-level game player. No game description found."
        return (
            "You are an expert-level game player. "
            f"{self.game_describer.describe_game()} "
            f"{self.game_describer.describe_goal()} "
            f"{self.game_describer.describe_action()}"
        )

    def format_observation(self, observation: gym.core.ObsType) -> str:
        """
        Convert the environment's raw observation into a text string for GPT-4o.
        """
        if self.obs_translator:
            return f"{self.obs_translator.translate(observation)}"
        return "No observation translator provided."

    def extract_action(self, response: str) -> gym.core.ActType:
        """
        Subclasses must override this to parse a valid game action from `response`.
        """
        raise NotImplementedError

    def _inject_think_step_by_step(self):
        """
        For chain-of-thought, just append 'Think step by step.' to the last user prompt.
        """
        self.current_episode_messages[-1]["content"] += "\nThink step by step."

    def _generate_single_response(self) -> str:
        """
        Calls GPT-4o (mini) to get a single response from the current conversation.
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # PLEASE USE 4oMINI ONLY for now. 
                # LIMITATION: Less than 30-40 dollars per day
                messages=self.current_episode_messages
            )
            content = response.choices[0].message["content"].strip()
        except Exception as e:
            print(f"OpenAI API call failed: {e}")
            content = "NOOP"
        return content

    def _majority_vote(self) -> str:
        """
        1) Sample multiple responses
        2) Extract action from each
        3) Pick majority action
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

        # return the first response that had that majority action
        for r, a in zip(responses, actions):
            if a == majority_action:
                return r
        return responses[0]

    def _best_of_n(self) -> str:
        """
        1) Generate multiple candidate responses
        2) Build a judge prompt to pick the best
        3) Return that best response
        """
        # Step 1: Collect N candidate responses
        candidate_responses = []
        for _ in range(self.num_cot_samples):
            r = self._generate_single_response()
            candidate_responses.append(r)

        # Step 2: Build a judge prompt that enumerates them
        judge_prompt = (
            "You are judging multiple possible answers to a game observation.\n"
            f"Here is the game description: {self.get_system_prompt()}\n"
            "Answers:\n"
        )
        for i, resp in enumerate(candidate_responses, start=1):
            judge_prompt += f"{i}. {resp}\n"
        judge_prompt += "\nPick the best answer. Return only the number corresponding to the best."

        # Temporarily replace conversation with the judge scenario
        backup_messages = list(self.current_episode_messages)
        self.current_episode_messages = [
            {"role": "system", "content": "System: You are the judge."},
            {"role": "user", "content": judge_prompt},
        ]

        # Step 3: Let GPT-4o produce "1", "2", etc.
        judge_decision = self._generate_single_response().strip()

        # Restore conversation
        self.current_episode_messages = backup_messages

        # parse out the integer
        try:
            chosen_idx = int("".join([c for c in judge_decision if c.isdigit()])) - 1
        except:
            chosen_idx = 0
        if not (0 <= chosen_idx < len(candidate_responses)):
            chosen_idx = 0

        return candidate_responses[chosen_idx]

    def act(self, observation):
        """
        For each step, rebuild conversation with only the system prompt + user message,
        then pick a reasoning strategy and parse the final action from the chosen response.
        """
        self.current_episode_messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": self.format_observation(observation)},
        ]

        if self.reasoning_mode == "BASE":
            response = self._generate_single_response()
        elif self.reasoning_mode == "COT":
            self._inject_think_step_by_step()
            response = self._generate_single_response()
        elif self.reasoning_mode == "MVOTE":
            response = self._majority_vote()
        elif self.reasoning_mode == "BEST":
            response = self._best_of_n()
        else:
            raise ValueError(f"Unknown reasoning_mode: {self.reasoning_mode}")

        action = self.extract_action(response)

        # Optionally store the final assistant response
        self.current_episode_messages.append({"role": "assistant", "content": response})

        return action


class SpaceInvadersAgent(TranslationAgent):
    def __init__(
        self,
        openai_api_key=None,
        use_env_api_key=False,
        reasoning_mode="COT",
        num_votes=3,
        num_cot_samples=5,
        obs_translator=None,
        game_describer=None,
    ):
        if obs_translator is None:
            obs_translator = atari.SpaceInvaders_translator.ObsTranslator()
        if game_describer is None:
            parser = argparse.ArgumentParser(description="Placeholder args to init stuff.")
            parser.add_argument("--is_only_local_obs", type=int, default=1)
            parser.add_argument("--max_episode_len", type=int, default=108000 // 6)
            parser.add_argument("--frameskip", type=int, default=4)
            args = parser.parse_args()
            game_describer = atari.SpaceInvaders_translator.GameDescriber(args)

        # Pass everything up to the parent
        super().__init__(
            openai_api_key=openai_api_key,
            use_env_api_key=use_env_api_key,
            reasoning_mode=reasoning_mode,
            num_votes=num_votes,
            num_cot_samples=num_cot_samples,
            obs_translator=obs_translator,
            game_describer=game_describer
        )

    def extract_action(self, response: str) -> gym.core.ActType:
        digits = [char for char in response if char.isdigit()]
        out = 0
        if not digits:
            # fallback to text-based clues
            if "move left and fire" in response.lower():
                out = 5
            elif "move right and fire" in response.lower():
                out = 4
            elif "move left" in response.lower():
                out = 3
            elif "move right" in response.lower():
                out = 2
            elif "fire" in response.lower():
                out = 1
            else:
                out = 0
        else:
            if digits[-1] in ("0", "1", "2", "3", "4", "5"):
                out = int(digits[-1])
            else:
                out = 0

        if out not in range(6):
            print(f"[SpaceInvadersAgent] Invalid action '{out}'. Using 0 (NOOP).")
            out = 0
        return out


class PongAgent(TranslationAgent):
    def __init__(
        self,
        openai_api_key=None,
        use_env_api_key=False,
        reasoning_mode="COT",
        num_votes=3,
        num_cot_samples=5,
        obs_translator=None,
        game_describer=None,
    ):
        if obs_translator is None:
            obs_translator = atari.Pong_translator.ObsTranslator()
        if game_describer is None:
            parser = argparse.ArgumentParser(description="Placeholder args to init stuff.")
            parser.add_argument("--is_only_local_obs", type=int, default=1)
            parser.add_argument("--max_episode_len", type=int, default=108000 // 6)
            parser.add_argument("--frameskip", type=int, default=4)
            args = parser.parse_args()
            game_describer = atari.Pong_translator.GameDescriber(args)

        super().__init__(
            openai_api_key=openai_api_key,
            use_env_api_key=use_env_api_key,
            reasoning_mode=reasoning_mode,
            num_votes=num_votes,
            num_cot_samples=num_cot_samples,
            obs_translator=obs_translator,
            game_describer=game_describer
        )

    def extract_action(self, response: str) -> gym.core.ActType:
        digits = [char for char in response if char.isdigit()]
        out = 0
        if not digits:
            # fallback to text-based clues
            if "move left while hiting the ball" in response.lower():
                out = 5
            elif "move right while hiting the ball" in response.lower():
                out = 4
            elif "move left" in response.lower():
                out = 3
            elif "move right" in response.lower():
                out = 2
            elif "hit your ball" in response.lower():
                out = 1
            else:
                out = 0
        else:
            if digits[-1] in ("0", "1", "2", "3", "4", "5"):
                out = int(digits[-1])
            else:
                out = 0

        if out not in range(6):
            print(f"[PongAgent] Invalid action '{out}'. Using 0 (NOOP).")
            out = 0
        return out