# [Translator classes and functions for Atari Boxing environment]
#'labels': {'player_y': 109, 'player_x': 188, 'enemy_y': 20, 'enemy_x': 64, 'ball_x': 0, 'ball_y': 0, 'enemy_score': 0, 'player_score': 0}
class ObsTranslator:
    def __init__(
        self,
    ):
        pass

    def translate(self, state):
        (
            player_y,
            player_x,
            enemy_y,
            enemy_x,
            ball_x,
            ball_y,
            enemy_score,
            player_score,
            v_ball_x,
            v_ball_y,
        ) = state
        return (
            f"The origin (0,0) is in the top left corner. You are at position ({player_x}, {player_y}), your opponent is at position ({enemy_x}, {enemy_y}), "
            f"the ball is at ({ball_x}, {ball_y}), the ball velocity is ({v_ball_x}, {v_ball_y}). "
            f"Your opponent's score is {enemy_score}, your score is {player_score}."
        )


class GameDescriber:
    def __init__(self, args):
        self.is_only_local_obs = args.is_only_local_obs == 1
        self.max_episode_len = args.max_episode_len
        self.frameskip = args.frameskip
        self.action_desc_dict = {}
        self.reward_desc_dict = {}

    def describe_goal(self):
        return "The goal is to knock out your opponent."

    def translate_terminate_state(self, state, episode_len, max_episode_len):
        return ""

    def translate_potential_next_state(self, state, action):
        return ""

    def describe_game(self):
        return (
            "In the Pong game, you play the ball with your opponent, each player rallys the ball by moving the paddles on the playfield. "
            "Paddles move only vertically on the playfield. A player scores one point when the opponent hits the ball out of bounds or misses a hit. "
            "The first player to score 21 points wins the game. The number of frameskip is set to 4. "
        )

    def describe_action(self):
        return (
            "Type 1 for NOOP (no operation), 3 to move up, 4 to move down. "
            "Ensure you only provide the action number from the valid action list, i.e., [1, 3, 4]. "
        )


class TransitionTranslator(ObsTranslator):
    def translate(self, infos, is_current=False):
        descriptions = []
        if is_current:
            state_desc = ObsTranslator().translate(infos[-1]["state"])
            return state_desc
        for i, info in enumerate(infos):
            assert "state" in info, "info should contain state information"

            state_desc = ObsTranslator().translate(info["state"])
            if info["action"] == 0:
                action_desc = f"Take Action: 'Do nothing'"
            elif info["action"] == 1:
                action_desc = f"Take Action: 'Hit your ball'"
            elif info["action"] == 2:
                action_desc = f"Take Action: 'Move right'"
            elif info["action"] == 3:
                action_desc = f"Take Action: 'Move left'"
            elif info["action"] == 4:
                action_desc = f"Take Action: 'Move right while hiting the ball'"
            else:
                action_desc = f"Take Action: 'Move left while hiting the ball'"
            reward_desc = f"Result: Reward of {info['reward']}, "
            next_state_desc = ObsTranslator().translate(info["next_state"])
            descriptions.append(
                f"{state_desc}.\n {action_desc} \n {reward_desc} \n Transit to {next_state_desc}"
            )
        return descriptions
