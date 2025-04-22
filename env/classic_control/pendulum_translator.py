import math

class ObsTranslator:
    def __init__(self,):
        pass

    def translate(self, state):
        x, y, angular_velocity = state
        angle = math.atan2(y, x)  # Calculate the angle from the x-y coordinates
        direction = "counterclockwise" if angular_velocity > 0 else "clockwise"  # Corrected direction logic
        res = (f"The pendulum is at an angle of {angle:.3f} radians from the vertical (zero when upright), "
               f"rotating at {abs(angular_velocity):.2f} radians per second in the {direction} direction.")
        return res

class GameDescriber:
    def __init__(self, args):
        self.is_only_local_obs = args.is_only_local_obs == 1
        self.max_episode_len = args.max_episode_len
        self.action_desc_dict = {
        }
        self.reward_desc_dict = {
        }

    def describe_goal(self):
        return "The goal is to swing the pendulum upright and keep it balanced."

    def translate_terminate_state(self, state, episode_len, max_episode_len): 
        return "The game ends when the maximum episode length is reached."

    def translate_potential_next_state(self, state, action):
        return f"Applying a torque of {action:.2f} may change the pendulum's angle and angular velocity."

    def describe_game(self):
        return "In the Pendulum game, you control a pendulum attached to a fixed pivot point. The goal is to apply " \
               "torques to swing the pendulum upright and keep it balanced. The game ends if the pendulum cannot be " \
               "stabilized within the given time limit. The closer the pendulum is to the upright position, the higher your score."

    def describe_action(self):
        return "Provide a torque value (e.g., a float between -2.0 and 2.0) to control the pendulum's movement. Return the torque value enclosed in < and >, e.g., <1.5>."

class TransitionTranslator(ObsTranslator):
    def translate(self, infos, is_current=False):
        descriptions = []
        if is_current: 
            state_desc = ObsTranslator().translate(infos[-1]['state'])
            return state_desc
        for i, info in enumerate(infos):
            assert 'state' in info, "info should contain state information"
        
            state_desc = ObsTranslator().translate(info['state'])
            action_desc = f"Take Action: Apply a torque of {info['action']:.2f}."
            reward_desc = f"Result: Reward of {info['reward']:.2f}, "
            next_state_desc = ObsTranslator().translate(info['next_state'])
            descriptions.append(f"{state_desc}.\n {action_desc} \n {reward_desc} \n Transit to {next_state_desc}")
        return descriptions
