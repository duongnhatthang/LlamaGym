import gymnasium as gym
import ale_py
import numpy as np
from atariari.benchmark.wrapper import AtariARIWrapper
from typing import Optional, Union

class GymCompatWrapper:
    """A wrapper to make the AtariARIWrapper compatible with the gym API. Specifically, it ensures that the step method returns a tuple of (obs, reward, done, info) instead of (obs, reward, terminated, truncated, info)."""
    def __init__(self, env):
        self.env = env
        self.spec = getattr(self.env, 'spec', None)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    # Forward other methods as needed
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
    
    # Forward other common environment attributes/methods as needed
    def close(self):
        return self.env.close()

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    # Handle any attribute access not explicitly defined
    def __getattr__(self, name):
        return getattr(self.env, name)

class GymCompatWrapper2(GymCompatWrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, done, info

class MaxAndSkip(gym.Wrapper):
    """Return only every `skip`-th frame"""

    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        info = {}
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class RepresentedAtariEnv(gym.Wrapper):
    def __init__(self, env_name, render_mode=None, frameskip=4, repeat_action_probability=0.0):
        if frameskip > 0:
            self.env = MaxAndSkip(AtariARIWrapper(GymCompatWrapper(gym.make(env_name, render_mode=render_mode, repeat_action_probability=repeat_action_probability))), skip=frameskip)
        else:
            super().__init__(AtariARIWrapper(GymCompatWrapper(gym.make(env_name, render_mode=render_mode, repeat_action_probability=repeat_action_probability))))
        self.metadata = self.env.metadata
        self.env_name = env_name
        self.observation = None
        self.info = {}
        self.action_space = self.env.action_space
        _ = self.env.reset()
        obs = self.env.labels()
        obs_dim = len(obs)
        self.obs_label = obs.keys()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def step(self, action):
        out = self.env.step(action)
        original_next_obs, reward, env_done, info = self.env.step(action)
        next_obs = self.env.labels()
        self.obs_label = next_obs.keys()
        self.observation = next_obs
        return np.array(list(next_obs.values())), reward, env_done, info

    def reset(self, seed=0):
        obs_original, info = self.env.reset(seed=seed)
        obs = self.env.labels()
        self.obs_label = obs.keys()
        self.observation = obs
        return np.array(list(obs.values())), info

    def get_info(self):
        return self.observation

    def render(self, render_mode=None):
        return self.env.render()

# class RepresentedMsPacman(RepresentedAtariEnv):
#     def __init__(self, render_mode: Optional[str]=None, frameskip: int=4):
#         env_name = "MsPacmanNoFrameskip-v4"
#         super().__init__(env_name=env_name, render_mode=render_mode, frameskip=frameskip, repeat_action_probability=0)

# class RepresentedBowling(RepresentedAtariEnv):
#     def __init__(self, render_mode: Optional[str]=None, frameskip: int=4):
#         env_name = "BowlingNoFrameskip-v4"
#         super().__init__(env_name=env_name, render_mode=render_mode, frameskip=frameskip, repeat_action_probability=0)

# class RepresentedBoxing(RepresentedAtariEnv):
#     def __init__(self, render_mode: Optional[str]=None, frameskip: int=4):
#         env_name = "BoxingNoFrameskip-v4"
#         super().__init__(env_name=env_name, render_mode=render_mode, frameskip=frameskip, repeat_action_probability=0)

# class RepresentedBreakout(RepresentedAtariEnv):
#     def __init__(self, render_mode: Optional[str]=None, frameskip: int=4):
#         env_name = "BreakoutNoFrameskip-v4"
#         super().__init__(env_name=env_name, render_mode=render_mode, frameskip=frameskip, repeat_action_probability=0)

# class RepresentedDemonAttack(RepresentedAtariEnv):
#     def __init__(self, render_mode: Optional[str]=None, frameskip: int=4):
#         env_name = "DemonAttackNoFrameskip-v4"
#         super().__init__(env_name=env_name, render_mode=render_mode, frameskip=frameskip, repeat_action_probability=0)

# class RepresentedFreeway(RepresentedAtariEnv):
#     def __init__(self, render_mode: Optional[str]=None, frameskip: int=4):
#         env_name = "FreewayNoFrameskip-v4"
#         super().__init__(env_name=env_name, render_mode=render_mode, frameskip=frameskip, repeat_action_probability=0)

# class RepresentedFrostbite(RepresentedAtariEnv):
#     def __init__(self, render_mode: Optional[str]=None, frameskip: int=4):
#         env_name = "FrostbiteNoFrameskip-v4"
#         super().__init__(env_name=env_name, render_mode=render_mode, frameskip=frameskip, repeat_action_probability=0)

# class RepresentedHero(RepresentedAtariEnv):
#     def __init__(self, render_mode: Optional[str]=None, frameskip: int=4):
#         env_name = "HeroNoFrameskip-v4"
#         super().__init__(env_name=env_name, render_mode=render_mode, frameskip=frameskip, repeat_action_probability=0)

# class RepresentedMontezumaRevenge(RepresentedAtariEnv):
#     def __init__(self, render_mode: Optional[str]=None, frameskip: int=4):
#         env_name = "MontezumaRevengeNoFrameskip-v4"
#         super().__init__(env_name=env_name, render_mode=render_mode, frameskip=frameskip, repeat_action_probability=0)

# class RepresentedPitfall(RepresentedAtariEnv):
#     def __init__(self, render_mode: Optional[str]=None, frameskip: int=4):
#         env_name = "PitfallNoFrameskip-v4"
#         super().__init__(env_name=env_name, render_mode=render_mode, frameskip=frameskip, repeat_action_probability=0)

class RepresentedPong(RepresentedAtariEnv):
    def __init__(self, render_mode: Optional[str]=None, frameskip: int=4):
        env_name = "PongNoFrameskip-v4"
        super().__init__(env_name=env_name, render_mode=render_mode, frameskip=frameskip, repeat_action_probability=0)
        self.prev_ball_x, self.prev_ball_y = None, None
        obs = self.env.labels()
        obs = self._customize_observation(obs)
        obs_dim = len(obs)
        self.obs_label = obs.keys()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _customize_observation(self, obs):
        # Add additional information of ball velocity or transform the observation
        ball_x, ball_y = obs['ball_x'], obs['ball_y']
        if self.prev_ball_x is None or self.prev_ball_x is None:
            v_ball_x, v_ball_y = 0, 0
        else:
            # Calculate ball velocity
            v_ball_x = ball_x - self.prev_ball_x
            v_ball_y = ball_y - self.prev_ball_y
        self.prev_ball_x, self.prev_ball_y = ball_x, ball_y
        obs['v_ball_x'] = v_ball_x
        obs['v_ball_y'] = v_ball_y
        return obs

    def step(self, action):
        out = self.env.step(action)
        original_next_obs, reward, env_done, info = self.env.step(action)
        next_obs = self.env.labels()
        next_obs = self._customize_observation(next_obs)
        self.obs_label = next_obs.keys()
        self.observation = next_obs
        return np.array(list(next_obs.values())), reward, env_done, info

    def reset(self, seed=0):
        obs_original, info = self.env.reset(seed=seed)
        obs = self.env.labels()
        self.prev_ball_x, self.prev_ball_y = None, None
        obs = self._customize_observation(obs)
        self.obs_label = obs.keys()
        self.observation = obs
        return np.array(list(obs.values())), info

# class RepresentedPrivateEye(RepresentedAtariEnv):
#     def __init__(self, render_mode: Optional[str]=None, frameskip: int=4):
#         env_name = "PrivateEyeNoFrameskip-v4"
#         super().__init__(env_name=env_name, render_mode=render_mode, frameskip=frameskip, repeat_action_probability=0)

# class RepresentedQbert(RepresentedAtariEnv):
#     def __init__(self, render_mode: Optional[str]=None, frameskip: int=4):
#         env_name = "QbertNoFrameskip-v4"
#         super().__init__(env_name=env_name, render_mode=render_mode, frameskip=frameskip, repeat_action_probability=0)

# class RepresentedRiverraid(RepresentedAtariEnv):
#     def __init__(self, render_mode: Optional[str]=None, frameskip: int=4):
#         env_name = "RiverraidNoFrameskip-v4"
#         super().__init__(env_name=env_name, render_mode=render_mode, frameskip=frameskip, repeat_action_probability=0)

# class RepresentedSeaquest(RepresentedAtariEnv):
#     def __init__(self, render_mode: Optional[str]=None, frameskip: int=4):
#         env_name = "SeaquestNoFrameskip-v4"
#         super().__init__(env_name=env_name, render_mode=render_mode, frameskip=frameskip, repeat_action_probability=0)

class RepresentedSpaceInvaders(RepresentedAtariEnv):
    def __init__(self, render_mode: Optional[str]=None, frameskip: int=4):
        env_name = "SpaceInvadersNoFrameskip-v4"
        super().__init__(env_name=env_name, render_mode=render_mode, frameskip=frameskip, repeat_action_probability=0)

# class RepresentedTennis(RepresentedAtariEnv):
#     def __init__(self, render_mode: Optional[str]=None, frameskip: int=4):
#         env_name = "TennisNoFrameskip-v4"
#         super().__init__(env_name=env_name, render_mode=render_mode, frameskip=frameskip, repeat_action_probability=0)

# class RepresentedVenture(RepresentedAtariEnv):
#     def __init__(self, render_mode: Optional[str]=None, frameskip: int=4):
#         env_name = "VentureNoFrameskip-v4"
#         super().__init__(env_name=env_name, render_mode=render_mode, frameskip=frameskip, repeat_action_probability=0)

# class RepresentedVideoPinball(RepresentedAtariEnv):
#     def __init__(self, render_mode: Optional[str]=None, frameskip: int=4):
#         env_name = "VideoPinballNoFrameskip-v4"
#         super().__init__(env_name=env_name, render_mode=render_mode, frameskip=frameskip, repeat_action_probability=0)

# class RepresentedAsteroids(RepresentedAtariEnv):
#     def __init__(self, render_mode: Optional[str]=None, frameskip: int=4):
#         env_name = "AsteroidsNoFrameskip-v4"
#         super().__init__(env_name=env_name, render_mode=render_mode, frameskip=frameskip, repeat_action_probability=0)

# class RepresentedBattleZone(RepresentedAtariEnv):
#     def __init__(self, render_mode: Optional[str]=None, frameskip: int=4):
#         env_name = "BattleZoneNoFrameskip-v4"
#         super().__init__(env_name=env_name, render_mode=render_mode, frameskip=frameskip, repeat_action_probability=0)

# class RepresentedBerzerk(RepresentedAtariEnv):
#     def __init__(self, render_mode: Optional[str]=None, frameskip: int=4):
#         env_name = "BerzerkNoFrameskip-v4"
#         super().__init__(env_name=env_name, render_mode=render_mode, frameskip=frameskip, repeat_action_probability=0)

# class RepresentedHero(RepresentedAtariEnv):
#     def __init__(self, render_mode: Optional[str]=None, frameskip: int=4):
#         env_name = "HeroNoFrameskip-v4"
#         super().__init__(env_name=env_name, render_mode=render_mode, frameskip=frameskip, repeat_action_probability=0)

# class RepresentedSkiing(RepresentedAtariEnv):
#     def __init__(self, render_mode: Optional[str]=None, frameskip: int=4):
#         env_name = "SkiingNoFrameskip-v4"
#         super().__init__(env_name=env_name, render_mode=render_mode, frameskip=frameskip, repeat_action_probability=0)


def env_factory(env_class):
    def _create_instance(render_mode=None, frameskip=4):
        return env_class(render_mode=render_mode, frameskip=frameskip)
    return _create_instance


def register_environments():
    env_classes = {
        # 'RepresentedAsteroids-v0': RepresentedAsteroids,
        # 'RepresentedBattleZone-v0': RepresentedBattleZone,
        # 'RepresentedBerzerk-v0': RepresentedBerzerk,
        # 'RepresentedSkiing-v0': RepresentedSkiing,
        # 'RepresentedMsPacman-v0': RepresentedMsPacman,
        # 'RepresentedBowling-v0': RepresentedBowling,
        # 'RepresentedBoxing-v0': RepresentedBoxing,
        # 'RepresentedBreakout-v0': RepresentedBreakout,
        # 'RepresentedDemonAttack-v0': RepresentedDemonAttack,
        # 'RepresentedFreeway-v0': RepresentedFreeway,
        # 'RepresentedFrostbite-v0': RepresentedFrostbite,
        # 'RepresentedHero-v0': RepresentedHero,
        # 'RepresentedMontezumaRevenge-v0': RepresentedMontezumaRevenge,
        # 'RepresentedPitfall-v0': RepresentedPitfall,
        'RepresentedPong-v0': RepresentedPong,
        # 'RepresentedPrivateEye-v0': RepresentedPrivateEye,
        # 'RepresentedQbert-v0': RepresentedQbert,
        # 'RepresentedRiverraid-v0': RepresentedRiverraid,
        # 'RepresentedSeaquest-v0': RepresentedSeaquest,
        'RepresentedSpaceInvaders-v0': RepresentedSpaceInvaders,
        # 'RepresentedTennis-v0': RepresentedTennis,
        # 'RepresentedVenture-v0': RepresentedVenture,
        # 'RepresentedVideoPinball-v0': RepresentedVideoPinball
    }

    for env_name, env_class in env_classes.items():
        gym.register(
            id=env_name,
            entry_point=env_factory(env_class),
        )


# register_environments()
# env_classes = {
#     'RepresentedMsPacman-v0': RepresentedMsPacman,
#     'RepresentedBowling-v0': RepresentedBowling,
#     'RepresentedBoxing-v0': RepresentedBoxing,
#     'RepresentedBreakout-v0': RepresentedBreakout,
#     'RepresentedDemonAttack-v0': RepresentedDemonAttack,
#     'RepresentedFreeway-v0': RepresentedFreeway,
#     'RepresentedFrostbite-v0': RepresentedFrostbite,
#     'RepresentedHero-v0': RepresentedHero,
#     'RepresentedMontezumaRevenge-v0': RepresentedMontezumaRevenge,
#     'RepresentedPitfall-v0': RepresentedPitfall,
#     'RepresentedPong-v0': RepresentedPong,
#     'RepresentedPrivateEye-v0': RepresentedPrivateEye,
#     'RepresentedQbert-v0': RepresentedQbert,
#     'RepresentedRiverraid-v0': RepresentedRiverraid,
#     'RepresentedSeaquest-v0': RepresentedSeaquest,
#     'RepresentedSpaceInvaders-v0': RepresentedSpaceInvaders,
#     'RepresentedTennis-v0': RepresentedTennis,
#     'RepresentedVenture-v0': RepresentedVenture,
#     'RepresentedVideoPinball-v0': RepresentedVideoPinball
# }
#
# for env, env_class in env_classes.items():
#     env_1 = env_class()
#     env_name = env_1.env_name
#     env_2 = gym.make(env_name)
#     print(env_name, env_1.action_space == env_2.action_space, env_1.action_space)import gym
