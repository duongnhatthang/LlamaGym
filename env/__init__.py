from .atari import register_environments
from .atari import SpaceInvaders_translator, Pong_translator

from .toy_text import taxi_translator
from .toy_text import cliffwalking_translator
from .toy_text import frozenlake_translator

from .classic_control import cartpole_translator
from .classic_control import acrobot_translator
from .classic_control import mountaincar_translator

register_environments()

# REGISTRY = {}
# REGISTRY["RepresentedSpaceInvaders_init_translator"] = SpaceInvaders_translator.GameDescriber
# REGISTRY["RepresentedSpaceInvaders_basic_translator"] = SpaceInvaders_translator.TransitionTranslator