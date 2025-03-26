from .atari import register_environments
from .atari import SpaceInvaders_translator

register_environments()

# REGISTRY = {}
# REGISTRY["RepresentedSpaceInvaders_init_translator"] = SpaceInvaders_translator.GameDescriber
# REGISTRY["RepresentedSpaceInvaders_basic_translator"] = SpaceInvaders_translator.TransitionTranslator