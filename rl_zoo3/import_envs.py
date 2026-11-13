import gym
from gym.envs.registration import register

from rl_zoo3.wrappers import MaskVelocityWrapper

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import highway_env
except ImportError:
    highway_env = None

try:
    import neck_rl
except ImportError:
    neck_rl = None

try:
    import mocca_envs
except ImportError:
    mocca_envs = None

try:
    import custom_envs
except ImportError:
    custom_envs = None

try:
    import gym_donkeycar
except ImportError:
    gym_donkeycar = None

try:
    import panda_gym
except ImportError:
    panda_gym = None

try:
    import rocket_lander_gym
except ImportError:
    rocket_lander_gym = None


def create_no_vel_env(env_id: str):
    def make_env():
        env = gym.make(env_id)
        env = MaskVelocityWrapper(env)
        return env

    return make_env


for env_id in MaskVelocityWrapper.velocity_indices.keys():
    name, version = env_id.split("-v")
    register(
        id=f"{name}NoVel-v{version}",
        entry_point=create_no_vel_env(env_id),
    )
