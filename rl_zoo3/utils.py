import argparse
import glob
import importlib
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import stable_baselines3 as sb3
import torch as th
import yaml
from gym import spaces
from huggingface_hub import HfApi
from huggingface_sb3 import EnvironmentName, ModelName
from sb3_contrib import ARS, QRDQN, TQC, TRPO, RecurrentPPO
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack, VecNormalize

from torch import nn as nn

from bdq.bdq import BDQ
from mask_DQN.dqn_original import OriginalDQN
from mask_DQN.dqn_mask import MaskDQN


ALGOS: Dict[str, Type[BaseAlgorithm]] = {
    "a2c": A2C,
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    "ars": ARS,
    "qrdqn": QRDQN,
    "tqc": TQC,
    "trpo": TRPO,
    "ppo_lstm": RecurrentPPO,
    "bdq": BDQ,
    "dqn_original": OriginalDQN,
    "dqn_mask": MaskDQN,
}


def flatten_dict_observations(env: gym.Env) -> gym.Env:
    assert isinstance(env.observation_space, spaces.Dict)
    try:
        return gym.wrappers.FlattenObservation(env)
    except AttributeError:
        keys = env.observation_space.spaces.keys()
        return gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))


def get_wrapper_class(hyperparams: Dict[str, Any], key: str = "env_wrapper") -> Optional[Callable[[gym.Env], gym.Env]]:

    def get_module_name(wrapper_name):
        return ".".join(wrapper_name.split(".")[:-1])

    def get_class_name(wrapper_name):
        return wrapper_name.split(".")[-1]

    if key in hyperparams.keys():
        wrapper_name = hyperparams.get(key)

        if wrapper_name is None:
            return None

        if not isinstance(wrapper_name, list):
            wrapper_names = [wrapper_name]
        else:
            wrapper_names = wrapper_name

        wrapper_classes = []
        wrapper_kwargs = []
        for wrapper_name in wrapper_names:
            if isinstance(wrapper_name, dict):
                assert len(wrapper_name) == 1, (
                    "You have an error in the formatting "
                    f"of your YAML file near {wrapper_name}. "
                    "You should check the indentation."
                )
                wrapper_dict = wrapper_name
                wrapper_name = list(wrapper_dict.keys())[0]
                kwargs = wrapper_dict[wrapper_name]
            else:
                kwargs = {}
            wrapper_module = importlib.import_module(get_module_name(wrapper_name))
            wrapper_class = getattr(wrapper_module, get_class_name(wrapper_name))
            wrapper_classes.append(wrapper_class)
            wrapper_kwargs.append(kwargs)

        def wrap_env(env: gym.Env) -> gym.Env:
            for wrapper_class, kwargs in zip(wrapper_classes, wrapper_kwargs):
                env = wrapper_class(env, **kwargs)
            return env

        return wrap_env
    else:
        return None


def get_class_by_name(name: str) -> Type:

    def get_module_name(name: str) -> str:
        return ".".join(name.split(".")[:-1])

    def get_class_name(name: str) -> str:
        return name.split(".")[-1]

    module = importlib.import_module(get_module_name(name))
    return getattr(module, get_class_name(name))


def get_callback_list(hyperparams: Dict[str, Any]) -> List[BaseCallback]:

    callbacks: List[BaseCallback] = []

    if "callback" in hyperparams.keys():
        callback_name = hyperparams.get("callback")

        if callback_name is None:
            return callbacks

        if not isinstance(callback_name, list):
            callback_names = [callback_name]
        else:
            callback_names = callback_name

        for callback_name in callback_names:
            if isinstance(callback_name, dict):
                assert len(callback_name) == 1, (
                    "You have an error in the formatting "
                    f"of your YAML file near {callback_name}. "
                    "You should check the indentation."
                )
                callback_dict = callback_name
                callback_name = list(callback_dict.keys())[0]
                kwargs = callback_dict[callback_name]
            else:
                kwargs = {}

            callback_class = get_class_by_name(callback_name)
            callbacks.append(callback_class(**kwargs))

    return callbacks


def create_test_env(
    env_id: str,
    n_envs: int = 1,
    stats_path: Optional[str] = None,
    seed: int = 0,
    log_dir: Optional[str] = None,
    should_render: bool = True,
    hyperparams: Optional[Dict[str, Any]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    from rl_zoo3.exp_manager import ExperimentManager

    assert hyperparams is not None
    env_wrapper = get_wrapper_class(hyperparams)

    hyperparams = {} if hyperparams is None else hyperparams

    if "env_wrapper" in hyperparams.keys():
        del hyperparams["env_wrapper"]

    vec_env_kwargs: Dict[str, Any] = {}
    vec_env_cls = DummyVecEnv
    if n_envs > 1 or (ExperimentManager.is_bullet(env_id) and should_render):
        vec_env_cls = SubprocVecEnv

    if ExperimentManager.is_panda_gym(env_id) and should_render:
        if env_kwargs is None:
            env_kwargs = {"render": True}
        else:
            env_kwargs["render"] = True

    env = make_vec_env(
        env_id,
        n_envs=n_envs,
        monitor_dir=log_dir,
        seed=seed,
        wrapper_class=env_wrapper,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
    )

    if "vec_env_wrapper" in hyperparams.keys():

        vec_env_wrapper = get_wrapper_class(hyperparams, "vec_env_wrapper")
        assert vec_env_wrapper is not None
        env = vec_env_wrapper(env)
        del hyperparams["vec_env_wrapper"]

    if stats_path is not None:
        if hyperparams["normalize"]:
            print("Loading running average")
            print(f"with params: {hyperparams['normalize_kwargs']}")
            path_ = os.path.join(stats_path, "vecnormalize.pkl")
            if os.path.exists(path_):
                env = VecNormalize.load(path_, env)
                env.training = False
                env.norm_reward = False
            else:
                raise ValueError(f"VecNormalize stats {path_} not found")

        n_stack = hyperparams.get("frame_stack", 0)
        if n_stack > 0:
            print(f"Stacking {n_stack} frames")
            env = VecFrameStack(env, n_stack)
    return env


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    initial_value_ = float(initial_value)

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value_

    return func


def get_trained_models(log_folder: str) -> Dict[str, Tuple[str, str]]:
    trained_models = {}
    for algo in os.listdir(log_folder):
        if not os.path.isdir(os.path.join(log_folder, algo)):
            continue
        for model_folder in os.listdir(os.path.join(log_folder, algo)):
            args_files = glob.glob(os.path.join(log_folder, algo, model_folder, "*/args.yml"))
            if len(args_files) != 1:
                continue
            with open(args_files[0], "r") as fh:
                env_id = yaml.load(fh, Loader=yaml.UnsafeLoader)["env"]

            model_name = ModelName(algo, EnvironmentName(env_id))
            trained_models[model_name] = (algo, env_id)
    return trained_models


def get_hf_trained_models(organization: str = "sb3", check_filename: bool = False) -> Dict[str, Tuple[str, str]]:
    api = HfApi()
    models = api.list_models(author=organization, cardData=True)

    trained_models = {}
    for model in models:
        try:
            env_id = model.cardData["model-index"][0]["results"][0]["dataset"]["name"]
            algo = model.cardData["model-index"][0]["name"].lower()
            if algo == "recurrentppo":
                algo = "ppo_lstm"
        except (KeyError, IndexError):
            print(f"Skipping {model.modelId}")
            continue

        env_name = EnvironmentName(env_id)
        model_name = ModelName(algo, env_name)

        if check_filename and not any(f.rfilename == model_name.filename for f in api.model_info(model.modelId).siblings):
            continue

        trained_models[model_name] = (algo, env_id)

    return trained_models


def get_latest_run_id(log_path: str, env_name: EnvironmentName) -> int:
    max_run_id = 0
    for path in glob.glob(os.path.join(log_path, env_name + "_[0-9]*")):
        run_id = path.split("_")[-1]
        path_without_run_id = path[: -len(run_id) - 1]
        if path_without_run_id.endswith(env_name) and run_id.isdigit() and int(run_id) > max_run_id:
            max_run_id = int(run_id)
    return max_run_id


def get_saved_hyperparams(
    stats_path: str,
    norm_reward: bool = False,
    test_mode: bool = False,
) -> Tuple[Dict[str, Any], Optional[str]]:
    hyperparams: Dict[str, Any] = {}
    if not os.path.isdir(stats_path):
        return hyperparams, None
    else:
        config_file = os.path.join(stats_path, "config.yml")
        if os.path.isfile(config_file):
            with open(os.path.join(stats_path, "config.yml")) as f:
                hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)
            hyperparams["normalize"] = hyperparams.get("normalize", False)
        else:
            obs_rms_path = os.path.join(stats_path, "obs_rms.pkl")
            hyperparams["normalize"] = os.path.isfile(obs_rms_path)

        if hyperparams["normalize"]:
            if isinstance(hyperparams["normalize"], str):
                normalize_kwargs = eval(hyperparams["normalize"])
                if test_mode:
                    normalize_kwargs["norm_reward"] = norm_reward
            else:
                normalize_kwargs = {"norm_obs": hyperparams["normalize"], "norm_reward": norm_reward}
            hyperparams["normalize_kwargs"] = normalize_kwargs
    return hyperparams, stats_path


class StoreDict(argparse.Action):

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        arg_dict = {}
        for arguments in values:
            key = arguments.split(":")[0]
            value = ":".join(arguments.split(":")[1:])
            arg_dict[key] = eval(value)
        setattr(namespace, self.dest, arg_dict)


def get_model_path(
    exp_id: int,
    folder: str,
    algo: str,
    env_name: EnvironmentName,
    load_best: bool = False,
    load_checkpoint: Optional[str] = None,
    load_last_checkpoint: bool = False,
) -> Tuple[str, str, str]:

    if exp_id == 0:
        exp_id = get_latest_run_id(os.path.join(folder, algo), env_name)
        print(f"Loading latest experiment, id={exp_id}")
    if exp_id > 0:
        log_path = os.path.join(folder, algo, f"{env_name}_{exp_id}")
    else:
        log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), f"The {log_path} folder was not found"

    model_name = ModelName(algo, env_name)

    if load_best:
        model_path = os.path.join(log_path, "best_model.zip")
        name_prefix = f"best-model-{model_name}"
    elif load_checkpoint is not None:
        model_path = os.path.join(log_path, f"rl_model_{load_checkpoint}_steps.zip")
        name_prefix = f"checkpoint-{load_checkpoint}-{model_name}"
    elif load_last_checkpoint:
        checkpoints = glob.glob(os.path.join(log_path, "rl_model_*_steps.zip"))
        if len(checkpoints) == 0:
            raise ValueError(f"No checkpoint found for {algo} on {env_name}, path: {log_path}")

        def step_count(checkpoint_path: str) -> int:
            return int(checkpoint_path.split("_")[-2])

        checkpoints = sorted(checkpoints, key=step_count)
        model_path = checkpoints[-1]
        name_prefix = f"checkpoint-{step_count(model_path)}-{model_name}"
    else:
        model_path = os.path.join(log_path, f"{env_name}.zip")
        name_prefix = f"final-model-{model_name}"

    found = os.path.isfile(model_path)
    if not found:
        raise ValueError(f"No model found for {algo} on {env_name}, path: {model_path}")

    return name_prefix, model_path, log_path
