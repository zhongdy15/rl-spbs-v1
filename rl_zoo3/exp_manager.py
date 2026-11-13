import argparse
import importlib
import os
import pickle as pkl
import time
import warnings
from collections import OrderedDict
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import optuna
import torch as th
import yaml
from gym import spaces
from huggingface_sb3 import EnvironmentName
from optuna.pruners import BasePruner, MedianPruner, NopPruner, SuccessiveHalvingPruner
from optuna.samplers import BaseSampler, RandomSampler, TPESampler
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from optuna.visualization import plot_optimization_history, plot_param_importances
from sb3_contrib.common.vec_env import AsyncEval

from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, ProgressBarCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecFrameStack,
    VecNormalize,
    VecTransposeImage,
    is_vecenv_wrapped,
)

from torch import nn as nn

import rl_zoo3.import_envs
from rl_zoo3.callbacks import SaveVecNormalizeCallback, TrialEvalCallback
from rl_zoo3.hyperparams_opt import HYPERPARAMS_SAMPLER
from rl_zoo3.utils import ALGOS, get_callback_list, get_class_by_name, get_latest_run_id, get_wrapper_class, linear_schedule


class ExperimentManager:

    def __init__(
        self,
        args: argparse.Namespace,
        algo: str,
        env_id: str,
        log_folder: str,
        tensorboard_log: str = "",
        n_timesteps: int = 0,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        save_freq: int = -1,
        hyperparams: Optional[Dict[str, Any]] = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
        trained_agent: str = "",
        optimize_hyperparameters: bool = False,
        storage: Optional[str] = None,
        study_name: Optional[str] = None,
        n_trials: int = 1,
        max_total_trials: Optional[int] = None,
        n_jobs: int = 1,
        sampler: str = "tpe",
        pruner: str = "median",
        optimization_log_path: Optional[str] = None,
        n_startup_trials: int = 0,
        n_evaluations: int = 1,
        truncate_last_trajectory: bool = False,
        uuid_str: str = "",
        seed: int = 0,
        log_interval: int = 0,
        save_replay_buffer: bool = False,
        verbose: int = 1,
        vec_env_type: str = "dummy",
        n_eval_envs: int = 1,
        no_optim_plots: bool = False,
        device: Union[th.device, str] = "auto",
        config: Optional[str] = None,
        show_progress: bool = False,
    ):
        super().__init__()
        self.algo = algo
        self.env_name = EnvironmentName(env_id)
        self.custom_hyperparams = hyperparams
        if (Path(__file__).parent / "hyperparams").is_dir():
            default_path = Path(__file__).parent
        else:
            default_path = Path(__file__).parent.parent

        self.config = config or str(default_path / f"hyperparams/{self.algo}.yml")
        self.env_kwargs: Dict[str, Any] = {} if env_kwargs is None else env_kwargs
        self.n_timesteps = n_timesteps
        self.normalize = False
        self.normalize_kwargs: Dict[str, Any] = {}
        self.env_wrapper = None
        self.frame_stack = None
        self.seed = seed
        self.optimization_log_path = optimization_log_path

        self.vec_env_class = {"dummy": DummyVecEnv, "subproc": SubprocVecEnv}[vec_env_type]
        self.vec_env_wrapper = None

        self.vec_env_kwargs: Dict[str, Any] = {}

        self.specified_callbacks: List = []
        self.callbacks: List[BaseCallback] = []
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.n_eval_envs = n_eval_envs

        self.n_envs = 1
        self.n_actions = None
        self._hyperparams: Dict[str, Any] = {}
        self.monitor_kwargs: Dict[str, Any] = {}

        self.trained_agent = trained_agent
        self.continue_training = trained_agent.endswith(".zip") and os.path.isfile(trained_agent)
        self.truncate_last_trajectory = truncate_last_trajectory

        self._is_atari = self.is_atari(env_id)
        self.optimize_hyperparameters = optimize_hyperparameters
        self.storage = storage
        self.study_name = study_name
        self.no_optim_plots = no_optim_plots
        self.n_trials = n_trials
        self.max_total_trials = max_total_trials
        self.n_jobs = n_jobs
        self.sampler = sampler
        self.pruner = pruner
        self.n_startup_trials = n_startup_trials
        self.n_evaluations = n_evaluations
        self.deterministic_eval = not self.is_atari(env_id)
        self.device = device

        self.log_folder = log_folder
        self.tensorboard_log = None if tensorboard_log == "" else os.path.join(tensorboard_log, self.env_name)
        self.verbose = verbose
        self.args = args
        self.log_interval = log_interval
        self.save_replay_buffer = save_replay_buffer
        self.show_progress = show_progress

        self.log_path = f"{log_folder}/{self.algo}/"
        self.save_path = os.path.join(
            self.log_path, f"{self.env_name}_{get_latest_run_id(self.log_path, self.env_name) + 1}{uuid_str}"
        )
        self.params_path = f"{self.save_path}/{self.env_name}"

    def setup_experiment(self) -> Optional[Tuple[BaseAlgorithm, Dict[str, Any]]]:
        hyperparams, saved_hyperparams = self.read_hyperparameters()
        hyperparams, self.env_wrapper, self.callbacks, self.vec_env_wrapper = self._preprocess_hyperparams(hyperparams)

        self.create_log_folder()
        self.create_callbacks()

        n_envs = 1 if self.algo == "ars" or self.optimize_hyperparameters else self.n_envs
        env = self.create_envs(n_envs, no_log=False)

        self._hyperparams = self._preprocess_action_noise(hyperparams, saved_hyperparams, env)

        if self.continue_training:
            model = self._load_pretrained_agent(self._hyperparams, env)
        elif self.optimize_hyperparameters:
            env.close()
            return None
        else:
            model = ALGOS[self.algo](
                env=env,
                tensorboard_log=self.tensorboard_log,
                seed=self.seed,
                verbose=self.verbose,
                device=self.device,
                **self._hyperparams,
            )

        self._save_config(saved_hyperparams)
        return model, saved_hyperparams

    def learn(self, model: BaseAlgorithm) -> None:
        kwargs = {}
        if self.log_interval > -1:
            kwargs = {"log_interval": self.log_interval}

        if len(self.callbacks) > 0:
            kwargs["callback"] = self.callbacks

        if self.algo == "ars" and self.n_envs > 1:
            kwargs["async_eval"] = AsyncEval(
                [lambda: self.create_envs(n_envs=1, no_log=True) for _ in range(self.n_envs)], model.policy
            )

        try:
            model.learn(self.n_timesteps, **kwargs)
        except KeyboardInterrupt:
            pass
        finally:
            if len(self.callbacks) > 0:
                self.callbacks[0].on_training_end()
            try:
                model.env.close()
            except EOFError:
                pass

    def save_trained_model(self, model: BaseAlgorithm) -> None:
        print(f"Saving to {self.save_path}")
        model.save(f"{self.save_path}/{self.env_name}")

        if hasattr(model, "save_replay_buffer") and self.save_replay_buffer:
            print("Saving replay buffer")
            model.save_replay_buffer(os.path.join(self.save_path, "replay_buffer.pkl"))

        if self.normalize:
            model.get_vec_normalize_env().save(os.path.join(self.params_path, "vecnormalize.pkl"))

    def _save_config(self, saved_hyperparams: Dict[str, Any]) -> None:
        with open(os.path.join(self.params_path, "config.yml"), "w") as f:
            yaml.dump(saved_hyperparams, f)

        with open(os.path.join(self.params_path, "args.yml"), "w") as f:
            ordered_args = OrderedDict([(key, vars(self.args)[key]) for key in sorted(vars(self.args).keys())])
            yaml.dump(ordered_args, f)

        print(f"Log path: {self.save_path}")

    def read_hyperparameters(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        print(f"Loading hyperparameters from: {self.config}")

        if self.config.endswith(".yml") or self.config.endswith(".yaml"):
            with open(self.config) as f:
                hyperparams_dict = yaml.safe_load(f)
        elif self.config.endswith(".py"):
            global_variables = {}
            exec(Path(self.config).read_text(), global_variables)
            hyperparams_dict = global_variables["hyperparams"]
        else:
            hyperparams_dict = importlib.import_module(self.config).hyperparams

        if self.env_name.gym_id in list(hyperparams_dict.keys()):
            hyperparams = hyperparams_dict[self.env_name.gym_id]
        elif self._is_atari:
            hyperparams = hyperparams_dict["atari"]
        else:
            raise ValueError(f"Hyperparameters not found for {self.algo}-{self.env_name.gym_id} in {self.config}")

        if self.custom_hyperparams is not None:
            hyperparams.update(self.custom_hyperparams)
        saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

        print("Default hyperparameters for environment (ones being tuned will be overridden):")
        pprint(saved_hyperparams)

        return hyperparams, saved_hyperparams

    @staticmethod
    def _preprocess_schedules(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        for key in ["learning_rate", "clip_range", "clip_range_vf", "delta_std"]:
            if key not in hyperparams:
                continue
            if isinstance(hyperparams[key], str):
                schedule, initial_value = hyperparams[key].split("_")
                initial_value = float(initial_value)
                hyperparams[key] = linear_schedule(initial_value)
            elif isinstance(hyperparams[key], (float, int)):
                if hyperparams[key] < 0:
                    continue
                hyperparams[key] = constant_fn(float(hyperparams[key]))
            else:
                raise ValueError(f"Invalid value for {key}: {hyperparams[key]}")
        return hyperparams

    def _preprocess_normalization(self, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        if "normalize" in hyperparams.keys():
            self.normalize = hyperparams["normalize"]

            if isinstance(self.normalize, str):
                self.normalize_kwargs = eval(self.normalize)
                self.normalize = True

            if isinstance(self.normalize, dict):
                self.normalize_kwargs = self.normalize
                self.normalize = True

            if "gamma" in hyperparams:
                self.normalize_kwargs["gamma"] = hyperparams["gamma"]

            del hyperparams["normalize"]
        return hyperparams

    def _preprocess_hyperparams(
        self, hyperparams: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Optional[Callable], List[BaseCallback], Optional[Callable]]:
        self.n_envs = hyperparams.get("n_envs", 1)

        if self.verbose > 0:
            print(f"Using {self.n_envs} environments")

        hyperparams = self._preprocess_schedules(hyperparams)

        if "train_freq" in hyperparams and isinstance(hyperparams["train_freq"], list):
            hyperparams["train_freq"] = tuple(hyperparams["train_freq"])

        if self.n_timesteps > 0:
            if self.verbose:
                print(f"Overwriting n_timesteps with n={self.n_timesteps}")
        else:
            self.n_timesteps = int(hyperparams["n_timesteps"])

        if self.n_evaluations is None and self.optimize_hyperparameters:
            self.n_evaluations = max(1, self.n_timesteps // int(1e5))
            print(
                f"Doing {self.n_evaluations} intermediate evaluations for pruning based on the number of timesteps."
                " (1 evaluation every 100k timesteps)"
            )

        hyperparams = self._preprocess_normalization(hyperparams)

        for kwargs_key in {"policy_kwargs", "replay_buffer_class", "replay_buffer_kwargs"}:
            if kwargs_key in hyperparams.keys() and isinstance(hyperparams[kwargs_key], str):
                hyperparams[kwargs_key] = eval(hyperparams[kwargs_key])

        if "monitor_kwargs" in hyperparams.keys():
            self.monitor_kwargs = hyperparams["monitor_kwargs"]
            if isinstance(self.monitor_kwargs, str):
                self.monitor_kwargs = eval(self.monitor_kwargs)
            del hyperparams["monitor_kwargs"]

        if "n_envs" in hyperparams.keys():
            del hyperparams["n_envs"]
        del hyperparams["n_timesteps"]

        if "frame_stack" in hyperparams.keys():
            self.frame_stack = hyperparams["frame_stack"]
            del hyperparams["frame_stack"]

        if "policy" in hyperparams and "." in hyperparams["policy"]:
            hyperparams["policy"] = get_class_by_name(hyperparams["policy"])

        env_wrapper = get_wrapper_class(hyperparams)
        if "env_wrapper" in hyperparams.keys():
            del hyperparams["env_wrapper"]

        vec_env_wrapper = get_wrapper_class(hyperparams, "vec_env_wrapper")
        if "vec_env_wrapper" in hyperparams.keys():
            del hyperparams["vec_env_wrapper"]

        callbacks = get_callback_list(hyperparams)
        if "callback" in hyperparams.keys():
            self.specified_callbacks = hyperparams["callback"]
            del hyperparams["callback"]

        return hyperparams, env_wrapper, callbacks, vec_env_wrapper

    def _preprocess_action_noise(
        self, hyperparams: Dict[str, Any], saved_hyperparams: Dict[str, Any], env: VecEnv
    ) -> Dict[str, Any]:
        if hyperparams.get("noise_type") is not None:
            noise_type = hyperparams["noise_type"].strip()
            noise_std = hyperparams["noise_std"]

            self.n_actions = env.action_space.shape[0]

            if "normal" in noise_type:
                hyperparams["action_noise"] = NormalActionNoise(
                    mean=np.zeros(self.n_actions),
                    sigma=noise_std * np.ones(self.n_actions),
                )
            elif "ornstein-uhlenbeck" in noise_type:
                hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
                    mean=np.zeros(self.n_actions),
                    sigma=noise_std * np.ones(self.n_actions),
                )
            else:
                raise RuntimeError(f'Unknown noise type "{noise_type}"')

            print(f"Applying {noise_type} noise with std {noise_std}")

            del hyperparams["noise_type"]
            del hyperparams["noise_std"]

        return hyperparams

    def create_log_folder(self):
        os.makedirs(self.params_path, exist_ok=True)

    def create_callbacks(self):

        if self.show_progress:
            self.callbacks.append(ProgressBarCallback())

        if self.save_freq > 0:
            self.save_freq = max(self.save_freq // self.n_envs, 1)
            self.callbacks.append(
                CheckpointCallback(
                    save_freq=self.save_freq,
                    save_path=self.save_path,
                    name_prefix="rl_model",
                    verbose=1,
                )
            )

        if self.eval_freq > 0 and not self.optimize_hyperparameters:
            self.eval_freq = max(self.eval_freq // self.n_envs, 1)

            if self.verbose > 0:
                print("Creating test environment")

            save_vec_normalize = SaveVecNormalizeCallback(save_freq=1, save_path=self.params_path)
            eval_callback = EvalCallback(
                self.create_envs(self.n_eval_envs, eval_env=True),
                callback_on_new_best=save_vec_normalize,
                best_model_save_path=self.save_path,
                n_eval_episodes=self.n_eval_episodes,
                log_path=self.save_path,
                eval_freq=self.eval_freq,
                deterministic=self.deterministic_eval,
            )

            self.callbacks.append(eval_callback)

    @staticmethod
    def is_atari(env_id: str) -> bool:
        entry_point = gym.envs.registry.env_specs[env_id].entry_point
        return "AtariEnv" in str(entry_point)

    @staticmethod
    def is_bullet(env_id: str) -> bool:
        entry_point = gym.envs.registry.env_specs[env_id].entry_point
        return "pybullet_envs" in str(entry_point)

    @staticmethod
    def is_robotics_env(env_id: str) -> bool:
        entry_point = gym.envs.registry.env_specs[env_id].entry_point
        return "gym.envs.robotics" in str(entry_point) or "panda_gym.envs" in str(entry_point)

    @staticmethod
    def is_panda_gym(env_id: str) -> bool:
        entry_point = gym.envs.registry.env_specs[env_id].entry_point
        return "panda_gym.envs" in str(entry_point)

    def _maybe_normalize(self, env: VecEnv, eval_env: bool) -> VecEnv:
        path_ = os.path.join(os.path.dirname(self.trained_agent), self.env_name)
        path_ = os.path.join(path_, "vecnormalize.pkl")

        if os.path.exists(path_):
            print("Loading saved VecNormalize stats")
            env = VecNormalize.load(path_, env)
            if eval_env:
                env.training = False
                env.norm_reward = False

        elif self.normalize:
            local_normalize_kwargs = self.normalize_kwargs.copy()
            if eval_env:
                local_normalize_kwargs["norm_reward"] = False
                local_normalize_kwargs["training"] = False

            if self.verbose > 0:
                if len(local_normalize_kwargs) > 0:
                    print(f"Normalization activated: {local_normalize_kwargs}")
                else:
                    print("Normalizing input and reward")
            env = VecNormalize(env, **local_normalize_kwargs)
        return env

    def create_envs(self, n_envs: int, eval_env: bool = False, no_log: bool = False) -> VecEnv:
        log_dir = None if eval_env or no_log else self.save_path

        if (
            "Neck" in self.env_name.gym_id
            or self.is_robotics_env(self.env_name.gym_id)
            or "parking-v0" in self.env_name.gym_id
            and len(self.monitor_kwargs) == 0
        ):
            self.monitor_kwargs = dict(info_keywords=("is_success",))

        spec = gym.spec(self.env_name.gym_id)

        def make_env(**kwargs) -> gym.Env:
            env = spec.make(**kwargs)
            return env

        env = make_vec_env(
            make_env,
            n_envs=n_envs,
            seed=self.seed,
            env_kwargs=self.env_kwargs,
            monitor_dir=log_dir,
            wrapper_class=self.env_wrapper,
            vec_env_cls=self.vec_env_class,
            vec_env_kwargs=self.vec_env_kwargs,
            monitor_kwargs=self.monitor_kwargs,
        )

        if self.vec_env_wrapper is not None:
            env = self.vec_env_wrapper(env)

        env = self._maybe_normalize(env, eval_env)

        if self.frame_stack is not None:
            n_stack = self.frame_stack
            env = VecFrameStack(env, n_stack)
            if self.verbose > 0:
                print(f"Stacking {n_stack} frames")

        if not is_vecenv_wrapped(env, VecTransposeImage):
            wrap_with_vectranspose = False
            if isinstance(env.observation_space, spaces.Dict):
                for space in env.observation_space.spaces.values():
                    wrap_with_vectranspose = wrap_with_vectranspose or (
                        is_image_space(space) and not is_image_space_channels_first(space)
                    )
            else:
                wrap_with_vectranspose = is_image_space(env.observation_space) and not is_image_space_channels_first(
                    env.observation_space
                )

            if wrap_with_vectranspose:
                if self.verbose >= 1:
                    print("Wrapping the env in a VecTransposeImage.")
                env = VecTransposeImage(env)

        return env

    def _load_pretrained_agent(self, hyperparams: Dict[str, Any], env: VecEnv) -> BaseAlgorithm:
        print("Loading pretrained agent")
        del hyperparams["policy"]

        if "policy_kwargs" in hyperparams.keys():
            del hyperparams["policy_kwargs"]

        model = ALGOS[self.algo].load(
            self.trained_agent,
            env=env,
            seed=self.seed,
            tensorboard_log=self.tensorboard_log,
            verbose=self.verbose,
            device=self.device,
            **hyperparams,
        )

        replay_buffer_path = os.path.join(os.path.dirname(self.trained_agent), "replay_buffer.pkl")

        if os.path.exists(replay_buffer_path):
            print("Loading replay buffer")
            model.load_replay_buffer(replay_buffer_path, truncate_last_traj=self.truncate_last_trajectory)
        return model

    def _create_sampler(self, sampler_method: str) -> BaseSampler:
        if sampler_method == "random":
            sampler = RandomSampler(seed=self.seed)
        elif sampler_method == "tpe":
            sampler = TPESampler(n_startup_trials=self.n_startup_trials, seed=self.seed, multivariate=True)
        elif sampler_method == "skopt":
            from optuna.integration.skopt import SkoptSampler

            sampler = SkoptSampler(skopt_kwargs={"base_estimator": "GP", "acq_func": "gp_hedge"})
        else:
            raise ValueError(f"Unknown sampler: {sampler_method}")
        return sampler

    def _create_pruner(self, pruner_method: str) -> BasePruner:
        if pruner_method == "halving":
            pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
        elif pruner_method == "median":
            pruner = MedianPruner(n_startup_trials=self.n_startup_trials, n_warmup_steps=self.n_evaluations // 3)
        elif pruner_method == "none":
            pruner = NopPruner()
        else:
            raise ValueError(f"Unknown pruner: {pruner_method}")
        return pruner

    def objective(self, trial: optuna.Trial) -> float:

        kwargs = self._hyperparams.copy()

        trial.n_actions = self.n_actions
        trial.using_her_replay_buffer = kwargs.get("replay_buffer_class") == HerReplayBuffer
        if trial.using_her_replay_buffer:
            trial.her_kwargs = kwargs.get("replay_buffer_kwargs", {})
        sampled_hyperparams = HYPERPARAMS_SAMPLER[self.algo](trial)
        kwargs.update(sampled_hyperparams)

        n_envs = 1 if self.algo == "ars" else self.n_envs
        env = self.create_envs(n_envs, no_log=True)

        trial_verbosity = 0
        if self.verbose >= 2:
            trial_verbosity = self.verbose

        model = ALGOS[self.algo](
            env=env,
            tensorboard_log=None,
            seed=None,
            verbose=trial_verbosity,
            device=self.device,
            **kwargs,
        )

        eval_env = self.create_envs(n_envs=self.n_eval_envs, eval_env=True)

        optuna_eval_freq = int(self.n_timesteps / self.n_evaluations)
        optuna_eval_freq = max(optuna_eval_freq // self.n_envs, 1)
        path = None
        if self.optimization_log_path is not None:
            path = os.path.join(self.optimization_log_path, f"trial_{str(trial.number)}")
        callbacks = get_callback_list({"callback": self.specified_callbacks})
        eval_callback = TrialEvalCallback(
            eval_env,
            trial,
            best_model_save_path=path,
            log_path=path,
            n_eval_episodes=self.n_eval_episodes,
            eval_freq=optuna_eval_freq,
            deterministic=self.deterministic_eval,
        )
        callbacks.append(eval_callback)

        learn_kwargs = {}
        if self.algo == "ars" and self.n_envs > 1:
            learn_kwargs["async_eval"] = AsyncEval(
                [lambda: self.create_envs(n_envs=1, no_log=True) for _ in range(self.n_envs)], model.policy
            )

        try:
            model.learn(self.n_timesteps, callback=callbacks, **learn_kwargs)
            model.env.close()
            eval_env.close()
        except (AssertionError, ValueError) as e:
            model.env.close()
            eval_env.close()
            print(e)
            print("============")
            print("Sampled hyperparams:")
            pprint(sampled_hyperparams)
            raise optuna.exceptions.TrialPruned()
        is_pruned = eval_callback.is_pruned
        reward = eval_callback.last_mean_reward

        del model.env, eval_env
        del model

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return reward

    def hyperparameters_optimization(self) -> None:

        if self.verbose > 0:
            print("Optimizing hyperparameters")

        if self.storage is not None and self.study_name is None:
            warnings.warn(
                f"You passed a remote storage: {self.storage} but no `--study-name`."
                "The study name will be generated by Optuna, make sure to re-use the same study name "
                "when you want to do distributed hyperparameter optimization."
            )

        if self.tensorboard_log is not None:
            warnings.warn("Tensorboard log is deactivated when running hyperparameter optimization")
            self.tensorboard_log = None

        sampler = self._create_sampler(self.sampler)
        pruner = self._create_pruner(self.pruner)

        if self.verbose > 0:
            print(f"Sampler: {self.sampler} - Pruner: {self.pruner}")

        study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            storage=self.storage,
            study_name=self.study_name,
            load_if_exists=True,
            direction="maximize",
        )

        try:
            if self.max_total_trials is not None:
                counted_states = [
                    TrialState.COMPLETE,
                    TrialState.RUNNING,
                    TrialState.PRUNED,
                ]
                completed_trials = len(study.get_trials(states=counted_states))
                if completed_trials < self.max_total_trials:
                    study.optimize(
                        self.objective,
                        n_jobs=self.n_jobs,
                        callbacks=[
                            MaxTrialsCallback(
                                self.max_total_trials,
                                states=counted_states,
                            )
                        ],
                    )
            else:
                study.optimize(self.objective, n_jobs=self.n_jobs, n_trials=self.n_trials)
        except KeyboardInterrupt:
            pass

        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("Value: ", trial.value)

        print("Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        report_name = (
            f"report_{self.env_name}_{self.n_trials}-trials-{self.n_timesteps}"
            f"-{self.sampler}-{self.pruner}_{int(time.time())}"
        )

        log_path = os.path.join(self.log_folder, self.algo, report_name)

        if self.verbose:
            print(f"Writing report to {log_path}")

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        study.trials_dataframe().to_csv(f"{log_path}.csv")

        with open(f"{log_path}.pkl", "wb+") as f:
            pkl.dump(study, f)

        if self.no_optim_plots:
            return

        try:
            fig1 = plot_optimization_history(study)
            fig2 = plot_param_importances(study)

            fig1.show()
            fig2.show()
        except (ValueError, ImportError, RuntimeError):
            pass
