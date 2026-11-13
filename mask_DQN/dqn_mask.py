import torch as th
from gym import spaces
from torch import nn

import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from mask_DQN.replay_buffer.MaskableReplayBuffer import MaskableReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, is_vectorized_observation, polyak_update
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy

import io
import pathlib
import sys
import time
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from action_utils.action_transformation import action_index_to_array, array_to_action_index, get_action_mask_fast

import io
import pathlib
import sys
import time
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

import io
import pathlib
import time
import warnings
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union

import gym
import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common import utils
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback, ProgressBarCallback
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import check_for_nested_spaces, is_image_space, is_image_space_channels_first
from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, save_to_zip_file
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import (
    check_for_correct_spaces,
    get_device,
    get_schedule_fn,
    get_system_info,
    set_random_seed,
    update_learning_rate,
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecNormalize,
    VecTransposeImage,
    is_vecenv_wrapped,
    unwrap_vec_normalize,
)


from stable_baselines3.dqn import DQN
from mask_DQN.policies_mask import DQNPolicy_Mask

SelfDQN = TypeVar("SelfMaskDQN", bound="MaskDQN")

class MaskDQN(DQN):

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": DQNPolicy_Mask,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[MaskableReplayBuffer]] = MaskableReplayBuffer,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            target_update_interval,
            exploration_fraction,
            exploration_initial_eps,
            exploration_final_eps,
            max_grad_norm,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
        )
        self._last_action_masks: Optional[np.ndarray] = None

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                next_q_values = self.q_net_target(replay_data.next_observations)

                if hasattr(replay_data, "next_action_masks"):
                    next_q_values[replay_data.next_action_masks == 0] = -1e10

                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.q_net(replay_data.observations)

            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    @staticmethod
    def _sample_with_mask(action_space: spaces.Discrete, mask: Optional[np.ndarray] = None) -> int:
        assert isinstance(action_space, spaces.Discrete), "Action space must be Discrete"

        if mask is None:
            return action_space.sample()

        assert action_space.n == mask.shape[0], "Mask must have the same length as action space"
        valid_actions = np.where(mask == 1)[0]

        if valid_actions.size > 0:
            return action_space.np_random.choice(valid_actions)
        else:
            return action_space.sample()

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        if not deterministic and np.random.rand() < self.exploration_rate:
            if is_vectorized_observation(maybe_transpose(observation, self.observation_space), self.observation_space):
                if isinstance(observation, dict):
                    n_batch = observation[list(observation.keys())[0]].shape[0]
                else:
                    n_batch = observation.shape[0]
                if action_mask is not None:
                    action = np.array(
                        [self._sample_with_mask(self.action_space, action_mask[i]) for i in range(n_batch)])
                else:
                    action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                squeezed_mask = np.squeeze(action_mask)
                action = np.array(self._sample_with_mask(self.action_space, squeezed_mask))
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic, action_mask=action_mask)
        return action, state

    def learn(
        self: SelfDQN,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDQN:

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: MaskableReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                self.actor.reset_noise(env.num_envs)


            current_action_masks = self._last_action_masks

            assert current_action_masks is not None, "current_action_masks is None"
            assert self._last_obs is not None, "self._last_obs was not set"


            if self.num_timesteps < learning_starts:
                actions = np.array([self._sample_with_mask(self.action_space, current_action_masks[i]) for i in
                                    range(env.num_envs)])
            else:
                assert self._last_obs is not None, "self._last_obs was not set"
                actions, _ = self.predict(self._last_obs, deterministic=False, action_mask=current_action_masks)

            buffer_actions = actions

            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            callback.update_locals(locals())
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            self._update_info_buffer(infos, dones)

            next_action_masks = np.stack([info['action_mask'] for info in infos])

            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos,
                                   action_mask=self._last_action_masks, next_action_mask=next_action_masks)
            self._last_action_masks = next_action_masks

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)
                    if "terminal_observation" in infos[idx]:
                        self._last_action_masks[idx] = infos[idx]['action_mask']

                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def _store_transition(
            self,
            replay_buffer: ReplayBuffer,
            buffer_action: np.ndarray,
            new_obs: Union[np.ndarray, dict[str, np.ndarray]],
            reward: np.ndarray,
            dones: np.ndarray,
            infos: list[dict[str, Any]],
            action_mask: np.ndarray,
            next_action_mask: np.ndarray,
    ) -> None:
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        next_obs = deepcopy(new_obs_)
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(
                            next_obs[i, :])

        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            reward_,
            dones,
            infos,
            action_mask=action_mask,
            next_action_mask=next_action_mask,
        )

        self._last_obs = new_obs
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def _setup_learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            reset_num_timesteps: bool = True,
            tb_log_name: str = "run",
            progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:


        if isinstance(self.replay_buffer, HerReplayBuffer):
            replay_buffer = self.replay_buffer.replay_buffer
        else:
            replay_buffer = self.replay_buffer

        truncate_last_traj = (
                self.optimize_memory_usage
                and reset_num_timesteps
                and replay_buffer is not None
                and (replay_buffer.full or replay_buffer.pos > 0)
        )

        if truncate_last_traj:
            warnings.warn(
                "The last trajectory in the replay buffer will be truncated, "
                "see https://github.com/DLR-RM/stable-baselines3/issues/46."
                "You should use `reset_num_timesteps=False` or `optimize_memory_usage=False`"
                "to avoid that issue."
            )
            pos = (replay_buffer.pos - 1) % replay_buffer.buffer_size
            replay_buffer.dones[pos] = True


        self.start_time = time.time_ns()

        if self.ep_info_buffer is None or reset_num_timesteps:
            from collections import deque
            self.ep_info_buffer = deque(maxlen=100)
            self.ep_success_buffer = deque(maxlen=100)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        if reset_num_timesteps or self._last_obs is None:
            self._last_obs = self.env.reset()

            self._last_action_masks = np.array([e.current_mask for e in self.env.envs], dtype=np.int8)

            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
            if self._vec_normalize_env is not None:
                self._last_original_obs = self._vec_normalize_env.get_original_obs()

        if not self._custom_logger:
            self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        callback = self._init_callback(callback, progress_bar)

        return total_timesteps, callback
