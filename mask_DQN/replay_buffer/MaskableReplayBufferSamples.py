import sys
from enum import Enum
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import gym
import numpy as np
import torch as th

class MaskableReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    action_masks: th.Tensor
    next_action_masks: th.Tensor
