import gym
import numpy as np
from gym import spaces
from typing import Callable, Union, Dict, Any, Tuple
from action_utils.action_transformation import action_index_to_array, array_to_action_index, get_action_mask_fast, build_mask_from_recommendations, build_mask_from_matrix
from llm_baseline_prompt.llm_chat import llm_chat
import json_repair
import configparser
from interpret_obs import interpret_obs, get_current_fan_speed_from_obs
from rl_zoo3.action_mask_action_feasibility_memory import ActionFeasibilityMemory, make_afm_mask_fn
from dataclasses import dataclass
from typing import Union, Callable, Optional, Dict, List, Tuple
import logging


ActionMaskFn = Callable[[Union[np.ndarray, Dict[str, np.ndarray]], int, spaces.Discrete], np.ndarray]

print("===================== llm config loading =====================")
config = configparser.ConfigParser()
config.read('llm_baseline_prompt/config/config.ini', encoding='utf-8')
test_model_key = config['llm_api']['model'].split('/')[-1]

with open('llm_baseline_prompt/zero_shot_prompt_action_reduce.txt', 'r', encoding='utf-8') as f:
    zero_shot_prompt_template = f.read()

with open('llm_baseline_prompt/zero_shot_prompt_action_candidates.txt', 'r', encoding='utf-8') as f:
    zero_shot_prompt_action_candidates_template = f.read()

print(f"===================== llm config loading: {test_model_key} =====================")


print("===================== knn config loading =====================")
from sklearn.neighbors import NearestNeighbors

class KNNCacheStorage:
    _instance = None
    knn_model = None
    expert_masks = None
    expert_obs = None

    @classmethod
    def initialize(cls, npz_path):
        if cls.knn_model is not None:
            return

        print(f"[KNNCache] Loading data from {npz_path}...")
        try:
            data = np.load(npz_path)
            cls.expert_obs = data["obs"]
            cls.expert_masks = data["valid_mask"]

            print(f"[KNNCache] Building KNN model with {len(cls.expert_obs)} samples...")
            cls.knn_model = NearestNeighbors(n_neighbors=1, metric="euclidean")
            cls.knn_model.fit(cls.expert_obs)
            print("[KNNCache] Initialization complete.")

        except FileNotFoundError:
            raise FileNotFoundError(f"[KNNCache] Error: Cannot find {npz_path}")
        except Exception as e:
            raise RuntimeError(f"[KNNCache] Initialization failed: {e}")

    @classmethod
    def is_ready(cls):
        return cls.knn_model is not None


knn_data_path = "knn_sft_dataset_construction/output3_obs_with_valid_mask_N50_ep0.00.npz"
KNNCacheStorage.initialize(knn_data_path)

print("===================== knn config loading =====================")

print("===================== afm config loading =====================")

@dataclass
class AFMConfig:
    knn_data_path: str = "knn_sft_dataset_construction/output3_obs_with_valid_mask_N50_ep0.00.npz"
    initial_load_n: Optional[int] = 50000
    initial_load_frac: Optional[float] = None
    tau: float = 12.0
    max_size: int = 300_000
    M: int = 16
    ef_construction: int = 100
    ef_search: int = 32
    seed: int = 42
    log_every: int = 500


afm_config = AFMConfig(
    knn_data_path="knn_sft_dataset_construction/output3_obs_with_valid_mask_N50_ep0.00.npz",
    initial_load_n=80000,
    tau=150,
    max_size=400_000,
    log_every=100
)

print("===================== afm config loading successfyly=====================")

def get_action_mask_from_cache(
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        last_action: int,
        action_space: spaces.Space
) -> np.ndarray:

    if not KNNCacheStorage.is_ready():
        raise RuntimeError("[KNNCache] Error: KNN Cache not initialized! Please call initialize() first.")

    if isinstance(obs, dict):
        query_obs = obs.get('observation', list(obs.values())[0])
    else:
        query_obs = obs

    query_obs = query_obs.reshape(1, -1)

    try:
        _, indices = KNNCacheStorage.knn_model.kneighbors(query_obs)
        neighbor_idx = indices[0][0]
        retrieved_mask = KNNCacheStorage.expert_masks[neighbor_idx]
    except Exception as e:
        raise ValueError(f"Warning: KNN query failed ({e}). Falling back to allow-all.")
        retrieved_mask = None


    if retrieved_mask is None:
        raise ValueError("[KNNCache] Error: KNN Cache returned None mask! Please check the data.")

    action_mask = build_mask_from_matrix(
        valid_mask_matrix=retrieved_mask,
        action_space_n=action_space.n,
        n_rooms=7,
        base=4
    )


    if not np.any(action_mask):
        raise ValueError(f"Critical Warning: Retrieved KNN mask (idx {neighbor_idx}) is all zeros!")

    return action_mask


def get_dummy_action_mask(
    obs: Union[np.ndarray, Dict[str, np.ndarray]],
    last_action: int,
    action_space: spaces.Discrete
) -> np.ndarray:
    return np.ones(action_space.n, dtype=np.int8)


def create_fixed_action_mask(
    obs: Union[np.ndarray, Dict[str, np.ndarray]],
    last_action: int,
    action_space: spaces.Discrete
) -> np.ndarray:

    controllable_rooms = np.array([1, 1, 1, 1, 0, 1, 0], dtype=np.int8)

    old_action_index = last_action

    action_mask = get_action_mask_fast(controllable_rooms, old_action_index, action_space)

    return action_mask


def create_fixed_action_mask_2nd(
    obs: Union[np.ndarray, Dict[str, np.ndarray]],
    last_action: int,
    action_space: spaces.Discrete
) -> np.ndarray:

    controllable_rooms = np.array([1, 1, 1, 1, 0, 1, 0], dtype=np.int8)

    old_action_index = 0

    action_mask = get_action_mask_fast(controllable_rooms, old_action_index, action_space)


    return action_mask


def get_confidence_scores_from_llm(obs: np.ndarray, prompt_template: str, config: configparser.ConfigParser) -> np.ndarray:
    status_text = interpret_obs(obs)

    full_prompt = prompt_template.replace("[Status Need Replacement]", status_text)

    llm_api_key = config['llm_api']['api_key']
    llm_base_url = config['llm_api']['url']
    llm_model = config['llm_api']['model']

    messages = [
        {"role": "user", "content": full_prompt}
    ]


    response_str = llm_chat(llm_api_key, llm_model, llm_base_url, messages)


    try:
        response_dict = json_repair.loads(response_str)
        confidence_scores = response_dict["confidence_scores"]
        confidence_list = [confidence_scores[i] for i in range(7)]
        confidence_scores = np.array(confidence_list)
    except Exception as e:
        print(f"Error parsing LLM response: {e}. Using default confidence_scores [0,0,0,0,0,0,0].")
        confidence_scores = np.zeros(7)

    try:
        action_dict = json_repair.loads(response_str)
        reason = action_dict["explanation_and_thought_process"]
    except Exception as e:
        print(f"Error parsing LLM response: {e}. Using default reason 'No reason provided'.")
        reason = "No reason provided"

    return confidence_scores, reason


def get_action_mask_from_llm(
    obs: Union[np.ndarray, Dict[str, np.ndarray]],
    last_action: int,
    action_space: spaces.Discrete
) -> np.ndarray:

    old_action = get_current_fan_speed_from_obs(obs)
    old_action_index = array_to_action_index(old_action, [4, 4, 4, 4, 4, 4, 4])

    confidence_scores, reason = get_confidence_scores_from_llm(obs, zero_shot_prompt_template, config)
    max_indices = np.argsort(confidence_scores)[-6:]

    controllable_rooms = np.zeros_like(old_action, dtype=int)
    controllable_rooms[max_indices] = 1
    print("controllable_rooms:", controllable_rooms)

    action_mask = get_action_mask_fast(controllable_rooms, old_action_index, action_space)

    return action_mask


def get_recommendations_from_llm(
        obs,
        prompt_template,
        config
):
    status_text = interpret_obs(obs)

    full_prompt = prompt_template.replace("[Status Need Replacement]", status_text)

    llm_api_key = config['llm_api']['api_key']
    llm_base_url = config['llm_api']['url']
    llm_model = config['llm_api']['model']

    messages = [
        {"role": "user", "content": full_prompt}
    ]


    response_str = llm_chat(llm_api_key, llm_model, llm_base_url, messages)


    try:
        response_dict = json_repair.loads(response_str)
        recommendations = response_dict["recommendations"]
    except Exception as e:
        print(f"Error parsing LLM response: {e}.")
        recommendations = None

    try:
        action_dict = json_repair.loads(response_str)
        reason = action_dict["analysis"]
    except Exception as e:
        print(f"Error parsing LLM response: {e}. Using default reason 'No analysis provided'.")
        reason = "No analysis provided"

    return recommendations, reason


def get_action_mask_from_llm_2nd(
    obs: Union[np.ndarray, Dict[str, np.ndarray]],
    last_action: int,
    action_space: spaces.Discrete
) -> np.ndarray:
    recommendations, reason = get_recommendations_from_llm(obs, zero_shot_prompt_action_candidates_template, config)


    if not recommendations:
        print("Warning: Using empty recommendations. Falling back to allowing ALL actions (or handle specifically).")
        return np.ones(action_space.n, dtype=np.int8)

    action_mask = build_mask_from_recommendations(
        recommendations,
        action_space_n=action_space.n,
        n_rooms=7,
        base=4
    )

    if not np.any(action_mask):
        print("Critical Warning: Action mask is all zeros! LLM constraints were too strict or disjoint.")
        fallback_mask = np.zeros(action_space.n, dtype=np.int8)
        fallback_mask[last_action] = 1
        return fallback_mask

    return action_mask


afm = ActionFeasibilityMemory(afm_config)

get_action_mask_from_afm = make_afm_mask_fn(
    afm,
    get_recommendations_from_llm,
    llm_config=config,
    prompt_template=zero_shot_prompt_action_candidates_template
)


class ActionMasker(gym.Wrapper):

    def __init__(self, env: gym.Env, action_mask_fn: ActionMaskFn = get_action_mask_from_afm):
        super().__init__(env)

        if not isinstance(env.action_space, spaces.Discrete):
            raise ValueError("ActionMasker only supports Discrete action spaces.")

        self.action_mask_fn = action_mask_fn
        self.current_mask = None

    def _get_action_mask(self, obs: Union[np.ndarray, Dict[str, np.ndarray]], last_action: int) -> np.ndarray:
        return self.action_mask_fn(obs=obs, last_action=last_action, action_space=self.action_space)

    def step(self, action: int) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], float, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)

        action_mask = self._get_action_mask(obs=obs, last_action=action)
        info['action_mask'] = action_mask
        self.current_mask = action_mask

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.current_mask = self._get_action_mask(obs=obs, last_action=None)
        return obs
