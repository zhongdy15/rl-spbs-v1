import numpy as np
from typing import Union, Callable, Optional, Dict, List, Tuple
import logging
from pathlib import Path
import hnswlib
import pickle
from dataclasses import dataclass
import gym
from gym import spaces
from typing import Callable, Union, Dict, Any, Tuple
from action_utils.action_transformation import action_index_to_array, array_to_action_index, get_action_mask_fast, build_mask_from_recommendations, build_mask_from_matrix
from llm_baseline_prompt.llm_chat import llm_chat
import json_repair
import configparser
from interpret_obs import interpret_obs, get_current_fan_speed_from_obs
from gym import spaces
from collections import deque

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class ActionFeasibilityMemory:
    def __init__(self, config):
        self.config = config
        self.dim = None
        self.index: Optional[hnswlib.Index] = None

        self.obs_embeddings: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []

        self.dists = deque(maxlen=getattr(config, "dist_window", 5000))

        self.stats = {
            "steps": 0, "hits": 0, "llm_calls": 0,
            "rebuilds": 0, "fallback_all": 0, "fallback_last": 0
        }

        self._preload()

        print(self.config)

    def _dist_quantiles_str(self, arr, ps=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
        if arr is None or len(arr) == 0:
            return "DistQ: n/a"
        x = np.asarray(arr, dtype=np.float32)
        qs = np.quantile(x, ps)
        parts = [f"p{int(p * 100)}={q:.4f}" for p, q in zip(ps, qs)]
        return "DistQ: " + ", ".join(parts)

    def _get_obs_embedding(self, obs: Union[np.ndarray, Dict]) -> np.ndarray:
        if isinstance(obs, dict):
            raw = obs.get('observation', next(iter(obs.values())))
        else:
            raw = obs

        emb = raw.astype(np.float32).flatten()

        if self.dim is None:
            self.dim = emb.shape[0]
        return emb

    def _init_index(self, count: int):
        if self.dim is None:
            return

        self.index = hnswlib.Index(space='l2', dim=self.dim)
        self.index.init_index(
            max_elements=self.config.max_size,
            ef_construction=self.config.ef_construction,
            M=self.config.M,
            random_seed=self.config.seed
        )
        self.index.set_ef(self.config.ef_search)

        if count > 0:
            data = np.array(self.obs_embeddings, dtype=np.float32)
            ids = np.arange(count)
            self.index.add_items(data, ids)

    def _preload(self):
        path = Path(self.config.knn_data_path)
        if not path.exists():
            logger.warning(f"[AFM] Cache file not found: {path}. Starting empty.")
            return

        try:
            data = np.load(path, allow_pickle=False)
            expert_obs = data["obs"]
            expert_masks = data["valid_mask"]

            if expert_masks.ndim != 3 or expert_masks.shape[1:] != (7, 4):
                raise ValueError(f"[AFM] valid_mask must be (N,7,4), got {expert_masks.shape}")

            n_total = len(expert_obs)
            n_load = n_total
            if self.config.initial_load_n is not None:
                n_load = min(self.config.initial_load_n, n_total)
            elif self.config.initial_load_frac is not None:
                n_load = int(n_total * self.config.initial_load_frac)

            n_load = min(n_load, self.config.max_size)

            load_obs = expert_obs[:n_load]
            load_masks = expert_masks[:n_load]

            for i in range(n_load):
                self.obs_embeddings.append(load_obs[i].flatten().astype(np.float32))
                self.masks.append(load_masks[i].astype(np.int8))

            self.dim = self.obs_embeddings[0].shape[0]
            self._init_index(count=len(self.obs_embeddings))

        except Exception as e:
            logger.error(f"[AFM] Preload failed: {e}", exc_info=True)

    def _add_online_sample(self, emb: np.ndarray, mask_data: np.ndarray):
        curr_size = len(self.obs_embeddings)

        if curr_size >= self.config.max_size:
            self._prune_and_rebuild()
            curr_size = len(self.obs_embeddings)

        new_id = curr_size

        self.obs_embeddings.append(emb)
        self.masks.append(mask_data)

        if self.index is None:
            self._init_index(count=0)

        self.index.add_items(emb.reshape(1, -1), ids=[new_id])

    def _prune_and_rebuild(self):
        self.stats["rebuilds"] += 1
        total = len(self.obs_embeddings)

        keep_n = max(1, int(total * (1 - self.config.prune_ratio)))
        keep_n = min(keep_n, total)

        self.obs_embeddings = self.obs_embeddings[-keep_n:]
        self.masks = self.masks[-keep_n:]
        self._init_index(count=len(self.obs_embeddings))

    def query_or_llm(
            self,
            obs: Union[np.ndarray, Dict],
            last_action: int,
            action_space: spaces.Discrete,
            get_llm_fn: Any,
            llm_config: Any,
            prompt_template: Any,
    ) -> np.ndarray:

        self.stats["steps"] += 1
        emb = self._get_obs_embedding(obs)

        action_mask = None
        hit = False

        if self.index is not None and self.index.get_current_count() > 0:
            labels, distances = self.index.knn_query(emb.reshape(1, -1), k=1)
            nearest_idx = labels[0][0]
            dist = distances[0][0]

            if dist < self.config.tau:
                hit = True
                self.stats["hits"] += 1
                retrieved_data = self.masks[nearest_idx]

                if retrieved_data.ndim == 2 and retrieved_data.shape == (7, 4):
                    action_mask = build_mask_from_matrix(
                        retrieved_data, action_space.n, n_rooms=7, base=4
                    )
                else:
                    action_mask = retrieved_data

        if not hit:
            self.stats["llm_calls"] += 1
            recommendations, _ = get_llm_fn(obs, prompt_template, llm_config)

            mask_to_store = None
            generated_mask = np.ones(action_space.n, dtype=np.int8)

            if recommendations:
                temp_mask = build_mask_from_recommendations(
                    recommendations, action_space.n, n_rooms=7, base=4
                )
                if np.any(temp_mask):
                    generated_mask = temp_mask
                else:
                    self.stats["fallback_all"] += 1
            else:
                self.stats["fallback_all"] += 1

            mask_to_store = generated_mask.astype(np.int8)

            self._add_online_sample(emb, mask_to_store)
            action_mask = generated_mask

        if not np.any(action_mask):
            self.stats["fallback_last"] += 1
            action_mask = np.zeros(action_space.n, dtype=np.int8)
            action_mask[last_action] = 1

        self.dists.append(dist)

        print("steps:" + str(self.stats["steps"]) +" Hit: "+str(hit) + " Dist: " + str(dist) + " Data: " + str(self.index.get_current_count()))
        if self.stats["steps"] % self.config.log_every == 0:
            s = self.stats
            hit_rate = s["hits"] / max(1, s["steps"])

            dist_q = self._dist_quantiles_str(self.dists)

            logger.info(
                f"[AFM] Step {s['steps']} | Hit: {hit_rate:.1%} | "
                f"Size: {len(self.masks)} | Rebuilds: {s['rebuilds']} | "
                f"FB(Last): {s['fallback_last']} | "
                f"{dist_q}"
            )

        return action_mask.astype(np.int8)


def make_afm_mask_fn(afm: ActionFeasibilityMemory, get_llm_fn, llm_config, prompt_template):
    def get_action_mask(obs, last_action, action_space):
        return afm.query_or_llm(obs, last_action, action_space, get_llm_fn, llm_config, prompt_template=prompt_template)
    return get_action_mask
