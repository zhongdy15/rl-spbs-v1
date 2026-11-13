import numpy as np


def action_index_to_array(action_index, dims):
    assert isinstance(action_index, int), "action_index must be an integer"
    result = []
    for dim_size in dims:
        result.append(action_index % dim_size)
        action_index //= dim_size
    return np.array(result, dtype=int)


def array_to_action_index(arr, dims):
    action_index = 0
    multiplier = 1
    for value, dim_size in zip(arr, dims):
        action_index += value * multiplier
        multiplier *= dim_size
    return action_index


def get_action_mask_fast(controllable_rooms, old_action_index, action_space):
    n_actions = action_space.n
    n_rooms = 7
    base = 4

    action_mask = np.ones(n_actions, dtype=np.int8)
    all_indices = np.arange(n_actions)

    for room_idx in range(n_rooms):
        if controllable_rooms[room_idx] != 0:
            continue
        multiplier = base ** room_idx
        old_room_action = (old_action_index // multiplier) % base
        room_actions_for_all_indices = (all_indices // multiplier) % base
        action_mask[room_actions_for_all_indices != old_room_action] = 0

    return action_mask


def build_mask_from_recommendations(
        recommendations,
        action_space_n: int = 16384,
        n_rooms: int = 7,
        base: int = 4
) -> np.ndarray:
    action_mask = np.ones(action_space_n, dtype=np.int8)
    all_indices = np.arange(action_space_n)

    for i in range(n_rooms):
        allowed_actions = recommendations.get(f"room_{i + 1}", list(range(base)))

        if len(allowed_actions) == base:
            continue

        multiplier = base ** i
        current_room_values = (all_indices // multiplier) % base
        is_valid = np.isin(current_room_values, allowed_actions)
        action_mask &= is_valid.astype(np.int8)

        if not np.any(action_mask):
            break

    return action_mask


def build_mask_from_matrix(
        valid_mask_matrix: np.ndarray,
        action_space_n: int = 16384,
        n_rooms: int = 7,
        base: int = 4
) -> np.ndarray:
    action_mask = np.ones(action_space_n, dtype=np.int8)
    all_indices = np.arange(action_space_n)

    for i in range(n_rooms):
        allowed_actions = np.where(valid_mask_matrix[i] == 1)[0]

        if len(allowed_actions) == base or len(allowed_actions) == 0:
            continue

        multiplier = base ** i
        current_room_values = (all_indices // multiplier) % base
        is_valid = np.isin(current_room_values, allowed_actions)
        action_mask &= is_valid.astype(np.int8)

        if not np.any(action_mask):
            break

    return action_mask
