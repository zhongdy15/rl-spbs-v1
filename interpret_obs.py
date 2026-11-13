import numpy as np
from typing import List, Dict


def interpret_obs(obs: np.ndarray, frame_skip: int = 5) -> str:
    if obs.shape[0] != 180:
        raise ValueError(f"输入数组 obs 的形状应为 (180,)，但实际为 {obs.shape}")

    room_state_keys = [
        "room_temp",
        "FCU_fan_feedback",
        "supply_temp",
        "return_temp",
        "occupant_num",
    ]

    observation_key_dict = {
        "sensor_outdoor": ["outdoor_temp"],
        "room1": room_state_keys,
        "room2": room_state_keys,
        "room3": room_state_keys,
        "room4": room_state_keys,
        "room5": room_state_keys,
        "room6": room_state_keys,
        "room7": room_state_keys,
    }

    single_frame_length = obs.shape[0] // frame_skip
    if single_frame_length != 36:
        raise ValueError(f"每个时间步的观测长度应为 36 (180/5)，但计算结果为 {single_frame_length}")

    frames = np.split(obs, frame_skip)

    frames.reverse()

    final_text_parts = []

    for i, frame_data in enumerate(frames):
        if i == 0:
            time_desc = "Current Status (at Time T, FCU Fan Speed was active from T-1min to T):"
        else:
            time_desc = f"Status {i} minute(s) ago (at Time T-{i}min, FCU Fan Speed was active from T-{i + 1}min to T-{i}min):"

        frame_text_parts = [time_desc]
        current_index = 0

        parsed_frame: Dict[str, Dict[str, float]] = {}

        for main_key, sub_keys in observation_key_dict.items():
            parsed_frame[main_key] = {}
            for sub_key in sub_keys:
                value = frame_data[current_index]
                parsed_frame[main_key][sub_key] = value
                current_index += 1

        for room_num in range(1, 8):
            room_key = f"room{room_num}"
            room_data = parsed_frame[room_key]

            temp = room_data["room_temp"]
            occupants = int(room_data["occupant_num"])
            fan_speed = int(room_data["FCU_fan_feedback"])

            room_str = (f"Room {room_num} — Temperature: {temp:.2f}°C, "
                        f"Occupant Num: {occupants}, "
                        f"Fan Speed: {fan_speed};")
            frame_text_parts.append(room_str)

        outdoor_temp = parsed_frame["sensor_outdoor"]["outdoor_temp"]
        outdoor_str = f"Outdoor - Temperature: {outdoor_temp:.2f}°C"
        frame_text_parts.append(outdoor_str)

        final_text_parts.append("\n".join(frame_text_parts))

    return "\n\n".join(final_text_parts)


def get_current_fan_speed_from_obs(obs: np.ndarray, frame_skip: int = 5) -> np.ndarray:
    if obs.shape[0] != 180:
        raise ValueError(f"输入数组 obs 的形状应为 (180,)，但实际为 {obs.shape}")

    single_frame_length = obs.shape[0] // frame_skip
    if single_frame_length != 36:
        raise ValueError(f"每个时间步的观测长度应为 36 (180/5)，但计算结果为 {single_frame_length}")

    latest_frame_start_index = (frame_skip - 1) * single_frame_length
    latest_frame = obs[latest_frame_start_index:]


    fan_speed_indices = [2 + (i * 5) for i in range(7)]

    fan_speeds = latest_frame[fan_speed_indices]

    return fan_speeds


def extract_current_key_features(obs: np.ndarray) -> np.ndarray:
    obs_dim = obs.ndim
    if obs_dim == 1:
        if obs.shape[0] != 180:
            raise ValueError(f"一维输入 obs 的形状应为 (180,)，但实际为 {obs.shape}")
        obs = obs[np.newaxis, :]
    elif obs_dim == 2:
        if obs.shape[1] != 180:
            raise ValueError(f"二维输入 obs 的第二个维度应为 180，但实际为 {obs.shape}")
    else:
        raise ValueError(f"输入 obs 的维度应为 1 或 2，但实际为 {obs_dim}")

    frame_size = 36
    start_index = 180 - frame_size

    latest_frame = obs[:, start_index:]


    idx_outdoor = [0]

    idx_room_temps = [1 + (i * 5) + 0 for i in range(7)]

    idx_occupants = [1 + (i * 5) + 4 for i in range(7)]

    feature_indices = idx_outdoor + idx_room_temps + idx_occupants

    feature_obs = latest_frame[:, feature_indices]

    if obs_dim == 1:
        return feature_obs.squeeze(0)
    else:
        return feature_obs
