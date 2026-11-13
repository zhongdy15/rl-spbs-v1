import os
import time

algo_list = ["dqn_mask"]
seed_list = [0,10,20,30,40]
gpu_list = [0]
reward_mode = "Baseline_OCC_PPD_with_energy"
tradeoff_constant = 10

env_list = ["SemiPhysBuildingSim-v0"]
time_flag = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

reward_mode_list = ["Baseline_without_energy",
                    "Baseline_with_energy",
                    "Baseline_OCC_PPD_without_energy",
                    "Baseline_OCC_PPD_with_energy",
                    ]


env_kwargs = f"reward_mode:\"'{reward_mode}'\" tradeoff_constant:{tradeoff_constant}"


algo_num = len(algo_list)

assert len(gpu_list) == algo_num, "gpu_setting is incorrect"

for seed in seed_list:
    for algo_id in range(algo_num):
        algo_filename = algo_list[algo_id] + "_" + reward_mode + \
                        "_" + str(tradeoff_constant) + \
                        "_" + time_flag
        cmd_line = f"CUDA_VISIBLE_DEVICES={gpu_list[algo_id]} " \
                   f"python train.py " \
                   f" --algo {algo_list[algo_id]} " \
                   f" --env {env_list[0]} " \
                   f" --env-kwargs {env_kwargs} " \
                   f" --seed {seed} " \
                   f" -f logs/{algo_filename}/ " \
                   f" -tb logs/{algo_filename}/tb/ &"
        print(cmd_line)
        os.system(cmd_line)
        time.sleep(5)
