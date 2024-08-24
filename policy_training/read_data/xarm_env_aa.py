import cv2
import random
import numpy as np
import pickle as pkl
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset
from scipy.spatial.transform import Rotation as R

from utils import history_key_map as HKM


def get_relative_action(actions, action_after_steps):
    """
    Convert absolute axis angle actions to relative axis angle actions
    Action has both position and orientation. Convert to transformation matrix, get
    relative transformation matrix, convert back to axis angle
    """

    relative_actions = []
    for i in range(len(actions)):
        # Get relative transformation matrix
        # previous pose
        pos_prev = actions[i, :3]
        ori_prev = actions[i, 3:6]
        r_prev = R.from_rotvec(ori_prev).as_matrix()
        matrix_prev = np.eye(4)
        matrix_prev[:3, :3] = r_prev
        matrix_prev[:3, 3] = pos_prev
        # current pose
        next_idx = min(i + action_after_steps, len(actions) - 1)
        pos = actions[next_idx, :3]
        ori = actions[next_idx, 3:6]
        gripper = actions[next_idx, 6:]
        r = R.from_rotvec(ori).as_matrix()
        matrix = np.eye(4)
        matrix[:3, :3] = r
        matrix[:3, 3] = pos
        # relative transformation
        matrix_rel = np.linalg.inv(matrix_prev) @ matrix
        # relative pose
        # pos_rel = matrix_rel[:3, 3]
        pos_rel = pos - pos_prev
        r_rel = R.from_matrix(matrix_rel[:3, :3]).as_rotvec()
        # # compute relative rotation
        # r_prev = R.from_rotvec(ori_prev).as_matrix()
        # r = R.from_rotvec(ori).as_matrix()
        # r_rel = np.linalg.inv(r_prev) @ r
        # r_rel = R.from_matrix(r_rel).as_rotvec()
        # # compute relative translation
        # pos_rel = pos - pos_prev
        relative_actions.append(np.concatenate([pos_rel, r_rel, gripper]))
        # next_idx = min(i + action_after_steps, len(actions) - 1)
        # curr_pose, _ = actions[i, :6], actions[i, 6:]
        # next_pose, next_gripper = actions[next_idx, :6], actions[next_idx, 6:]

    # last action
    last_action = np.zeros_like(actions[-1])
    last_action[-1] = actions[-1][-1]
    while len(relative_actions) < len(actions):
        relative_actions.append(last_action)
    return np.array(relative_actions, dtype=np.float32)


def get_absolute_action(rel_actions, base_action):
    """
    Convert relative axis angle actions to absolute axis angle actions
    """
    actions = np.zeros((len(rel_actions) + 1, rel_actions.shape[-1]))
    actions[0] = base_action
    for i in range(1, len(rel_actions) + 1):
        # if i == 0:
        #     actions.append(base_action)
        #     continue
        # Get relative transformation matrix
        # previous pose
        pos_prev = actions[i - 1, :3]
        ori_prev = actions[i - 1, 3:6]
        r_prev = R.from_rotvec(ori_prev).as_matrix()
        matrix_prev = np.eye(4)
        matrix_prev[:3, :3] = r_prev
        matrix_prev[:3, 3] = pos_prev
        # relative pose
        pos_rel = rel_actions[i - 1, :3]
        r_rel = rel_actions[i - 1, 3:6]
        # compute relative transformation matrix
        matrix_rel = np.eye(4)
        matrix_rel[:3, :3] = R.from_rotvec(r_rel).as_matrix()
        matrix_rel[:3, 3] = pos_rel
        # compute absolute transformation matrix
        matrix = matrix_prev @ matrix_rel
        # absolute pose
        pos = matrix[:3, 3]
        # r = R.from_matrix(matrix[:3, :3]).as_rotvec()
        r = R.from_matrix(matrix[:3, :3]).as_euler("xyz")
        actions[i] = np.concatenate([pos, r, rel_actions[i - 1, 6:]])
    return np.array(actions, dtype=np.float32)


def get_quaternion_orientation(cartesian):
    """
    Get quaternion orientation from axis angle representation
    """
    new_cartesian = []
    for i in range(len(cartesian)):
        pos = cartesian[i, :3]
        ori = cartesian[i, 3:]
        quat = R.from_rotvec(ori).as_quat()
        new_cartesian.append(np.concatenate([pos, quat], axis=-1))
    return np.array(new_cartesian, dtype=np.float32)


class BCDataset(IterableDataset):
    def __init__(
        self,
        path,
        tasks,
        num_demos_per_task,
        obs_type,
        history,
        history_len,
        temporal_agg,
        num_queries,
        img_size,
        action_after_steps,
        intermediate_goal_step,
        store_actions,
        pixel_keys,
        aux_keys,
        subsample,
        skip_first_n,
        relative_actions,
        random_mask_proprio,
        sensor_params,
    ):
        self._obs_type = obs_type
        self._history = history
        self._history_len = history_len if history else {"default": 1}
        self._img_size = img_size
        self._action_after_steps = action_after_steps
        self._intermediate_goal_step = intermediate_goal_step
        self._store_actions = store_actions
        self._pixel_keys = pixel_keys
        self._aux_keys = aux_keys
        self._random_mask_proprio = random_mask_proprio

        self._subtract_sensor_baseline = sensor_params.subtract_sensor_baseline
        self._use_sensor_diffs = sensor_params.use_sensor_diffs
        self._separate_sensors = sensor_params.separate_sensors

        # TODO: Currently hardcoded
        self._num_sensors = 2

        # temporal aggregation
        self._temporal_agg = temporal_agg
        self._num_queries = num_queries

        # get data paths
        self._paths = []
        self._paths.extend([Path(path) / f"{task}.pkl" for task in tasks])

        paths = {}
        idx = 0
        for path in self._paths:
            paths[idx] = path
            idx += 1
        del self._paths
        self._paths = paths

        # store actions
        if self._store_actions:
            self.actions = []

        # read data
        self._episodes = {}
        self._max_episode_len = 0
        self._max_state_dim = 0
        self._num_samples = 0
        min_stat, max_stat = None, None
        min_sensor_stat, max_sensor_stat = None, None
        min_sensor_diff_stat, max_sensor_diff_stat = None, None
        use_sensor_data = True
        min_act, max_act = None, None
        self.prob = []
        for _path_idx in self._paths:
            print(f"Loading {str(self._paths[_path_idx])}")
            # Add to prob
            if "fridge" in str(self._paths[_path_idx]):
                # self.prob.append(25.0/11.0)
                self.prob.append(22.0 / 9.0)
            else:
                self.prob.append(1)
            # read
            data = pkl.load(open(str(self._paths[_path_idx]), "rb"))
            observations = data["observations"]
            # actions = data["actions"]
            # task_emb = data["task_emb"]
            # store
            self._episodes[_path_idx] = []
            for i in range(min(num_demos_per_task, len(observations))):
                # compute actions
                # absolute actions
                actions = np.concatenate(
                    [
                        observations[i]["cartesian_states"],
                        observations[i]["gripper_states"][:, None],
                    ],
                    axis=1,
                )
                if len(actions) == 0:
                    continue
                # while len(actions) < len(observations[i]["cartesian_states"]):
                #     actions = np.concatenate([actions, actions[-1:]], axis=0)
                # skip first n
                if skip_first_n is not None:
                    for key in observations[i].keys():
                        observations[i][key] = observations[i][key][skip_first_n:]
                    actions = actions[skip_first_n:]
                # subsample
                if subsample is not None:
                    for key in observations[i].keys():
                        observations[i][key] = observations[i][key][::subsample]
                    actions = actions[::subsample]
                # action after steps
                if relative_actions:
                    actions = get_relative_action(actions, self._action_after_steps)
                    # base_action = actions[0]
                    # rel_actions = get_relative_action(actions, self._action_after_steps)
                    # reconstructed_actions = get_absolute_action(rel_actions, base_action)
                else:
                    actions = actions[self._action_after_steps :]
                # Convert cartesian states to quaternion orientation
                observations[i]["cartesian_states"] = get_quaternion_orientation(
                    observations[i]["cartesian_states"]
                )
                try:
                    sensor_baseline = np.median(
                        observations[i]["sensor_states"][:5], axis=0, keepdims=True
                    )
                    if self._subtract_sensor_baseline:
                        observations[i]["sensor_states"] = (
                            observations[i]["sensor_states"] - sensor_baseline
                        )
                        if max_sensor_stat is None:
                            max_sensor_stat = np.max(
                                observations[i]["sensor_states"], axis=0
                            )
                            min_sensor_stat = np.min(
                                observations[i]["sensor_states"], axis=0
                            )
                        else:
                            max_sensor_stat = np.maximum(
                                max_sensor_stat,
                                np.max(observations[i]["sensor_states"], axis=0),
                            )
                            min_sensor_stat = np.minimum(
                                min_sensor_stat,
                                np.min(observations[i]["sensor_states"], axis=0),
                            )
                    if self._separate_sensors:
                        for sensor_idx in range(self._num_sensors):
                            observations[i][
                                f"sensor{sensor_idx}_states"
                            ] = observations[i]["sensor_states"][
                                ..., sensor_idx * 15 : (sensor_idx + 1) * 15
                            ]
                    if self._use_sensor_diffs:
                        sensor_diffs = np.diff(observations[i]["sensor_states"], axis=0)
                        sensor_diffs = np.concatenate(
                            [np.zeros_like(sensor_diffs[:1]), sensor_diffs], axis=0
                        )
                        observations[i]["sensor_states"] = np.concatenate(
                            [observations[i]["sensor_states"], sensor_diffs], axis=-1
                        )
                        if max_sensor_diff_stat is None:
                            max_sensor_diff_stat = np.max(sensor_diffs, axis=0)
                            min_sensor_diff_stat = np.min(sensor_diffs, axis=0)
                        else:
                            max_sensor_diff_stat = np.maximum(
                                max_sensor_diff_stat,
                                np.max(sensor_diffs, axis=0),
                            )
                            min_sensor_diff_stat = np.minimum(
                                min_sensor_diff_stat,
                                np.min(sensor_diffs, axis=0),
                            )
                        if self._separate_sensors:
                            for sensor_idx in range(self._num_sensors):
                                observations[i][
                                    f"sensor{sensor_idx}_states"
                                ] = np.concatenate(
                                    [
                                        observations[i][f"sensor{sensor_idx}_states"],
                                        sensor_diffs[
                                            ..., sensor_idx * 15 : (sensor_idx + 1) * 15
                                        ],
                                    ],
                                    axis=-1,
                                )
                except KeyError:
                    print("WARN: Sensor data not found.")
                    use_sensor_data = False

                # Repeat first dimension of each observation for history_len times
                for key in observations[i].keys():
                    try:
                        history_len = self._history_len[HKM(key)]
                    except KeyError:
                        self._history_len[HKM(key)] = self._history_len["default"]
                        history_len = self._history_len[HKM(key)]
                    observations[i][key] = np.concatenate(
                        [
                            [observations[i][key][0]] * history_len,
                            observations[i][key],
                        ],
                        axis=0,
                    )
                try:
                    history_len = self._history_len["action"]
                except KeyError:
                    self._history_len["action"] = self._history_len["default"]
                    history_len = self._history_len["action"]
                # Repeat first action for history_len times
                remaining_actions = actions[0]
                if relative_actions:
                    pos = remaining_actions[:-1]
                    ori_gripper = remaining_actions[-1:]
                    remaining_actions = np.concatenate(
                        [np.zeros_like(pos), ori_gripper]
                    )
                actions = np.concatenate(
                    [
                        [remaining_actions] * history_len,
                        actions,
                    ],
                    axis=0,
                )
                # # Save images
                # Path("images").mkdir(exist_ok=True)
                # for idx, obs in enumerate(observations[i][self._pixel_keys[1]]):
                #     cv2.imwrite(f"images/{idx}.png", obs)
                # import ipdb; ipdb.set_trace()

                # store
                episode = dict(
                    observation=observations[i],
                    action=actions,
                    # task_emb=task_emb,
                )
                self._episodes[_path_idx].append(episode)
                self._max_episode_len = max(
                    self._max_episode_len,
                    (
                        len(observations[i])
                        if not isinstance(observations[i], dict)
                        else len(observations[i][self._pixel_keys[0]])
                    ),
                )
                # if obs_type == 'features':
                self._max_state_dim = 7  # 8  # max(
                #     self._max_state_dim, data["states"][i].shape[-1]
                # )
                self._num_samples += (
                    len(observations[i])
                    if self._obs_type == "features"
                    else len(observations[i][self._pixel_keys[0]])
                )

                # max, min action
                if min_act is None:
                    min_act = np.min(actions, axis=0)
                    max_act = np.max(actions, axis=0)
                else:
                    min_act = np.minimum(min_act, np.min(actions, axis=0))
                    max_act = np.maximum(max_act, np.max(actions, axis=0))

                # store actions
                if self._store_actions:
                    self.actions.append(actions)

            # keep record of max and min stat
            max_cartesian = data["max_cartesian"]
            min_cartesian = data["min_cartesian"]
            max_cartesian = np.concatenate(
                [data["max_cartesian"][:3], [1] * 4]
            )  # for quaternion
            min_cartesian = np.concatenate(
                [data["min_cartesian"][:3], [-1] * 4]
            )  # for quaternion
            max_gripper = data["max_gripper"]
            min_gripper = data["min_gripper"]
            max_val = np.concatenate([max_cartesian, max_gripper[None]], axis=0)
            min_val = np.concatenate([min_cartesian, min_gripper[None]], axis=0)
            if max_stat is None:
                max_stat = max_val
                min_stat = min_val
            else:
                max_stat = np.maximum(max_stat, max_val)
                min_stat = np.minimum(min_stat, min_val)
            if use_sensor_data:
                # If baseline is subtracted, use zero as shift and max as scale
                if self._subtract_sensor_baseline:
                    max_sensor_stat = np.maximum(
                        np.abs(max_sensor_stat), np.abs(min_sensor_stat)
                    )
                    min_sensor_stat = np.zeros_like(max_sensor_stat)
                # If baseline isn't subtracted, use usual min and max values
                else:
                    if max_sensor_stat is None:
                        max_sensor_stat = data["max_sensor"]
                        min_sensor_stat = data["min_sensor"]
                    else:
                        max_sensor_stat = np.maximum(
                            max_sensor_stat, data["max_sensor"]
                        )
                        min_sensor_stat = np.minimum(
                            min_sensor_stat, data["min_sensor"]
                        )
                if self._use_sensor_diffs:
                    max_sensor_diff_stat = np.maximum(
                        np.abs(max_sensor_diff_stat), np.abs(min_sensor_diff_stat)
                    )
                    min_sensor_diff_stat = np.zeros_like(max_sensor_diff_stat)
                    max_sensor_stat = np.concatenate(
                        (max_sensor_stat, max_sensor_diff_stat)
                    )
                    min_sensor_stat = np.concatenate(
                        (min_sensor_stat, min_sensor_diff_stat)
                    )

        # # setting max and min at 2/3rd and 1/3rd of the way between min and max
        # min_stat = min_stat + (max_stat - min_stat) / 3
        # max_stat = min_stat + (max_stat - min_stat) * 2 / 3

        # min_stat[3:9], max_stat[3:9] = 0, 1  #################################
        min_act[3:6], max_act[3:6] = 0, 1  #################################
        self.stats = {
            "actions": {
                "min": min_act,  # min_stat,
                "max": max_act,  # max_stat,
            },
            "proprioceptive": {
                "min": min_stat,
                "max": max_stat,
            },
        }
        if self._separate_sensors:
            for sensor_idx in range(self._num_sensors):
                sensor_mask = np.zeros_like(min_sensor_stat, dtype=bool)
                sensor_mask[sensor_idx * 15 : (sensor_idx + 1) * 15] = True
                if self._use_sensor_diffs:
                    sensor_mask[
                        (self._num_sensors + sensor_idx)
                        * 15 : (self._num_sensors + sensor_idx + 1)
                        * 15
                    ] = True
                self.stats[f"sensor{sensor_idx}"] = {
                    "min": min_sensor_stat[sensor_mask],
                    "max": max_sensor_stat[sensor_mask],
                }
        else:
            self.stats["sensor"] = {
                "min": min_sensor_stat,
                "max": max_sensor_stat,
            }

        # augmentation
        self.aug = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop(self._img_size, padding=4),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.ToTensor(),
            ]
        )

        # Samples from envs
        self.envs_till_idx = len(self._episodes)

        self.prob = np.array(self.prob) / np.sum(self.prob)

    def preprocess(self, key, x):
        return (x - self.stats[key]["min"]) / (
            self.stats[key]["max"] - self.stats[key]["min"] + 1e-5
        )

    def _sample_episode(self, env_idx=None):
        # idx = random.choice(list(self._episodes.keys())) if env_idx is None else env_idx
        idx = random.randint(0, self.envs_till_idx - 1) if env_idx is None else env_idx

        # sample idx with probability
        idx = np.random.choice(list(self._episodes.keys()), p=self.prob)

        episode = random.choice(self._episodes[idx])
        return (episode, idx) if env_idx is None else episode

    def _sample(self):
        episodes, env_idx = self._sample_episode()
        observations = episodes["observation"]
        actions = episodes["action"]
        # task_emb = episodes["task_emb"]
        sample_idx = np.random.randint(
            1, len(observations[self._pixel_keys[0]]) - self._history_len["pixels"]
        )
        # sample_idx = np.random.randint(
        #     self._history_len, len(observations[self._pixel_keys[0]])
        # )
        if self._obs_type == "pixels":
            # Sample obs, action
            sampled_pixel = {}
            for key in self._pixel_keys:
                sampled_pixel[key] = observations[key][
                    -(sample_idx + self._history_len[HKM(key)]) : -sample_idx
                ]
                # sampled_pixel[key] = observations[key][
                #     sample_idx : sample_idx + self._history_len
                # ]
                sampled_pixel[key] = torch.stack(
                    [
                        self.aug(sampled_pixel[key][i])
                        for i in range(len(sampled_pixel[key]))
                    ]
                )
            # sampled_pixel = observations["pixels"][
            #     sample_idx : sample_idx + self._history_len
            # ]
            # sampled_pixel = torch.stack(
            #     [self.aug(sampled_pixel[i]) for i in range(len(sampled_pixel))]
            # )
            sampled_state = {}
            # sampled_state["proprioceptive"] = np.concatenate(
            #     [
            #         observations["cartesian_states"][
            #             sample_idx : sample_idx + self._history_len
            #         ],
            #         observations["gripper_states"][
            #             sample_idx : sample_idx + self._history_len
            #         ][:, None],
            #     ],
            #     axis=1,
            # )
            sampled_state["proprioceptive"] = np.concatenate(
                [
                    observations["cartesian_states"][
                        -(
                            sample_idx + self._history_len["proprioceptive"]
                        ) : -sample_idx
                    ],
                    observations["gripper_states"][
                        -(
                            sample_idx + self._history_len["proprioceptive"]
                        ) : -sample_idx
                    ][:, None],
                ],
                axis=1,
            )

            if self._random_mask_proprio and np.random.rand() < 0.5:
                sampled_state["proprioceptive"] = (
                    np.ones_like(sampled_state["proprioceptive"])
                    * self.stats["proprioceptive"]["min"]
                )
            try:
                if self._separate_sensors:
                    for sensor_idx in range(self._num_sensors):
                        skey = f"sensor{sensor_idx}"
                        sampled_state[f"{skey}"] = observations[f"{skey}_states"][
                            -(sample_idx + self._history_len["sensor"]) : -sample_idx
                        ]
                else:
                    sampled_state["sensor"] = observations["sensor_states"][
                        -(sample_idx + self._history_len["sensor"]) : -sample_idx
                    ]
            except KeyError:
                pass

            act_history_len = self._history_len["action"]
            if self._temporal_agg:
                # arrange sampled action to be of shape (act_history_len, num_queries, action_dim)
                sampled_action = np.zeros(
                    (act_history_len, self._num_queries, actions.shape[-1])
                )
                num_actions = (
                    act_history_len + self._num_queries - 1
                )  # -1 since its num_queries including the last action of the history
                act = np.zeros((num_actions, actions.shape[-1]))
                # TODO: Fix different history modes.
                if num_actions - sample_idx < 0:
                    act[:num_actions] = actions[
                        -(sample_idx) : -sample_idx + num_actions
                    ]
                    # act[
                    #     : min(len(actions), sample_idx + num_actions) - sample_idx
                    # ] = actions[- (sample_idx) : -sample_idx + num_actions]
                else:
                    act[:sample_idx] = actions[-sample_idx:]
                    act[sample_idx:] = actions[-1]
                # print(sample_idx, num_actions, len(actions))
                # act[:num_actions] = actions[- sample_idx : -sample_idx + num_actions]
                # if len(actions) < sample_idx + num_actions:
                #     act[len(actions) - sample_idx :] = actions[
                #         -1
                #     ]  # act[-1] # TODO: Bug here - should be actions[-1]?
                sampled_action = np.lib.stride_tricks.sliding_window_view(
                    act, (self._num_queries, actions.shape[-1])
                )
                sampled_action = sampled_action[:, 0]
            else:
                sampled_action = actions[-(sample_idx + act_history_len) : -sample_idx]

            return_dict = {}
            for key in self._pixel_keys:
                return_dict[key] = sampled_pixel[key]
            for key in self._aux_keys:
                return_dict[key] = self.preprocess(key, sampled_state[key])
            # return_dict["proprioceptive"] = self.preprocess("proprioceptive",
            #     sampled_proprioceptive_state
            # )
            return_dict["actions"] = self.preprocess("actions", sampled_action)
            # return_dict["task_emb"] = task_emb
            return return_dict
            # prompt
            # if self._prompt == None or self._prompt == "text":
            #     # return {
            #     #     "pixels": sampled_pixel,
            #     #     "actions": self.preprocess("actions", sampled_action),
            #     #     "task_emb": task_emb,
            #     # }
            #     return return_dict
            # elif self._prompt == "goal":
            #     prompt_episode = self._sample_episode(env_idx)
            #     prompt_observations = prompt_episode["observation"]
            #     for pixel_key in self._pixel_keys:
            #         prompt_pixel = self.aug(prompt_observations[pixel_key][-1])[None]
            #         return_dict["prompt_" + pixel_key] = prompt_pixel
            #     prompt_action = prompt_episode["action"][-1:]
            #     return_dict["prompt_actions"] = self.preprocess("actions",
            #         prompt_action
            #     )
            #     # prompt_pixel = self.aug(prompt_observations["pixels"][-1])[None]
            #     # prompt_action = prompt_episode["action"][-1:]
            #     # return {
            #     #     "pixels": sampled_pixel,
            #     #     "actions": self.preprocess("actions", sampled_action),
            #     #     "prompt_pixels": prompt_pixel,
            #     #     "prompt_actions": self.preprocess("actions", prompt_action),
            #     #     "task_emb": task_emb,
            #     # }
            #     return return_dict
            # elif self._prompt == "intermediate_goal":
            #     # prompt_episode = self._sample_episode(env_idx)
            #     prompt_episode = episodes
            #     prompt_observations = prompt_episode["observation"]
            #     # goal_idx  = min(sample_idx + self.intermediate_goal_step, len(prompt_observations['pixels'])-1)
            #     intermediate_goal_step = (
            #         self._intermediate_goal_step + np.random.randint(-30, 30)
            #     )
            #     goal_idx = min(
            #         sample_idx + intermediate_goal_step,
            #         len(prompt_observations["pixels"]) - 1,
            #     )
            #     prompt_pixel = self.aug(
            #         prompt_observations[self._pixel_keys[0]][goal_idx]
            #     )[None]
            #     prompt_action = prompt_episode["action"][goal_idx : goal_idx + 1]

            #     return_dict["prompt_" + self._pixel_keys[0]] = prompt_pixel
            #     return_dict["prompt_actions"] = self.preprocess("actions",
            #         prompt_action
            #     )
            #     # return {
            #     #     "pixels": sampled_pixel,
            #     #     "actions": self.preprocess("actions", sampled_action),
            #     #     "prompt_pixels": prompt_pixel,
            #     #     "prompt_actions": self.preprocess("actions", prompt_action),
            #     #     "task_emb": task_emb,
            #     # }

        elif self._obs_type == "features":
            sampled_proprioceptive_state = np.concatenate(
                [
                    observations["cartesian_states"][
                        sample_idx : sample_idx + self._history_len
                    ],
                    observations["gripper_states"][
                        sample_idx : sample_idx + self._history_len
                    ][:, None],
                ],
                axis=1,
            )
            if self._temporal_agg:
                # arrange sampled action to be of shape (history_len, num_queries, action_dim)
                sampled_action = np.zeros(
                    (self._history_len, self._num_queries, actions.shape[-1])
                )
                num_actions = (
                    self._history_len + self._num_queries - 1
                )  # -1 since its num_queries including the last action of the history
                act = np.zeros((num_actions, actions.shape[-1]))
                act[
                    : min(len(actions), sample_idx + num_actions) - sample_idx
                ] = actions[sample_idx : sample_idx + num_actions]
                if len(actions) < sample_idx + num_actions:
                    act[len(actions) - sample_idx :] = actions[
                        -1
                    ]  # act[-1] # TODO: Bug here - should be actions[-1]?
                sampled_action = np.lib.stride_tricks.sliding_window_view(
                    act, (self._num_queries, actions.shape[-1])
                )
                sampled_action = sampled_action[:, 0]
            else:
                sampled_action = actions[sample_idx : sample_idx + self._history_len]

            return_dict = {}
            return_dict["proprioceptive"] = self.preprocess(
                "proprioceptive", sampled_proprioceptive_state
            )
            # return_dict["unnorm_proprioceptive"] = sampled_proprioceptive_state
            return_dict["actions"] = self.preprocess("actions", sampled_action)
            # return_dict["og_actions"] = sampled_action
            return return_dict

    def sample_test(self, env_idx, step=None):
        episode = self._sample_episode(env_idx)
        observations = episode["observation"]
        # actions = episode["action"]
        # task_emb = episode["task_emb"]

        if self._obs_type == "pixels":
            # pixels_shape = observations[self._pixel_keys[0]].shape

            # observation
            if self._prompt == None or self._prompt == "text":
                prompt_pixel = None
                prompt_action = None
            elif self._prompt == "goal":
                prompt_pixel = np.transpose(
                    observations[self._pixel_keys[0]][-1:], (0, 3, 1, 2)
                )
                prompt_action = None
            elif self._prompt == "intermediate_goal":
                goal_idx = min(
                    step + self._intermediate_goal_step, len(observations["pixels"]) - 1
                )
                prompt_pixel = np.transpose(
                    observations[self._pixel_keys[0]][goal_idx : goal_idx + 1],
                    (0, 3, 1, 2),
                )
                prompt_action = None

            return {
                "prompt_pixels": prompt_pixel,
                "prompt_actions": (
                    self.preprocess("actions", prompt_action)
                    if prompt_action is not None
                    else None
                ),
                # "task_emb": task_emb,
            }

        elif self._obs_type == "features":
            raise NotImplementedError

    def __iter__(self):
        while True:
            yield self._sample()

    def __len__(self):
        return self._num_samples
