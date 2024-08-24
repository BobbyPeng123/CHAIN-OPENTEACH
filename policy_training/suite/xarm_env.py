from collections import deque
from typing import Any, NamedTuple

import gym
from gym import Wrapper, spaces

# from gym.wrappers import FrameStack

import xarm_env
import dm_env
import numpy as np
from dm_env import StepType, specs, TimeStep

import cv2

# from libero.libero import benchmark, get_libero_path
# from libero.libero.envs import OffScreenRenderEnv


class RGBArrayAsObservationWrapper(dm_env.Environment):
    """
    Use env.render(rgb_array) as observation
    rather than the observation environment provides

    From: https://github.com/hill-a/stable-baselines/issues/915
    """

    def __init__(
        self,
        env,
        # height,
        # width,
        max_episode_len=300,
        max_state_dim=100,
        task_description="",
        pixel_keys=["pixels0"],
        aux_keys=["proprioceptive"],
        use_robot=True,
    ):
        self._env = env
        # self._width = width
        # self._height = height
        self._max_episode_len = max_episode_len
        self._max_state_dim = max_state_dim
        self._task_description = task_description
        self.pixel_keys = pixel_keys
        self.aux_keys = aux_keys
        self.use_robot = use_robot

        # task emb

        obs = self._env.reset()
        if self.use_robot:
            pixels = obs[pixel_keys[0]]
            self.observation_space = spaces.Box(
                low=0, high=255, shape=pixels.shape, dtype=pixels.dtype
            )

            # Action spec
            action_spec = self._env.action_space
            # self._action_spec = specs.BoundedArray(
            #     action_spec[0].shape, np.float32, action_spec[0], action_spec[1], "action"
            # )
            self._action_spec = specs.Array(
                shape=action_spec.shape, dtype=action_spec.dtype, name="action"
            )
            # Observation spec
            # robot_state = np.concatenate(
            #     [obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]]
            # )
            features = obs["features"]
            self._obs_spec = {}
            for key in pixel_keys:
                self._obs_spec[key] = specs.BoundedArray(
                    shape=obs[key].shape,
                    dtype=np.uint8,
                    minimum=0,
                    maximum=255,
                    name=key,
                )
        else:
            pixels, features = obs["pixels"], obs["features"]
            self.observation_space = spaces.Box(
                low=0, high=255, shape=pixels.shape, dtype=pixels.dtype
            )

            # Action spec
            action_spec = self._env.action_space
            self._action_spec = specs.Array(
                shape=action_spec.shape, dtype=action_spec.dtype, name="action"
            )

            # Observation spec
            self._obs_spec = {}
            for key in pixel_keys:
                self._obs_spec[key] = specs.BoundedArray(
                    shape=pixels.shape,
                    dtype=np.uint8,
                    minimum=0,
                    maximum=255,
                    name=key,
                )

        self._obs_spec["proprioceptive"] = specs.BoundedArray(
            shape=features.shape,
            dtype=np.float32,
            minimum=-np.inf,
            maximum=np.inf,
            name="proprioceptive",
        )
        self._obs_spec["features"] = specs.BoundedArray(
            shape=(self._max_state_dim,),
            dtype=np.float32,
            minimum=-np.inf,
            maximum=np.inf,
            name="features",
        )

        for key in aux_keys:
            if key.startswith("sensor"):
                self._obs_spec[key] = specs.BoundedArray(
                    shape=obs[key].shape,
                    dtype=np.float32,
                    minimum=-np.inf,
                    maximum=np.inf,
                    name=key,
                )
        # if "sensor" in aux_keys:
        #     self._obs_spec["sensor"] = specs.BoundedArray(
        #         shape=obs["sensor"].shape,
        #         dtype=np.float32,
        #         minimum=-np.inf,
        #         maximum=np.inf,
        #         name="sensor",
        #     )

        self.render_image = None

    def reset(self, **kwargs):
        self._step = 0
        obs = self._env.reset(**kwargs)

        observation = {}
        for key in self.pixel_keys:
            observation[key] = obs[key]
        observation["proprioceptive"] = obs["features"]
        observation["features"] = obs["features"]
        # if "sensor" in self.aux_keys:
        #     observation["sensor"] = obs["sensor"]
        for key in self.aux_keys:
            if key.startswith("sensor"):
                observation[key] = obs[key]

        # observation["pixels"] = obs["agentview_image"][::-1, :]
        # observation["pixels_egocentric"] = obs["robot0_eye_in_hand_image"][::-1, :]
        # observation["proprioceptive"] = np.concatenate(
        #     [obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]]
        # )
        # # get state
        # observation["features"] = np.zeros(self._max_state_dim)
        # state = self._env.get_sim_state()  # TODO: Change to robot state
        # observation["features"][: state.shape[0]] = state
        observation["goal_achieved"] = False
        return observation

    def step(self, action):
        self._step += 1
        obs, reward, truncated, terminated, info = self._env.step(action)
        done = truncated or terminated
        # self.render_image = obs["agentview_image"][::-1, :]

        observation = {}
        for key in self.pixel_keys:
            observation[key] = obs[key]
        observation["proprioceptive"] = obs["features"]
        observation["features"] = obs["features"]
        if "sensor" in self.aux_keys:
            observation["sensor"] = obs["sensor"]
        for key in self.aux_keys:
            if key.startswith("sensor"):
                observation[key] = obs[key]
        # observation["pixels"] = obs["agentview_image"][::-1, :]
        # observation["pixels_egocentric"] = obs["robot0_eye_in_hand_image"][::-1, :]
        # observation["proprioceptive"] = np.concatenate(
        #     [obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]]
        # )
        # # get state
        # observation["features"] = np.zeros(self._max_state_dim)
        # state = self._env.get_sim_state()  # TODO: Change to robot state
        # observation["features"][: state.shape[0]] = state
        observation["goal_achieved"] = done  # (self._step == self._max_episode_len)
        return observation, reward, done, info

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def render(self, mode="rgb_array", width=256, height=256):
        # return cv2.resize(self.render_image, (width, height))
        return cv2.resize(self._env.render("rgb_array"), (width, height))

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames):
        self._env = env
        self._num_frames = num_frames

        self.pixel_keys = [
            keys for keys in env.observation_spec().keys() if "pixels" in keys
        ]
        wrapped_obs_spec = env.observation_spec()[self.pixel_keys[0]]

        # frames lists
        self._frames = {}
        for key in self.pixel_keys:
            self._frames[key] = deque([], maxlen=num_frames)
        # self._frames = deque([], maxlen=num_frames)
        # self._frames_egocentric = deque([], maxlen=num_frames)

        pixels_shape = wrapped_obs_spec.shape
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = {}
        self._obs_spec["features"] = self._env.observation_spec()["features"]
        self._obs_spec["proprioceptive"] = self._env.observation_spec()[
            "proprioceptive"
        ]
        for key in self._env.observation_spec().keys():
            if key.startswith("sensor"):
                self._obs_spec[key] = self._env.observation_spec()[key]
        # if "sensor" in self._env.observation_spec().keys():
        #     self._obs_spec["sensor"] = self._env.observation_spec()["sensor"]
        for key in self.pixel_keys:
            self._obs_spec[key] = specs.BoundedArray(
                shape=np.concatenate(
                    [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0
                ),
                dtype=np.uint8,
                minimum=0,
                maximum=255,
                name=key,
            )
        # self._obs_spec["pixels"] = specs.BoundedArray(
        #     shape=np.concatenate(
        #         [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0
        #     ),
        #     dtype=np.uint8,
        #     minimum=0,
        #     maximum=255,
        #     name="pixels",
        # )
        # self._obs_spec["pixels_egocentric"] = specs.BoundedArray(
        #     shape=np.concatenate(
        #         [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0
        #     ),
        #     dtype=np.uint8,
        #     minimum=0,
        #     maximum=255,
        #     name="pixels_egocentric",
        # )

    def _transform_observation(self, time_step):
        for key in self.pixel_keys:
            assert len(self._frames[key]) == self._num_frames
        # assert len(self._frames) == self._num_frames
        # assert len(self._frames_egocentric) == self._num_frames
        obs = {}
        obs["features"] = time_step.observation["features"]
        for key in self.pixel_keys:
            obs[key] = np.concatenate(list(self._frames[key]), axis=0)
        # obs["pixels"] = np.concatenate(list(self._frames), axis=0)
        # obs["pixels_egocentric"] = np.concatenate(list(self._frames_egocentric), axis=0)
        obs["proprioceptive"] = time_step.observation["proprioceptive"]
        try:
            for key in time_step.observation.keys():
                if key.startswith("sensor"):
                    obs[key] = time_step.observation[key]
            # obs["sensor"] = time_step.observation["sensor"]
        except KeyError:
            pass
        obs["goal_achieved"] = time_step.observation["goal_achieved"]
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = {}
        for key in self.pixel_keys:
            pixels[key] = time_step.observation[key]
            if len(pixels[key].shape) == 4:
                pixels[key] = pixels[key][0]
            pixels[key] = pixels[key].transpose(2, 0, 1)
        return pixels
        # return [pixels[key].transpose(2, 0, 1).copy() for key in self.pixel_keys]

        # # pixels = time_step.observation["pixels"] ixels_egocentric"]

        # # remove batch dim
        # if len(pixels.shape) == 4:
        #     pixels = pixels[0]
        # if len(pixels_egocentric.shape) == 4:
        #     pixels_egocentric = pixels_egocentric[0]
        # return (
        #     pixels.transpose(2, 0, 1).copy(),
        #     pixels_egocentric.transpose(2, 0, 1).copy(),
        # )

    def reset(self, **kwargs):
        time_step = self._env.reset(**kwargs)
        # pixels, pixels_egocentric = self._extract_pixels(time_step)
        pixels = self._extract_pixels(time_step)
        for key in self.pixel_keys:
            for _ in range(self._num_frames):
                self._frames[key].append(pixels[key])
        # for _ in range(self._num_frames):
        #     self._frames.append(pixels)
        #     self._frames_egocentric.append(pixels_egocentric)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        # pixels, pixels_egocentric = self._extract_pixels(time_step)
        for key in self.pixel_keys:
            self._frames[key].append(pixels[key])
        # self._frames.append(pixels)
        # self._frames_egocentric.append(pixels_egocentric)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        self._discount = 1.0

        # Action spec
        wrapped_action_spec = env.action_spec()
        # self._action_spec = specs.BoundedArray(
        #     wrapped_action_spec.shape,
        #     np.float32,
        #     wrapped_action_spec.minimum,
        #     wrapped_action_spec.maximum,
        #     "action",
        # )
        self._action_spec = specs.Array(
            shape=wrapped_action_spec.shape, dtype=dtype, name="action"
        )

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        # Make time step for action space
        observation, reward, done, info = self._env.step(action)
        # step_type = StepType.LAST if observation['goal_achieved'] else StepType.MID
        step_type = StepType.LAST if done else StepType.MID
        # step_type = (
        #     StepType.LAST
        #     if (
        #         self._env._step == self._env._max_episode_len
        #         or observation["goal_achieved"]
        #     )
        #     else StepType.MID
        # )

        return TimeStep(
            step_type=step_type,
            reward=reward,
            discount=self._discount,
            observation=observation,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec
        # return self._env.action_spec()

    def reset(self, **kwargs):
        obs = self._env.reset(**kwargs)
        return TimeStep(
            step_type=StepType.FIRST, reward=0, discount=self._discount, observation=obs
        )

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self, **kwargs):
        time_step = self._env.reset(**kwargs)
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward or 0.0,
            discount=time_step.discount or 1.0,
        )

    def _replace(
        self, time_step, observation=None, action=None, reward=None, discount=None
    ):
        if observation is None:
            observation = time_step.observation
        if action is None:
            action = time_step.action
        if reward is None:
            reward = time_step.reward
        if discount is None:
            discount = time_step.discount
        return ExtendedTimeStep(
            observation=observation,
            step_type=time_step.step_type,
            action=action,
            reward=reward,
            discount=discount,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make(
    # suite,
    # scenes,
    # tasks,
    frame_stack,
    action_repeat,
    seed,
    height,
    width,
    max_episode_len,
    max_state_dim,
    use_egocentric,
    use_fisheye,
    task_description,
    pixel_keys,
    aux_keys,
    sensor_params,
    eval,  # True means use_robot=True
):
    env = gym.make(
        "Robot-v1",
        height=height,
        width=width,
        use_robot=eval,
        use_egocentric=use_egocentric,
        use_fisheye=use_fisheye,
        subtract_sensor_baseline=sensor_params.subtract_sensor_baseline,
        use_sensor_diffs=sensor_params.use_sensor_diffs,
        separate_sensors=sensor_params.separate_sensors,
    )
    # env.seed(seed)

    # apply wrappers
    env = RGBArrayAsObservationWrapper(
        env,
        # height=height,
        # width=width,
        max_episode_len=max_episode_len,
        max_state_dim=max_state_dim,
        task_description=task_description,
        pixel_keys=pixel_keys,
        aux_keys=aux_keys,
        use_robot=eval,
    )
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = FrameStackWrapper(env, frame_stack)
    env = ExtendedTimeStepWrapper(env)

    return [env], [task_description]
