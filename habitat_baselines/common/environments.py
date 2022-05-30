#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

from typing import Optional, Type

import habitat
from habitat import Config, Dataset
from habitat_baselines.common.baseline_registry import baseline_registry

import numpy as np

def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="NavRLEnv")
class NavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE

        self._previous_measure = None
        self._previous_action = None
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None
        observations = super().reset()
        self._previous_measure = self._env.get_metrics()[
            self._reward_measure_name
        ]
        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD

        current_measure = self._env.get_metrics()[self._reward_measure_name]

        if (
            observations.get("num_steps", -1.0) == -1.0
            or self._rl_config.FULL_GEODESIC_DECAY == -1.0
        ):
            reward += self._previous_measure - current_measure
        else:
            reward += (
                self._previous_measure - current_measure
            ) * max(
                0,
                (
                    1 - observations["num_steps"]
                    / self._rl_config.FULL_GEODESIC_DECAY
                )
            )

        sim = self._env._sim

        if 'PROXIMITY_PENALTY' in self._rl_config:
            proximity_penalty = self._rl_config.PROXIMITY_PENALTY
            proximity_coeff = self._rl_config.PROXIMITY_COEFFICIENT
            speed_penalty = self._rl_config.get('SPEED_PENALTY', 0.0)
            agent_pos = sim.get_agent_state().position
            closest_human_zone = 0  # number to indicate in which social zone the robot is from the closest human (3 closest zone)

            if sim.social_nav and (proximity_penalty != 0 or speed_penalty != 0): # only run if penalty different from 0 specified
                for p in sim.people:
                    distance = np.sqrt(
                        (p.current_position[0]-agent_pos[0])**2
                        +(p.current_position[2]-agent_pos[2])**2
                    )
                    prox_type = self._rl_config.get('PROXIMITY_PENALTY_TYPE', "exp") # get type of penalty from configuration, either constant or exponential
                    penalty_radius = 1.0   # penalty radius variable, this is properly set in the proximity type if/else statements that follow
                    if  prox_type == "exp":
                        penalty_radius = self._rl_config.get('PENALTY_RADIUS', 3.6) # set penalty radius
                        if proximity_penalty != 0 and distance < penalty_radius:
                            reward -= pow(proximity_coeff, distance-0.2)*proximity_penalty

                    elif prox_type == "const":
                        penalty_radius = self._rl_config.get('PENALTY_RADIUS', 1.5) # set penalty radius
                        if proximity_penalty != 0 and distance < penalty_radius:
                            reward -= proximity_penalty

                    speed_penalty = self._rl_config.get('SPEED_PENALTY', 0.0)

                    if speed_penalty > 0.0 and \
                        distance < penalty_radius and \
                        "lin_speed" in observations:
                        lin_speed = abs(observations['lin_speed'])
                        speed_penalty_type = self._rl_config.get('SPEED_PENALTY_TYPE','lin')
                        if speed_penalty_type == "lin":
                            reward -= speed_penalty * (penalty_radius-distance)/penalty_radius * lin_speed
                        else:
                            print(distance)
                            if distance < 0.5 and lin_speed > 0.1:
                                if closest_human_zone < 3:
                                    closest_human_zone = 3
                            elif distance < 1.2 and lin_speed > 0.15:
                                if closest_human_zone < 2:
                                    closest_human_zone = 2
                            elif distance < 3.6 and lin_speed > 0.20:
                                if closest_human_zone < 1:
                                    closest_human_zone = 1

                reward -= speed_penalty * closest_human_zone

        self._previous_measure = current_measure

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD

        if observations.get("hit_navmesh", False):
            reward -= self._rl_config.COLLISION_PENALTY

        if observations.get("moving_backwards", False):
            reward -= self._rl_config.BACKWARDS_PENALTY

        if 'ang_accel' in observations:
            reward -= min(
                abs(observations['ang_accel'])
                * self._rl_config.ANG_ACCEL_PENALTY_COEFF,
                self._rl_config.MAX_ANG_ACCEL_PENALTY,
            )

        if 'lin_accel' in observations:
            reward -= min(
                abs(observations['lin_accel'])
                * self._rl_config.LIN_ACCEL_PENALTY_COEFF,
                self._rl_config.MAX_LIN_ACCEL_PENALTY,
            )

        if 'change_direction' in observations and observations["change_direction"]:
            reward -= self._rl_config.DIRECTION_CHANGE_PENALTY



        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()
