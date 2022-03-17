import numpy as np
import random

from PIL import Image
import matplotlib.pyplot as plt
import cv2


import habitat
import habitat_baselines
from habitat_baselines.config.default import get_config
from habitat.utils.visualizations.maps import colorize_draw_agent_and_people_and_fit_to_height
from habitat_baselines.common.environments import NavRLEnv

################################################################################
# Utilities
################################################################################

def display_sample(rgb_obs):
    rgb_img = Image.fromarray(rgb_obs, mode="RGB")
    cv2.imshow("social-navigation", rgb_obs[..., ::-1])
    k = cv2.waitKey()
    if k == ord("q"):
        exit(0)

    # arr = [rgb_img]
    # titles = ['rgb']
    # plt.figure(figsize=(12 ,8))
    # for i, data in enumerate(arr):
    #     ax = plt.subplot(1, len(arr), i+1)
    #     ax.axis('off')
    #     ax.set_title(titles[i])
    #     plt.imshow(data)
    # plt.show()

################################################################################
# Demo
################################################################################

forward_action = {
    "action":"VELOCITY_CONTROL",
    "action_args": {
        "lin_vel":0.25,
        "ang_vel":0.0,
        "allow_sliding":True
    }
}

left_action = {
    "action":"VELOCITY_CONTROL",
    "action_args": {
        "lin_vel":0.0,
        "ang_vel":10.0,
        "allow_sliding":True
    }
}

right_action = {
    "action":"VELOCITY_CONTROL",
    "action_args": {
        "lin_vel":0.0,
        "ang_vel":-10.0,
        "allow_sliding":True
    }
}

none_action = {
    "action":"VELOCITY_CONTROL",
    "action_args": {
        "lin_vel":0.0,
        "ang_vel":0.0,
        "allow_sliding":True
    }
}



cv2.namedWindow("social-navigation")

config = get_config(config_paths='habitat_baselines/config/pointnav/ddppo_pointnav_social.yaml')
env = habitat_baselines.common.environments.NavRLEnv(config=config)

max_steps = 1000

for i in range(len(env.episodes)):
    # Initial observations.
    observations = env.reset()
    observations, _, _, info = env.step(action=none_action)



    count_steps = 0
    while count_steps < max_steps:
        # Create visualization.
        social_map = colorize_draw_agent_and_people_and_fit_to_height(info['social_top_down_map'], 256)
        rgb_obs = observations['rgb']
        img = np.concatenate([rgb_obs, social_map], axis=1)

        cv2.imshow("social-navigation", img[..., ::-1])
        k = cv2.waitKey()

        # Debug velocity control action.
        action = None
        if k == ord("w"):
            action = forward_action
        elif k == ord("a"):
            action = left_action
        elif k == ord("d"):
            action = right_action
        elif k == ord(" "):
            action = none_action
        elif k == ord("q"):
            exit(0)
        elif k == ord("t"):
            break
        else:
            action = none_action

        observations, _, _, info = env.step(action=action)
        reward = env.get_reward(observations)
        print("reward:  ", reward)
        count_steps += 1

        agent_state = env._env._sim.get_agent_state()
        print("position: ", agent_state.position)
        print(len(env.habitat_env.sim.people))
        for person in env.habitat_env.sim.people:
            print(person.object_id, person.current_position)

        if env.get_done(observations):
            print("Episode terminated.")
            break

env.close()
