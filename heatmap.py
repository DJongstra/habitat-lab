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
from matplotlib.colors import LogNorm

none_action = {
    "action":"VELOCITY_CONTROL",
    "action_args": {
        "lin_vel":0.0,
        "ang_vel":0.0,
        "allow_sliding":True
    }
}




config = get_config(config_paths='habitat_baselines/config/pointnav/ddppo_pointnav_social.yaml')
config.defrost()
config.TASK_CONFIG.TASK.MEASUREMENTS.append("SOCIAL_TOP_DOWN_MAP")
config.TASK_CONFIG.SIMULATOR.NUM_PEOPLE = 500
config.freeze()
env = habitat_baselines.common.environments.NavRLEnv(config=config)

max_steps = 1000



for i in range(8):
    data = []
    x = []
    y = []
    # Initial observations.
    observations = env.reset()
    cur_ep = env.current_episode
    start = cur_ep.start_position
    end = cur_ep.goals[0].position
    observations, _, _, info = env.step(action=none_action)
    for person in env.habitat_env.sim.people:
        x.append(person.current_position[0])
        y.append(person.current_position[2])
        data.append([person.current_position[0], person.current_position[2]])

    print(start)
    print(end)
    plt.figure()
    plt.hist2d(x,y, bins=[np.arange(-15,16,0.5),np.arange(-15,16,0.5)], norm=LogNorm())
    plt.plot([start[0], end[0]], [start[2], end[2]],marker='o')
    # Plot a colorbar with label.
    cb = plt.colorbar()
    cb.set_label('Number of entries')

    # Add title and labels to plot.
    plt.title(f"Heatmap of episode from ({start[0]:.2f},{start[2]:.2f}) to ({end[0]:.2f}, {end[2]:.2f})")
    plt.xlabel('x axis')
    plt.ylabel('z axis')
    plt.savefig("./figures/heat"+str(i))
    plt.close()

env.close()
