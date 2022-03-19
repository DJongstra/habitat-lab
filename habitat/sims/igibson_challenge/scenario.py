# import collections
from typing import Union, Optional, List

# from habitat.core.dataset import Dataset
# from habitat.core.embodied_task import Action, EmbodiedTask, Measure
# from habitat.core.simulator import ActionSpaceConfiguration, Sensor, Simulator
# from habitat.core.utils import Singleton

from habitat.core.simulator import (
    AgentState,
    Config,
    DepthSensor,
    Observations,
    RGBSensor,
    SemanticSensor,
    Sensor,
    SensorSuite,
    ShortestPathPoint,
    Simulator,
    VisualObservation,
)
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.config.default import Config

from habitat.utils.geometry_utils import (
    get_heading_error,
    quat_to_rad,
)
from habitat.tasks.utils import cartesian_to_polar

import habitat_sim

import random
import quaternion
import magnum as mn
import math
import numpy as np
import json

class Scenario:
    def __init__(self, filename):
        self.objects=[]
        self.people=[]
        self.walls=[]
        self.read_scenario(filename)

    def read_scenario(self, filename):
        f = open(filename, "r")
        deserialized = json.loads(f.read())
        f.close()
        for obj in deserialized["objects"]:
            self._set_object_info(obj)

        for wall in deserialized["walls"]:
            self._set_wall_info(wall)

        for person in deserialized["people"]:
            self._set_people_info(person)


    def _set_wall_info(self, wall):
        start = np.array(wall["start"])
        end = np.array(wall["end"])
        length = np.linalg.norm(start - end)
        scale = length / 3.0
        spawnpoint = (start + end) / 2

        rotation = np.array([0.0, 0.0, 0.0])

        p1 = np.array([0.0, 1.0])
        p2 = abs(start - end)
        p2 = p2 / np.linalg.norm(p2)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        theta = math.atan2(dy, dx) * 2

        wall_info = {
            "spawnpoint": spawnpoint,
            "scale": scale,
            "rotation": theta
        }
        self.walls.append(wall_info)


    def _set_object_info(self, obj):
        position = np.array(obj["position"])
        scale = np.array(obj["scale"])
        object_info = {
            "position": position,
            "scale": scale
        }
        self.objects.append(object_info)

    def _set_people_info(self, person):
        start = person["start"]
        end = person["end"]
        lin_speed = person["lin_speed"]
        ang_speed = person["ang_speed"]
        if start == end:
            lin_speed = 0.0
            ang_speed = 0.0

        person_info = {
            "start": start,
            "end": end,
            "lin_speed": lin_speed,
            "ang_speed": ang_speed
        }

        self.people.append(person_info)



