from typing import Any, Dict

import numpy as np

from habitat.config import Config
from habitat.core.dataset import Episode
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.tasks.nav.nav import NavigationTask, TopDownMap
from habitat.utils.visualizations import maps
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat_sim.utils.common import quat_from_magnum, angle_between_quats

@registry.register_task(name="SocialNav-v0")
class SocialNavigationTask(NavigationTask):
    def reset(self, episode: Episode):
        self._sim.reset_people()
        episode.people_paths = [
            p.waypoints
            for p in self._sim.people
        ]
        observations = super().reset(episode)
        return observations

    def step(self, action: Dict[str, Any], episode: Episode):
        if "action_args" not in action or action["action_args"] is None:
            action["action_args"] = {}
        action_name = action["action"]
        if isinstance(action_name, (int, np.integer)):
            action_name = self.get_action_name(action_name)
        assert (
            action_name in self.actions
        ), f"Can't find '{action_name}' action in {self.actions.keys()}."

        task_action = self.actions[action_name]


        # Move people
        for p in self._sim.people:
            p.step()

        observations = task_action.step(**action["action_args"], task=self)
        observations.update(
            self.sensor_suite.get_observations(
                observations=observations,
                episode=episode,
                action=action,
                task=self,
            )
        )

        self._is_episode_active = self._check_episode_is_active(
            observations=observations, action=action, episode=episode
        )

        return observations

# # '_config', '_sim', '_dataset', 'measurements', 'sensor_suite',
# # 'actions', '_action_keys', 'is_stop_called'


@registry.register_measure
class SocialTopDownMap(TopDownMap):

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "social_top_down_map"

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        super().reset_metric(episode, *args, **kwargs)
        # Draw the paths of the people
        for person_path in episode.people_paths:
            map_corners = [
                maps.to_grid(
                    p[2],
                    p[0],
                    self._top_down_map.shape[0:2],
                    sim=self._sim,
                )
                for p in person_path
            ]
            maps.draw_path(
                self._top_down_map,
                map_corners,
                [255, 165, 0], # Orange
                self.line_thickness,
            )

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        self._step_count += 1
        house_map, map_agent_x, map_agent_y = self.update_map(
            self._sim.get_agent_state().position
        )

        people_map_coords = [
            maps.to_grid(
                p.current_position[2],
                p.current_position[0],
                self._top_down_map.shape[0:2],
                sim=self._sim,
            )
            for p in self._sim.people
        ]

        self._metric = {
            "map": house_map,
            "fog_of_war_mask": self._fog_of_war_mask,
            "agent_map_coord": (map_agent_x, map_agent_y),
            "people_map_coord": people_map_coords,
            "agent_angle": self.get_polar_angle(),
        }


@registry.register_measure
class HumanCollision(Measure):


    def __init__(
        self, sim: "HabitatSim", config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self.uuid = "human_collision"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "human_collision"

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._metric = False

    def update_metric(
        self,
        episode,
        task: EmbodiedTask,
        *args: Any,
        **kwargs: Any
    ):
        agent_pos = self._sim.get_agent_state().position
        for p in self._sim.people:
            distance = np.sqrt(
                (p.current_position[0]-agent_pos[0])**2
                +(p.current_position[2]-agent_pos[2])**2
            )
            if distance < self._config.get('TERMINATION_RADIUS', 0.3):
                self._metric = True
                break

        if self._metric:
            task.is_stop_called = True


@registry.register_measure
class ObjectDistance(Measure):

    def __init__(
        self, sim: "HabitatSim", config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self.uuid = "object_distance"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "object_distance"

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._metric = False

    def update_metric(
        self,
        episode,
        task: EmbodiedTask,
        *args: Any,
        **kwargs: Any
    ):
        agent_pos = self._sim.get_agent_state().position

        distance = None
        for obj in self._sim.objects:
            pos = self._sim.get_translation(obj)
            current_distance = np.sqrt(
                (pos[0] - agent_pos[0]) ** 2
                + (pos[2] - agent_pos[2]) ** 2
            )
            if distance == None or current_distance < distance:
                distance = current_distance

        if distance is None:
            distance = np.NaN

        self._metric = distance



@registry.register_measure
class PeoplePositioning(Measure):

    def __init__(
        self, sim: "HabitatSim", config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self.uuid = "people_positioning"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "people_positioning"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self.update_metric(  # type:ignore
            episode=episode, task=task, *args, **kwargs
        )

    def update_metric(
        self,
        episode,
        task: EmbodiedTask,
        *args: Any,
        **kwargs: Any
    ):
        agent_pos = self._sim.get_agent_state().position
        distance = None
        people_rotation = None
        for p in self._sim.people:
            current_distance = np.sqrt(
                (p.current_position[0]-agent_pos[0])**2
                +(p.current_position[2]-agent_pos[2])**2
            )
            if distance == None or current_distance < distance:
                distance = current_distance
                people_rotation = quat_from_magnum(self._sim.get_rotation(p.object_id)) # p.object_id

        agent_state = self._sim.get_agent_state()
        orientation = np.NaN

        if people_rotation is not None:
            orientation = angle_between_quats(agent_state.rotation, people_rotation)

        if distance is None:
            distance = np.NaN

        self._metric = {
            "distance": distance,
            "orientation": orientation
        }
