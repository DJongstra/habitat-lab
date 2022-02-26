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

@registry.register_simulator(name="iGibsonSocialNav")
class iGibsonSocialNav(HabitatSim):
    def __init__(self, config: Config) -> None:
        super().__init__(config=config)
        obj_templates_mgr = self.get_object_template_manager()
        self.num_people = config.get('NUM_PEOPLE', 0)
        self.people_template_ids = []
        self.people_template_ids = obj_templates_mgr.load_configs(
            # For each object, an <object>.object_config.json file is required.
            # See:https://aihabitat.org/docs/habitat-sim/attributesJSON.html#objectattributes
            # for more information.
            "./meshes/person_meshes"
        )

        self.obj_template_ids = obj_templates_mgr.load_configs("./meshes/simple_objects")
        self.person_ids = []
        self.people_mask = config.get('PEOPLE_MASK', False)
        self.num_people = config.get('NUM_PEOPLE', 1)
        self.social_nav = True
        self.interactive_nav = False

        # People params
        self.people_mask = config.get('PEOPLE_MASK', False)
        self.lin_speed = config.PEOPLE_LIN_SPEED
        self.ang_speed = np.deg2rad(config.PEOPLE_ANG_SPEED)
        self.time_step = config.TIME_STEP

        # Objects
        self.objects = []
        self.num_objects = config.get('NUM_OBJECTS', 10)


    def reset_people(self):
        agent_position = self.get_agent_state().position
        obj_templates_mgr = self.get_object_template_manager()

        for inst in self.get_existing_object_ids():
            self.remove_object(inst)

        # Check if humans have been erased (sim was reset)
        if not self.get_existing_object_ids():
            self.person_ids = []
            people_count = 0
            first_person = np.random.randint(0, len(self.people_template_ids))
            while people_count < self.num_people:
                self.person_ids.append(self.add_object(
                    self.people_template_ids[
                        (people_count + first_person) % len(
                            self.people_template_ids)]))
                people_count += 1

        #spawn objects
        self.objects = []
        obj_count = 0
        first_object = np.random.randint(0, len(self.obj_template_ids))
        while obj_count < self.num_objects:
            obj = self.add_object(self.obj_template_ids[(obj_count+first_object)%len(self.obj_template_ids)])
            start = np.array(self.sample_navigable_point())
            self.set_translation([start[0], start[1], start[2]], obj)
            self.set_object_motion_type(
                habitat_sim.physics.MotionType.STATIC,
                obj
            )
            self.objects.append(obj)
            obj_count += 1

        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_success = self.recompute_navmesh(
            self.pathfinder, navmesh_settings, include_static_objects=True
        )

        # Spawn humans
        min_path_dist = 3
        max_level = 0.6
        agent_x, agent_y, agent_z = self.get_agent_state(0).position
        self.people = []
        for person_id in self.person_ids:
            valid_walk = False
            while not valid_walk:
                start = np.array(self.sample_navigable_point())
                goal = np.array(self.sample_navigable_point())
                distance = self.geodesic_distance(start, goal)
                valid_distance = distance > min_path_dist
                valid_level = (
                    abs(start[1]-agent_position[1]) < max_level
                    and abs(goal[1]-agent_position[1]) < max_level
                )
                sp = habitat_sim.nav.ShortestPath()
                sp.requested_start = start
                sp.requested_end   = goal
                found_path = self.pathfinder.find_path(sp)
                valid_start = np.sqrt(
                    (start[0]-agent_x)**2
                    +(start[2]-agent_z)**2
                ) > 0.5
                valid_walk = (
                    valid_distance and valid_level
                    and found_path and valid_start
                )
                if not valid_distance:
                    min_path_dist *= 0.95

            waypoints = []
            if not (self.lin_speed == 0.0 and self.ang_speed == 0.0):
                waypoints = sp.points
            else:
                waypoints = [sp.points[0]]

            heading = np.random.rand() * 2 * np.pi - np.pi
            rotation = np.quaternion(np.cos(heading), 0, np.sin(heading), 0)
            rotation = np.normalized(rotation)
            rotation = mn.Quaternion(
                rotation.imag, rotation.real
            )

            self.set_translation([start[0], start[1]+0.8, start[2]], person_id)
            self.set_rotation(rotation, person_id)
            self.set_object_motion_type(
                habitat_sim.physics.MotionType.KINEMATIC,
                person_id
            )
            spf = ShortestPathFollowerv2(
                sim=self,
                object_id=person_id,
                waypoints=waypoints,
                lin_speed=self.lin_speed,
                ang_speed=self.ang_speed,
                time_step=self.time_step,
            )
            self.people.append(spf)

    def get_observations_at(
        self,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        keep_agent_at_new_pose: bool = False,
    ) -> Optional[Observations]:
        observations = super().get_observations_at(
            position,
            rotation,
            keep_agent_at_new_pose,
        )

        if observations is None:
            return None

        if not self.people_mask:
            return observations

        '''
        Get pixels of just people
        '''
        # 'Remove' people
        all_pos = []
        for person_id in self.get_existing_object_ids():
            pos = self.get_translation(person_id)
            all_pos.append(pos)
            self.set_translation([pos[0], pos[1]+10, pos[2]], person_id)

        # Refresh observations
        no_ppl_observations = super().get_observations_at(
            position=position,
            rotation=rotation,
            keep_agent_at_new_pose=True,
        )

        # Remove non-people pixels
        observations['people'] = observations['depth'].copy()
        observations['people'][
            observations['people'] == no_ppl_observations['depth']
        ] = 0

        # Put people back
        for pos, person_id in zip(all_pos, self.get_existing_object_ids()):
            self.set_translation(pos, person_id)

        return observations

class ShortestPathFollowerv2:
    def __init__(
        self,
        sim,
        object_id,
        waypoints,
        lin_speed,
        ang_speed,
        time_step,
    ):
        self._sim = sim
        self.object_id = object_id

        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local    = True
        self.vel_control.ang_vel_is_local    = True

        self.waypoints = list(waypoints)+list(waypoints)[::-1][1:-1]
        self.next_waypoint_idx = 1
        self.done_turning = False
        self.current_position = waypoints[0]

        # People params
        self.lin_speed = lin_speed
        self.ang_speed = ang_speed
        self.time_step = time_step
        self.max_linear_vel = np.random.rand()*(0.1)+self.lin_speed-0.1

    def step(self):
        waypoint_idx = self.next_waypoint_idx % len(self.waypoints)
        waypoint = np.array(self.waypoints[waypoint_idx])

        translation = self._sim.get_translation(self.object_id)
        mn_quat     = self._sim.get_rotation(self.object_id)

        # Face the next waypoint if we aren't already facing it
        if not self.done_turning:
            # Get current global heading
            heading = np.quaternion(mn_quat.scalar, *mn_quat.vector)
            heading = -quat_to_rad(heading)+np.pi/2

            # Get heading necessary to face next waypoint
            theta = math.atan2(
                waypoint[2]-translation[2], waypoint[0]-translation[0]
            )


            theta_diff = get_heading_error(heading, theta)
            direction = 1 if theta_diff < 0 else -1

            # If next turn would normally overshoot, turn just the right amount
            if self.ang_speed*self.time_step*1.2 >= abs(theta_diff):
                angular_velocity = -theta_diff / self.time_step
                self.done_turning = True
            else:
                angular_velocity = self.ang_speed*direction

            self.vel_control.linear_velocity = np.zeros(3)
            self.vel_control.angular_velocity = np.array([
                0.0, angular_velocity, 0.0
            ])

        # Move towards the next waypoint
        else:
            # If next move would normally overshoot, move just the right amount
            distance = np.sqrt(
                (translation[0]-waypoint[0])**2+(translation[2]-waypoint[2])**2
            )
            if self.max_linear_vel*self.time_step*1.2 >= distance:
                linear_velocity = distance / self.time_step
                self.done_turning = False
                self.next_waypoint_idx += 1
            else:
                linear_velocity = self.max_linear_vel

            self.vel_control.angular_velocity = np.zeros(3)
            self.vel_control.linear_velocity = np.array([
                0.0, 0.0, linear_velocity
            ])

        rigid_state = habitat_sim.bindings.RigidState(
            mn_quat,
            translation
        )
        rigid_state = self.vel_control.integrate_transform(
            self.time_step, rigid_state
        )

        self._sim.set_translation(rigid_state.translation, self.object_id)
        self._sim.set_rotation(rigid_state.rotation, self.object_id)
        self.current_position = rigid_state.translation

