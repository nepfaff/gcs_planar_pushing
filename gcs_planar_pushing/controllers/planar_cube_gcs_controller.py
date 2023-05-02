from .controller_base import ControllerBase

from functools import reduce
from typing import Dict, List

from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    LeafSystem,
    InverseDynamicsController,
    StateInterpolatorWithDiscreteDerivative,
    System,
)
import numpy as np
from pydrake.math import eq
from planning_through_contact.geometry.object_pair import ObjectPair
from planning_through_contact.geometry.contact_mode import (
    ContactModeType,
    PositionModeType,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.gcs import GcsContactPlanner
from planning_through_contact.planning.graph_builder import ContactModeConfig
from planning_through_contact.visualize.visualize import (
    animate_positions,
    plot_positions_and_forces,
)

from gcs_planar_pushing.utils import get_parser


class PositionSource(LeafSystem):
    def __init__(self, pos_path: np.ndarray, step_length_seconds: float):
        LeafSystem.__init__(self)
        self._pos_path = pos_path
        self._step_length_seconds = step_length_seconds

        self.DeclareVectorOutputPort(
            "pos_desired", pos_path.shape[1], self._get_next_pos
        )

    def _get_next_pos(self, context, output):
        idx = int(context.get_time() // self._step_length_seconds)
        if idx >= len(self._pos_path):
            output.SetFromVector(self._pos_path[-1])
        else:
            output.SetFromVector(self._pos_path[idx])


class PlanarCubeGCSController(ControllerBase):
    """An open-loop GCS controller."""

    def __init__(
        self,
        time_step: float,
        sphere_pid_gains: Dict[str, float],
    ):
        super().__init__(time_step)

        self._sphere_pid_gains = sphere_pid_gains
        self._num_sphere_positions = 2

    def _setup_sphere_controller(
        self, builder: DiagramBuilder, plant: MultibodyPlant
    ) -> System:
        sphere_model_instance = plant.GetModelInstanceByName("sphere")
        sphere_controller_plant = MultibodyPlant(time_step=self._time_step)
        parser = get_parser(sphere_controller_plant)
        parser.AddModelsFromUrl(
            "package://gcs_planar_pushing/models/planar_cube/actuated_sphere.urdf"
        )[0]
        sphere_controller_plant.set_name("sphere_controller_plant")
        sphere_controller_plant.Finalize()

        sphere_controller = builder.AddSystem(
            InverseDynamicsController(
                sphere_controller_plant,
                kp=[self._sphere_pid_gains.kp] * self._num_sphere_positions,
                ki=[self._sphere_pid_gains.ki] * self._num_sphere_positions,
                kd=[self._sphere_pid_gains.kd] * self._num_sphere_positions,
                has_reference_acceleration=False,
            )
        )
        sphere_controller.set_name("sphere_controller")
        builder.Connect(
            plant.get_state_output_port(sphere_model_instance),
            sphere_controller.get_input_port_estimated_state(),
        )
        builder.Connect(
            sphere_controller.get_output_port_control(),
            plant.get_actuation_input_port(sphere_model_instance),
        )
        return sphere_controller

    def _setup_open_loop_control(self, builder: DiagramBuilder) -> System:
        finger_position_path = self._plan_for_box_pushing_3d(
            initial_box_position=self._initial_box_position,
            initial_finger_position=self._initial_finger_position,
        )
        print(f"len(finger_position_path): {len(finger_position_path)}")

        step_length_seconds = 0.05  # 0.01
        finger_position_source = builder.AddSystem(
            PositionSource(
                finger_position_path, step_length_seconds=step_length_seconds
            )
        )
        self._sim_duration = step_length_seconds * len(finger_position_path)
        print(f"total time: {self._sim_duration}")

        self.desired_pos_source = finger_position_source
        # Add discrete derivative to command velocities.
        desired_state_source = builder.AddSystem(
            StateInterpolatorWithDiscreteDerivative(
                self._num_sphere_positions,
                self._time_step,
                suppress_initial_transient=True,
            )
        )
        desired_state_source.set_name("gcs_open_loop_desired_state_source")
        builder.Connect(
            finger_position_source.get_output_port(),
            desired_state_source.get_input_port(),
        )
        builder.ExportOutput(finger_position_source.get_output_port(), "action")
        return desired_state_source

    def setup(self, builder: DiagramBuilder, plant: MultibodyPlant, **kwargs) -> None:
        if self._meshcat is None:
            raise RuntimeError(
                "Need to call `add_meshcat` before calling `setup` of the teleop controller."
            )

        sphere_controller = self._setup_sphere_controller(builder, plant)
        finger_state_source = self._setup_open_loop_control(builder)
        builder.Connect(
            finger_state_source.get_output_port(),
            sphere_controller.get_input_port_desired_state(),
        )

    def _plan_for_box_pushing_3d(
        self,
        initial_finger_position: List[float],
        initial_box_position: List[float],
        visualize=True,
    ) -> np.ndarray:
        """Creates a planar pushing path.

        Args:
            visualize (bool, optional): Whether to visualize the plan. Defaults to True.

        Returns:
            np.ndarray: The finger position path of shape (N, 3).
        """
        # Bezier curve params
        problem_dim = 3
        bezier_curve_order = 1

        mass = 1  # kg
        g = 9.81  # m/s^2
        mg = mass * g
        # Depth, width, height are 0.5* actual values
        box_width = 0.5
        box_height = 0.5
        box_depth = 0.5
        floor_width = 20
        floor_height = 20
        floor_depth = 1
        friction_coeff = 0.5

        # Add in z = 0
        initial_finger_position = [*initial_finger_position, 0]
        initial_box_position = [*initial_box_position, box_depth]

        finger = RigidBody(
            dim=problem_dim,
            position_curve_order=bezier_curve_order,
            name="f",
            geometry="sphere",
            radius=0.2,
            actuated=True,
        )
        box = RigidBody(
            dim=problem_dim,
            position_curve_order=bezier_curve_order,
            name="b",
            geometry="box",
            width=box_width,
            height=box_height,
            depth=box_depth,
            actuated=False,
        )
        ground = RigidBody(
            dim=problem_dim,
            position_curve_order=bezier_curve_order,
            name="g",
            geometry="box",
            width=floor_width,
            height=floor_height,
            depth=floor_depth,
            actuated=True,
        )
        rigid_bodies = [finger, box, ground]

        x_f = finger.pos_x
        y_f = finger.pos_y
        z_f = finger.pos_z
        x_b = box.pos_x
        y_b = box.pos_y
        z_b = box.pos_z
        x_g = ground.pos_x
        y_g = ground.pos_y
        z_g = ground.pos_z

        p1 = ObjectPair(
            finger,
            box,
            friction_coeff,
            allowed_position_modes=[
                PositionModeType.LEFT,
                PositionModeType.BOTTOM_LEFT,
                PositionModeType.TOP_LEFT,
                PositionModeType.TOP,
                PositionModeType.TOP_RIGHT,
                PositionModeType.RIGHT,
                PositionModeType.BOTTOM,
                PositionModeType.BOTTOM_RIGHT,
                PositionModeType.LEFT_TRANSITION,
                PositionModeType.RIGHT_TRANSITION,
                PositionModeType.TOP_TRANSITION,
                PositionModeType.BOTTOM_TRANSITION,
            ],  # TODO: extend
            allowable_contact_mode_types=[
                ContactModeType.NO_CONTACT,
                ContactModeType.ROLLING,
            ],
            transition_eps=0.5,
            center_contact_buffer=0.0,
        )
        p2 = ObjectPair(
            box,
            ground,
            friction_coeff,
            allowed_position_modes=[PositionModeType.FRONT],
            allowable_contact_mode_types=[
                ContactModeType.ROLLING,
                ContactModeType.SLIDING_DOWN,
                ContactModeType.SLIDING_LEFT,
                ContactModeType.SLIDING_RIGHT,
                ContactModeType.SLIDING_UP,
            ],
        )
        object_pairs = [p1, p2]

        # Specify problem
        ground_position = [0, 0, -floor_depth]
        print(
            f"initial contact_pair of finger,box: {p1.get_contact_pair_for_positions(initial_finger_position, initial_box_position).name}"
        )
        print(
            f"initial contact_pair of box,ground: {p2.get_contact_pair_for_positions(initial_box_position, ground_position).name}"
        )
        no_ground_motion = [eq(x_g, 0), eq(y_g, 0), eq(z_g, -floor_depth)]
        no_vertical_movement = [eq(z_f, box_depth), eq(z_b, box_depth)]
        additional_constraints = [*no_ground_motion, *no_vertical_movement]

        source_config = ContactModeConfig(
            modes={

                p1.get_contact_pair_for_positions(
                    initial_finger_position, initial_box_position
                ).name: ContactModeType.NO_CONTACT,  # Finger not in contact with box
                p2.get_contact_pair_for_positions(
                    initial_box_position, ground_position
                ).name: ContactModeType.ROLLING,  # Box in contact with floor
            },
            additional_constraints=[
                # eq(x_f, initial_finger_position[0]),
                # eq(y_f, initial_finger_position[1]),
                # eq(x_f, -0.5),
                # eq(y_f, 0.0),
                eq(x_b, initial_box_position[0]),
                eq(y_b, initial_box_position[1]),
            ],
        )
        # target_config = source_config
        target_config = ContactModeConfig(
            modes={
                p2.get_contact_pair_for_positions(
                    initial_box_position, ground_position
                ).name: ContactModeType.ROLLING,  # Box in contact with floor
            },
            additional_constraints=[
                eq(x_b, 0.0),
                eq(y_b, 0.0),
            ],
        )

        # TODO this is very hardcoded
        gravitational_jacobian = np.array([[0, 0, -1, 0, 0, -1, 0, 0, -1]]).T
        external_forces = gravitational_jacobian.dot(mg)

        planner = GcsContactPlanner(
            rigid_bodies,
            object_pairs,
            external_forces,
            additional_constraints,
            allow_sliding=True,
        )
        planner.add_source_config(source_config)
        planner.add_target_config(target_config)
        # Build graph by having one convex set for each contact mode and edge between all
        # intersecting sets. A convex contact mode set is constructed from the constraints
        # that hold for that contact mode.
        planner.build_graph(prune=False)
        planner.save_graph_diagram("graph_box_pushing.svg")
        planner.allow_revisits_to_vertices(1)
        # planner.save_graph_diagram("graph_box_pushing_with_revisits.svg")

        # TODO add weights here
        planner.add_position_continuity_constraints()
        planner.add_position_path_length_cost()
        planner.add_force_path_length_cost()
        planner.add_num_visited_vertices_cost(100)
        planner.add_force_strength_cost()
        # planner.add_position_path_time_cost(target_position = [0,0]) # This doesn't work

        result = planner.solve(use_convex_relaxation=False)
        ctrl_points = planner.get_ctrl_points(result)
        (
            pos_curves,
            normal_force_curves,
            friction_force_curves,
        ) = planner.get_curves_from_ctrl_points(ctrl_points)

        # Filter out sections without movement
        filtered_pos = PlanarCubeGCSController._filter_desired_pos(
            pos_curves["f"][:, :2]
        )

        if visualize:
            plot_positions_and_forces(
                pos_curves, normal_force_curves, friction_force_curves
            )
            animate_positions(pos_curves, rigid_bodies)

        return filtered_pos

    @staticmethod
    def _filter_desired_pos(pos_curve):
        diff = np.diff(pos_curve, axis=0)
        dist = np.linalg.norm(diff, axis=1)
        mask = np.concatenate(([True], dist > 0.01))
        filtered_pos = pos_curve[mask]
        return filtered_pos
