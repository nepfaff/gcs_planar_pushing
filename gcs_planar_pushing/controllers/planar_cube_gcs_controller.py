from .controller_base import ControllerBase

from pydrake.all import DiagramBuilder, MultibodyPlant
import numpy as np
from pydrake.math import eq
from planning_through_contact.geometry.collision_pair import CollisionPair
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


class PlanarCubeGCSController(ControllerBase):
    """An open-loop GCS controller."""

    def __init__(self, time_step: float):
        super().__init__(time_step)

    def setup(self, builder: DiagramBuilder, plant: MultibodyPlant) -> None:
        super().setup(builder, plant)

    def plan_for_box_pushing_3d(self):
        # Bezier curve params
        problem_dim = 3
        bezier_curve_order = 1

        mass = 1  # kg
        g = 9.81  # m/s^2
        mg = mass * g
        # Depth, width, height are 0.5* actual values
        box_width = 1
        box_height = 1
        box_depth = 1
        floor_width = 20
        floor_height = 20
        floor_depth = 1
        friction_coeff = 0.5

        finger = RigidBody(
            dim=problem_dim,
            position_curve_order=bezier_curve_order,
            name="f",
            geometry="point",
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

        p1 = CollisionPair(
            finger,
            box,
            friction_coeff,
            position_mode=PositionModeType.LEFT,  # Finger on left side of box
        )
        p2 = CollisionPair(
            box,
            ground,
            friction_coeff,
            position_mode=PositionModeType.FRONT,  # Box on top of ground
        )
        collision_pairs = [p1, p2]

        # Specify problem
        no_ground_motion = [eq(x_g, 0), eq(y_g, 0), eq(z_g, -floor_depth)]
        no_vertical_movement = [eq(z_f, box_depth), eq(z_b, box_depth)]
        additional_constraints = [*no_ground_motion, *no_vertical_movement]
        source_config = ContactModeConfig(
            modes={
                p1.name: ContactModeType.NO_CONTACT,  # Finger not in contact with box
                p2.name: ContactModeType.ROLLING,  # Box in contact with floor
            },
            additional_constraints=[
                eq(x_f, 0),
                eq(y_f, 0),
                eq(x_b, 6.0),
                eq(y_b, 0.0),
            ],
        )
        target_config = ContactModeConfig(
            modes={
                p1.name: ContactModeType.NO_CONTACT,
                p2.name: ContactModeType.ROLLING,
            },
            additional_constraints=[
                eq(x_b, 10.0),
                eq(y_b, 0.0),
                eq(x_f, 0.0),
                eq(y_f, 0.0),
            ],
        )

        # TODO this is very hardcoded
        gravitational_jacobian = np.array([[0, 0, -1, 0, 0, -1, 0, 0, -1]]).T
        external_forces = gravitational_jacobian.dot(mg)

        planner = GcsContactPlanner(
            rigid_bodies,
            collision_pairs,
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
        planner.allow_revisits_to_vertices(1)  # Why do we need this?
        planner.save_graph_diagram("graph_box_pushing_with_revisits.svg")

        # TODO add weights here
        planner.add_position_continuity_constraints()
        planner.add_position_path_length_cost()
        planner.add_force_path_length_cost()
        planner.add_num_visited_vertices_cost(100)
        planner.add_force_strength_cost()

        result = planner.solve()
        ctrl_points = planner.get_ctrl_points(result)
        (
            pos_curves,
            normal_force_curves,
            friction_force_curves,
        ) = planner.get_curves_from_ctrl_points(ctrl_points)

        plot_positions_and_forces(
            pos_curves, normal_force_curves, friction_force_curves
        )
        animate_positions(pos_curves, rigid_bodies)
