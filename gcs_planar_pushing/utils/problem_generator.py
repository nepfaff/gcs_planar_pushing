import numpy as np
from pydrake.all import Rgba, RigidTransform, RotationMatrix


class ProblemGenerator:
    def __init__(
        self,
        n_samples: int,
        workspace_radius: float,
        object_max_radius: float,
        robot_max_radius: float,
        plant,
        plant_context,
        object,
        robot,
    ):
        self.n_samples = n_samples
        self.workspace_radius = workspace_radius
        self.object_max_radius = object_max_radius
        self.robot_max_radius = robot_max_radius

        self.plant = plant
        self.plant_context = plant_context
        self.object = object
        self.robot = robot

        assert (
            self.workspace_radius > self.object_max_radius + self.robot_max_radius
        ), "Workspace must be larger than the sum of the object and robot radii."

    def generate_initial_positions(self):

        # Hardcoded robot and object dimensions:
        dim_robot = 2
        dim_object = 2
        obj_bound = self.workspace_radius - self.object_max_radius
        object_bounds = [
            [-obj_bound, obj_bound],  # x bounds
            [-obj_bound, obj_bound],  # y bounds
        ]
        rob_bound = self.workspace_radius - self.robot_max_radius
        robot_bounds = [
            [-rob_bound, rob_bound],  # x bounds
            [-rob_bound, rob_bound],  # y bounds
        ]

        object_pos = np.zeros((self.n_samples, dim_object))
        robot_pos = np.zeros((self.n_samples, dim_robot))

        count = 0
        while count != self.n_samples:
            for i in range(dim_object):
                object_pos[count, i] = np.random.uniform(
                    object_bounds[i][0], object_bounds[i][1]
                )
            for i in range(dim_robot):
                robot_pos[count, i] = np.random.uniform(
                    robot_bounds[i][0], robot_bounds[i][1]
                )

            self.plant.SetPositions(
                self.plant_context, self.robot.model_instance(), robot_pos[count]
            )
            if not ProblemGenerator.is_body_in_contact(
                self.robot.index(),
                self.plant.GetOutputPort("contact_results").Eval(self.plant_context),
            ):
                count += 1
            else:
                print(
                    f"Rejected sample {count}, robot pos: {robot_pos[count]}, object pos: {object_pos[count]}"
                )

        return object_pos, robot_pos

    @staticmethod
    def is_body_in_contact(body_of_interest, contact_results, print_result=False):
        formatter = {"float": lambda x: "{:5.2f}".format(x)}

        if contact_results.num_point_pair_contacts() == 0:
            if print_result:
                print("no contact")
        for i in range(contact_results.num_point_pair_contacts()):
            info = contact_results.point_pair_contact_info(i)
            if (
                info.bodyA_index() == body_of_interest
                or info.bodyB_index() == body_of_interest
            ):
                pair = info.point_pair()
                force_string = np.array2string(
                    info.contact_force(), formatter=formatter
                )
                if print_result:
                    print(
                        f"body A: {info.bodyA_index()}, "
                        f"body B: {info.bodyB_index()}, "
                        f"slip speed:{info.slip_speed():.4f}, "
                        f"depth:{pair.depth:.4f}, "
                        f"force:{force_string}\n"
                    )
                return True
        return False
