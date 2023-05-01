"""A script for generating demonstrations."""

import pathlib

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm
from pydrake.all import (
    StartMeshcat,
)
from gcs_planar_pushing.environments import EnvironmentBase
from gcs_planar_pushing.utils.problem_generator import ProblemGenerator
from gcs_planar_pushing.controllers import ControllerBase, PlanarCubeTeleopController


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("../..", "config")),
)
def main(cfg: OmegaConf):
    print(cfg)
    meshcat = StartMeshcat()
    # Use teleop environment to generate initial positions:
    sphere_pid_gains = OmegaConf.create({"kp": 100, "kd": 10, "ki": 1})
    teleop = OmegaConf.create(
        {"input_limit": 10.0, "step_size": 0.1, "start_translation": [0.0, 0.0]}
    )
    introspection_controller: ControllerBase = PlanarCubeTeleopController(
        time_step=1e-3, sphere_pid_gains=sphere_pid_gains, teleop=teleop
    )
    environment: EnvironmentBase = instantiate(
        cfg.environment, controller=introspection_controller
    )
    environment.setup(meshcat)

    # Generate initial robot and box positions:
    problem_generator = instantiate(
        cfg.problem_generator,
        plant=environment.plant,
        plant_context=environment.plant_context,
        robot=environment.robot,
        object=environment.object,
    )
    object_pos, robot_pos = problem_generator.generate_initial_positions()
    print(f"Object position: {object_pos}")
    print(f"Robot position: {robot_pos}")

    # Now use execution controller
    for i in tqdm(range(len(object_pos)), desc="Generating demonstrations"):
        meshcat.Delete()  # Clear meshcat between runs
        meshcat.DeleteAddedControls()
        cfg.environment.initial_box_position = object_pos[i].tolist()
        cfg.environment.initial_finger_position = robot_pos[i].tolist()
        print(f"Initial box position: {cfg.environment.initial_box_position}")
        print(f"Initial finger position: {cfg.environment.initial_finger_position}")
        controller: ControllerBase = instantiate(cfg.controller)
        environment: EnvironmentBase = instantiate(
            cfg.environment, controller=controller
        )
        environment.setup(meshcat)
        environment.simulate()


if __name__ == "__main__":
    main()
