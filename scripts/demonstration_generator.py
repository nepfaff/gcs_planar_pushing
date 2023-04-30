"""A script for generating demonstrations."""

import pathlib

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from gcs_planar_pushing.environments import EnvironmentBase
from gcs_planar_pushing.utils.problem_generator import ProblemGenerator


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("../..", "config")),
)
def main(cfg: OmegaConf):
    print(cfg)
    environment: EnvironmentBase = instantiate(cfg.environment)
    environment.setup()

    # Generate initial robot and box positions:
    problem_generator = ProblemGenerator(
        workspace_radius=10,
        object_max_radius=0.5,
        robot_max_radius=0.1,
        plant=environment.plant,
        plant_context=environment.plant_context,
        robot=environment.robot,
        object=environment.object,
    )
    object_pos, robot_pos = problem_generator.generate_initial_positions(30)
    print(f"Object position: {object_pos}")
    print(f"Robot position: {robot_pos}")

    # environment.simulate()
    print(f"Initial box position: {cfg.environment.initial_box_position}")
    print(f"Initial box position: {cfg.environment.initial_finger_position}")


if __name__ == "__main__":
    main()
