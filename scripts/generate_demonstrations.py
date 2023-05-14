"""A script for generating demonstrations."""
import os
import pathlib
import numpy as np
from gcs_planar_pushing.utils.util import save_meshcat
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from tqdm import tqdm
import zarr
from pydrake.all import (
    StartMeshcat,
)
from datetime import datetime

from gcs_planar_pushing.environments import EnvironmentBase
from gcs_planar_pushing.utils.problem_generator import ProblemGenerator
from gcs_planar_pushing.controllers import ControllerBase, PlanarCubeTeleopController
from diffusion_policy.common.replay_buffer import ReplayBuffer


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("../..", "config")),
)
def main(cfg: OmegaConf):
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    print(cfg)
    hydra_config = HydraConfig.get()
    full_log_dir = hydra_config.runtime.output_dir

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
    initial_positions = zarr.open_group(
        os.path.join(full_log_dir, f"initial_positions_{date_time}.zarr"), mode="w"
    )
    initial_positions["object_pos"] = object_pos
    initial_positions["robot_pos"] = robot_pos
    if cfg.only_generate_initial_positions:
        return
    # print(f"Object position: {object_pos}")
    # print(f"Robot position: {robot_pos}")

    # # Override for testing: 1 trajectory
    # object_pos, robot_pos = np.array([[-1.0, 0.0]]), np.array([[5.0, 0.0]])

    # # Override for testing: 3 trajectories
    # object_pos = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    # robot_pos = np.array([[-5.0, 0.0], [5.0, 0.0], [0.0, 5.0]])

    buffer = ReplayBuffer.create_empty_numpy()
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
        image_data, state_data, action_data = environment.generate_data(
            cfg.log_every_k_sim_timesteps
        )
        # Add data to ReplayBuffer
        episode = {
            "state": state_data,
            "action": action_data,
            "img": image_data,
        }
        buffer.add_episode(episode)
        if cfg.save_meshcats:
            save_meshcat(
                os.path.join(full_log_dir, "meshcats", f"meshcat_{date_time}_{i}"),
                meshcat,
            )

        # Save after every episode
        buffer.save_to_path(
            os.path.join(full_log_dir, f"replay_{date_time}.zarr"),
            chunk_length=cfg.chunk_length,
        )
        print(f"Completed episode {i} of {len(object_pos)}")


if __name__ == "__main__":
    main()
