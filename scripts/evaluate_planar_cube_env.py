import pathlib
import time

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
import wandb
import zarr
from tqdm import tqdm
import numpy as np

from gcs_planar_pushing.environments import EnvironmentBase
from gcs_planar_pushing.controllers import ControllerBase


def setup_env_and_simulate(cfg: OmegaConf) -> bool:
    controller: ControllerBase = instantiate(cfg.controller)
    environment: EnvironmentBase = instantiate(cfg.environment, controller=controller)
    environment.setup()
    success = environment.simulate()
    return success


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("../..", "config")),
)
def main(cfg: OmegaConf):
    current_time = time.strftime("%Y-%b-%d-%H-%M-%S")
    wandb.init(
        project="gcs_planar_pushing",
        name=f"evaluate_planar_cube_env_{current_time}",
        mode="online",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    initial_condition_data = zarr.open(cfg.initial_conditions_zarr_path)
    object_positions = initial_condition_data.object_pos[:]
    robot_positions = initial_condition_data.robot_pos[:]

    num_success = 0
    sim_times = []
    for i, (object_pos, robot_pos) in tqdm(
        enumerate(zip(object_positions, robot_positions))
    ):
        cfg.environment.initial_box_position = object_pos.tolist()
        cfg.environment.initial_finger_position = robot_pos.tolist()
        success, simulation_time = setup_env_and_simulate(cfg)
        if success:
            num_success += 1
            sim_times.append([f"sim_{i}", simulation_time])

        # Save simulation
        html = open("simulation.html")
        wandb.log(
            {
                f"sim_{i}_{'success' if success else 'failure'}": wandb.Html(
                    html, inject=False
                )
            }
        )

    wandb.log(
        {
            "success_simulation_times": wandb.Table(
                data=sim_times, columns=["simulation_idx", "time_s"]
            )
        }
    )

    sim_times = np.array(sim_times)[:, 1]
    metric_dict = {
        "Success rate": num_success / len(object_positions),
        "Average success simulation time": np.mean(sim_times),
        "Std success simulation time": np.std(sim_times),
        "Max success simulation time": np.max(sim_times),
        "Minsuccess simulation time": np.min(sim_times),
    }
    wandb.log(metric_dict)
    print(metric_dict)


if __name__ == "__main__":
    main()
