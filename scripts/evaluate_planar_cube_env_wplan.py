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
    success_sim_times = []
    for i in range(cfg.num_evaluation_rounds):
        for j, (object_pos, robot_pos) in tqdm(
            enumerate(zip(object_positions, robot_positions))
        ):
            cfg.environment.initial_box_position = object_pos.tolist()
            cfg.environment.initial_finger_position = robot_pos.tolist()

            controller: ControllerBase = instantiate(cfg.controller)
            environment: EnvironmentBase = instantiate(
                cfg.environment, controller=controller
            )
            environment.setup()
            plan_time = (
                controller._plan_time
            )  # This needs to be after setup() because gcs planning is done in setup

            start_time = time.time()
            success, simulation_time = environment.simulate()
            run_time = time.time() - start_time

            if success:
                num_success += 1
                success_sim_times.append(
                    [f"sim_{i}_{j}", simulation_time, run_time, plan_time]
                )

            # Save simulation
            html = open("simulation.html")
            wandb.log(
                {
                    f"sim_{i}_{j}_{'success' if success else 'failure'}": wandb.Html(
                        html, inject=False
                    )
                }
            )

            # Cleanup to prevent running out of GPU memory
            del controller, environment

    wandb.log(
        {
            "success_simulation_times": wandb.Table(
                data=success_sim_times,
                columns=["simulation_idx", "sim_time_s", "run_time_s", "plan_time_s"],
            )
        }
    )

    success_sim_times = np.asarray(success_sim_times)
    sim_times = [float(el) for el in success_sim_times[:, 1]]
    run_times = [float(el) for el in success_sim_times[:, 2]]
    plan_times = [float(el) for el in success_sim_times[:, 3]]
    metric_dict = {
        "Success rate": num_success / len(object_positions),
        "Average success simulation time": np.mean(sim_times),
        "Std success simulation time": np.std(sim_times),
        "Max success simulation time": np.max(sim_times),
        "Min success simulation time": np.min(sim_times),
        "Average success run time": np.mean(run_times),
        "Std success run time": np.std(run_times),
        "Max success run time": np.max(run_times),
        "Minsuccess run time": np.min(run_times),
        "Average success plan time": np.mean(plan_times),
        "Std success plan time": np.std(plan_times),
        "Max success plan time": np.max(plan_times),
        "Min success run time": np.min(plan_times),
    }
    wandb.log(metric_dict)
    print(metric_dict)


if __name__ == "__main__":
    main()
