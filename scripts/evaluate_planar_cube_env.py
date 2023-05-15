import pathlib
import time
import os
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
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
    hydra_config = HydraConfig.get()
    full_log_dir = hydra_config.runtime.output_dir

    script_start_time = time.time()

    initial_condition_data = zarr.open(cfg.initial_conditions_zarr_path)
    object_positions = initial_condition_data.object_pos[:]
    robot_positions = initial_condition_data.robot_pos[:]

    num_success = 0
    num_successes_per_initial_condition = np.zeros(len(object_positions))
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

            positions_log = zarr.open_group(
                os.path.join(full_log_dir, f"positions_{i}_{j}.zarr"), mode="w"
            )
            positions_log["object_pos"] = environment.episode_box_positions
            positions_log["robot_pos"] = environment.episode_robot_positions

            if success:
                num_success += 1
                num_successes_per_initial_condition[j] += 1
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
    wandb.log(
        {
            "success_ratio_per_initial_condition": wandb.Table(
                data=np.concatenate(
                    (
                        [[f"sim_{i}"] for i in range(len(object_positions))],
                        num_successes_per_initial_condition[:, np.newaxis]
                        / cfg.num_evaluation_rounds,
                    ),
                    axis=1,
                ),
                columns=["simulation_idx", "success_ratio"],
            )
        }
    )

    success_sim_times = np.asarray(success_sim_times)
    sim_times = [float(el) for el in success_sim_times[:, 1]]
    run_times = [float(el) for el in success_sim_times[:, 2]]
    plan_times = [float(el) for el in success_sim_times[:, 3]]
    metric_dict = {
        "Success rate": num_success
        / (len(object_positions) * cfg.num_evaluation_rounds),
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
    metric_data = [[name, value] for name, value in metric_dict.items()]
    wandb.log({"metrics": wandb.Table(data=metric_data, columns=["metric", "value"])})
    print(metric_dict)

    print(f"The evaluation took {time.time()-script_start_time} seconds.")


if __name__ == "__main__":
    main()
