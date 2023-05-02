from .controller_base import ControllerBase

from typing import Dict, Tuple
import pathlib
from collections import deque

from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    LeafSystem,
    InverseDynamicsController,
    StateInterpolatorWithDiscreteDerivative,
    System,
    AbstractValue,
    RgbdSensor,
    Image,
)
import skimage
import numpy as np
import torch
import dill
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
    DiffusionUnetHybridImagePolicy,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.drake_cube_image_dataset import DrakeCubeImageDataset

from gcs_planar_pushing.utils import get_parser


class DiffusionPolicy(LeafSystem):
    """A Diffusion Policy controller for controlling a robot."""

    def __init__(
        self,
        initial_pos: np.ndarray,
        checkpoint_path: str,
        dataset_path: str,
    ):
        """
        Args:
            initial_pos (np.ndarray): The initial robot position of shape (N,).
        """
        super().__init__()
        self._desired_pos = initial_pos

        # TODO: Make these arguments
        self._pred_horizon = 16
        self._obs_horizon = 2
        self._action_horizon = 8
        self._pose_observation_cache = deque([], maxlen=self._obs_horizon)
        self._image_observation_cache = deque([], maxlen=self._obs_horizon)
        self._action_cache = deque([], maxlen=self._action_horizon)

        # Get dataset specific normalizer
        dataset = DrakeCubeImageDataset(
            horizon=self._pred_horizon,
            max_train_episodes=90,
            pad_after=7,
            pad_before=1,
            seed=42,
            val_ratio=0.02,
            zarr_path=dataset_path,
        )
        image_shape = dataset[0]["obs"]["image"].shape[-2:]
        normalizer = dataset.get_normalizer()

        self._policy = self._load_diffusion_policy(
            pathlib.Path(checkpoint_path), normalizer
        )

        self.DeclareVectorOutputPort(
            "robot_pos_desired", len(initial_pos), self._get_desired_pos
        )
        self._robot_state_actual_port = self.DeclareVectorInputPort(
            "robot_state_actual", len(initial_pos) * 2
        )
        self._image_obs_current_port = self.DeclareAbstractInputPort(
            "image_obs_current",
            AbstractValue.Make(Image(image_shape[0], image_shape[1])),
        )

    def _load_diffusion_policy(
        self,
        checkpoint_path: pathlib.Path,
        normalizer: LinearNormalizer,
    ) -> DiffusionUnetHybridImagePolicy:
        print(f"Getting diffusion policy from checkpoint {checkpoint_path}.")
        assert checkpoint_path.is_file()

        # TODO: Make this configurable with hyra
        noise_scheduler = DDPMScheduler(
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            beta_start=0.0001,
            clip_sample=True,
            num_train_timesteps=100,
            prediction_type="epsilon",
            variance_type="fixed_small",
        )
        ema_model = DiffusionUnetHybridImagePolicy(
            cond_predict_scale=True,
            crop_shape=[84, 84],
            diffusion_step_embed_dim=128,
            down_dims=[512, 1024, 2048],
            eval_fixed_crop=True,
            horizon=self._pred_horizon,  # Prediction horizon
            kernel_size=5,
            n_action_steps=self._action_horizon,  # Actions to take
            n_groups=8,
            n_obs_steps=self._obs_horizon,  # Observations to use for action prediction
            noise_scheduler=noise_scheduler,
            num_inference_steps=100,
            obs_as_global_cond=True,
            obs_encoder_group_norm=True,
            shape_meta={
                "action": {"shape": [2]},
                "obs": {
                    "agent_pos": {"shape": [2], "type": "low_dim"},
                    "image": {"shape": [3, 96, 96], "type": "rgb"},
                },
            },
        )
        ema_model.set_normalizer(normalizer)

        payload = torch.load(checkpoint_path.open("rb"), pickle_module=dill)
        ema_model.load_state_dict(payload["state_dicts"]["ema_model"])

        return ema_model

    def _get_desired_pos(self, context, output) -> None:
        robot_pos_actual = self._robot_state_actual_port.Eval(context)[
            : len(self._desired_pos)
        ]
        image_obs_current = self._image_obs_current_port.Eval(context).data
        # Discard alpha channel
        image_obs_current = image_obs_current[:, :, :3]
        # Convert pixels from int to float
        image_obs_current = skimage.img_as_float32(image_obs_current)
        # Move color chanel to front
        image_obs_current = np.transpose(image_obs_current, (2, 0, 1))

        self._pose_observation_cache.append(robot_pos_actual)
        self._image_observation_cache.append(image_obs_current)

        if len(self._pose_observation_cache) < self._obs_horizon:
            # Only start when the observation cache is full
            output.SetFromVector(self._desired_pos)
            return

        if not np.allclose(self._desired_pos, robot_pos_actual):
            # Demand the desired pose until we reach it
            output.SetFromVector(self._desired_pos)
            return

        if len(self._action_cache) == 0:  # Compute new actions
            obs_dict_np = {
                "agent_pos": np.asarray(self._pose_observation_cache)[np.newaxis, :],
                "image": np.asarray(self._image_observation_cache)[np.newaxis, :],
            }
            obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x))

            self._policy.eval()
            with torch.no_grad():
                action_dict = self._policy.predict_action(obs_dict)

            action_dict_np = dict_apply(
                action_dict, lambda x: x.squeeze(0).detach().to("cpu").numpy()
            )
            actions = action_dict_np["action"]
            self._action_cache.extend(actions)

        self._desired_pos = self._action_cache.popleft()
        output.SetFromVector(self._desired_pos)


class DiffusionPolicyController(ControllerBase):
    """A Diffusion Policy controller."""

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

    def _setup_diffusion_policy_controller(
        self, builder: DiagramBuilder, plant: MultibodyPlant, rgbd_sensor: RgbdSensor
    ) -> Tuple[System, System]:
        # TODO: Make these arguments
        finger_position_source = builder.AddSystem(
            DiffusionPolicy(
                initial_pos=self._initial_finger_position,
                checkpoint_path="../diffusion_policy/data/outputs/2023.04.30/15.17.37_train_"
                + "diffusion_unet_hybrid_planar_cube_image/checkpoints/latest.ckpt",
                dataset_path="../diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr",
            )
        )

        # Add discrete derivative to command velocities.
        desired_state_source = builder.AddSystem(
            StateInterpolatorWithDiscreteDerivative(
                self._num_sphere_positions,
                self._time_step,
                suppress_initial_transient=True,
            )
        )
        desired_state_source.set_name("diffusion_policy_desired_finger_state_source")
        builder.Connect(
            finger_position_source.get_output_port(),
            desired_state_source.get_input_port(),
        )
        builder.Connect(
            plant.get_state_output_port(plant.GetModelInstanceByName("sphere")),
            finger_position_source.get_input_port(0),
        )
        builder.Connect(
            rgbd_sensor.color_image_output_port(),
            finger_position_source.get_input_port(1),
        )

        return desired_state_source

    def setup(
        self,
        builder: DiagramBuilder,
        plant: MultibodyPlant,
        rgbd_sensor: RgbdSensor,
        **kwargs,
    ) -> None:
        if self._meshcat is None:
            raise RuntimeError(
                "Need to call `add_meshcat` before calling `setup` of the teleop controller."
            )

        sphere_controller = self._setup_sphere_controller(builder, plant)
        finger_state_source = self._setup_diffusion_policy_controller(
            builder, plant, rgbd_sensor
        )
        builder.Connect(
            finger_state_source.get_output_port(),
            sphere_controller.get_input_port_desired_state(),
        )
