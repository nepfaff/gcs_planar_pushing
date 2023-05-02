from typing import Dict, Any

from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    InverseDynamicsController,
    Multiplexer,
    StateInterpolatorWithDiscreteDerivative,
    System,
)
from underactuated.meshcat_utils import MeshcatSliders

from .controller_base import ControllerBase
from gcs_planar_pushing.utils import get_parser


class PlanarCubeTeleopController(ControllerBase):
    """An open-loop teleop controller."""

    def __init__(
        self,
        time_step: float,
        sphere_pid_gains: Dict[str, float],
        teleop: Dict[str, Any],
    ):
        super().__init__(time_step)
        self._meshcat = None
        self._sphere_pid_gains = sphere_pid_gains
        self._teleop_config = teleop
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

    def _setup_sphere_teleop(self, builder: DiagramBuilder) -> System:
        self._sim_duration = 5.0

        input_limit = self._teleop_config.input_limit
        step = self._teleop_config.step_size
        sphere_starting_translation = self._teleop_config.start_translation
        self._meshcat.AddSlider(
            "x",
            min=-input_limit,
            max=input_limit,
            step=step,
            value=sphere_starting_translation[0],
        )
        self._meshcat.AddSlider(
            "y",
            min=-input_limit,
            max=input_limit,
            step=step,
            value=sphere_starting_translation[1],
        )
        force_system = builder.AddSystem(MeshcatSliders(self._meshcat, ["x", "y"]))
        mux = builder.AddNamedSystem("teleop_mux", Multiplexer(2))
        builder.Connect(force_system.get_output_port(0), mux.get_input_port(0))
        builder.Connect(force_system.get_output_port(1), mux.get_input_port(1))

        # Add discrete derivative to command velocities.
        desired_state_source = builder.AddSystem(
            StateInterpolatorWithDiscreteDerivative(
                self._num_sphere_positions,
                self._time_step,
                suppress_initial_transient=True,
            )
        )
        desired_state_source.set_name("teleop_desired_state_source")
        builder.Connect(mux.get_output_port(), desired_state_source.get_input_port())
        return desired_state_source

    def setup(self, builder: DiagramBuilder, plant: MultibodyPlant, **kwargs) -> None:
        if self._meshcat is None:
            raise RuntimeError(
                "Need to call `add_meshcat` before calling `setup` of the teleop controller."
            )

        sphere_controller = self._setup_sphere_controller(builder, plant)
        teleop_state_source = self._setup_sphere_teleop(builder)
        builder.Connect(
            teleop_state_source.get_output_port(),
            sphere_controller.get_input_port_desired_state(),
        )

        self.sphere_controller = sphere_controller
