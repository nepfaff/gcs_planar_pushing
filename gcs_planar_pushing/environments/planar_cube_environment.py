import atexit
import copy
from typing import List, Tuple
from gcs_planar_pushing.images.image_generator import ImageGenerator
from pydrake.all import (
    StartMeshcat,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    RigidTransform,
    RotationMatrix,
    MeshcatVisualizer,
    Simulator,
    MeshcatVisualizerParams,
    Role,
    LogVectorOutput,
    Rgba,
    Box,
)
import numpy as np
from gcs_planar_pushing.utils.util import AddRgbdSensors

from .environment_base import EnvironmentBase
from gcs_planar_pushing.controllers import ControllerBase
from gcs_planar_pushing.utils import get_parser


class PlanarCubeEnvironment(EnvironmentBase):
    def __init__(
        self,
        controller: ControllerBase,
        time_step: float,
        scene_directive_path: str,
        initial_box_position: List[float],
        initial_finger_position: List[float],
    ):
        super().__init__(controller, time_step, scene_directive_path)

        self._initial_box_position = initial_box_position
        self._initial_finger_position = initial_finger_position
        self._controller.set_initial_state(
            initial_box_position, initial_finger_position
        )

        self._meshcat = None
        self._simulator = None

    def setup(self, meshcat=None) -> None:
        if meshcat is None:
            self._meshcat = StartMeshcat()
        else:
            self._meshcat = meshcat

        # Setup environment
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=self._time_step
        )
        parser = get_parser(plant)
        parser.AddAllModelsFromFile(self._scene_directive_path)
        plant.Finalize()

        rgbd_sensors = AddRgbdSensors(builder, plant, scene_graph)

        assert (
            len(rgbd_sensors) == 1
        ), f"Expected a single camera but got {len(rgbd_sensors)}"
        rgbd_sensor = rgbd_sensors[0]

        # Setup controller
        self._controller.add_meshcat(self._meshcat)
        self._controller.setup(builder, plant, rgbd_sensor=rgbd_sensor)

        visualizer_params = MeshcatVisualizerParams()
        visualizer_params.role = Role.kIllustration
        self._visualizer = MeshcatVisualizer.AddToBuilder(
            builder,
            scene_graph,
            self._meshcat,
            visualizer_params,
        )

        # Set up loggers Note: Not being used to generate data, just for debugging
        self._state_logger = LogVectorOutput(plant.get_state_output_port(), builder)
        # self._action_logger = LogVectorOutput(
        #     self._controller.sphere_controller.get_output_port_control(), builder
        # )
        # self._image_logger = AbstractValueLogger(rgbd_sensors[0].color_image_output_port(), builder)

        diagram = builder.Build()

        self._simulator = Simulator(diagram)

        # Set initial cube and sphere position
        context = self._simulator.get_mutable_context()
        plant_context = plant.GetMyMutableContextFromRoot(context)
        box = plant.GetBodyByName("box")
        box_model_instance = box.model_instance()

        if box.is_floating():
            plant.SetFreeBodyPose(
                plant_context,
                box,
                RigidTransform(RotationMatrix(), self._initial_box_position),
            )
        else:
            plant.SetPositions(
                plant_context, box_model_instance, self._initial_box_position
            )

        sphere = plant.GetBodyByName("sphere_actuated")
        sphere_model_instance = sphere.model_instance()
        plant.SetPositions(
            plant_context, sphere_model_instance, self._initial_finger_position
        )

        # This could be refactored to be cleaner
        self.plant = plant
        self.plant_context = plant_context
        self.robot = sphere
        self.object = box

        # For manual logging
        self._diagram = diagram

        # Set up image generator
        self._image_generator = ImageGenerator(
            max_depth_range=10.0, diagram=diagram, scene_graph=scene_graph
        )

        # Add target # Doing it like this doesn't show up on the camera for some reason
        # self._draw_object("target", [0.0, 0.0])

    def _draw_object(
        self, name: str, x: np.array, color: Rgba = Rgba(0, 1, 0, 1.0)
    ) -> None:

        # Assumes x = [x, y]
        pose = RigidTransform(RotationMatrix(), [*x, 0])
        self._meshcat.SetObject(name, Box(1, 1, 0.3), rgba=color)
        self._meshcat.SetTransform(name, pose)

    def simulate(self) -> None:
        print("Press 'Stop Simulation' in MeshCat to continue.")

        self._visualizer.StartRecording()

        print(f"Meshcat URL: {self._meshcat.web_url()}")

        sim_duration = 5.0
        for t in np.arange(0.0, sim_duration, self._time_step):
            self._simulator.AdvanceTo(t)

        self._visualizer.StopRecording()
        self._visualizer.PublishRecording()

        context = self._simulator.get_mutable_context()
        action_log = self._action_logger.FindLog(context)
        state_log = self._state_logger.FindLog(context)
        self._plot_logs(state_log, action_log)

    def generate_data(self, n_data: int) -> Tuple[np.array, np.array, np.array]:
        print(f"Meshcat URL: {self._meshcat.web_url()}")
        sim_duration = self._controller.get_sim_duration()
        sample_times = np.linspace(0.0, sim_duration, n_data)
        n_data = len(sample_times)
        image_data = np.zeros((n_data, 96, 96, 3))
        state_data = np.zeros((n_data, 2))
        action_data = np.zeros((n_data, 2))  # finger_x, finger_y

        for i, t in enumerate(sample_times):
            context = self._simulator.get_context()
            self._simulator.AdvanceTo(t)
            (
                rgb_image,
                depth_image,
                object_labels,
                masks,
            ) = self._image_generator.get_camera_data(
                camera_name="camera0", context=context
            )
            image_data[i] = rgb_image
            state_data[i] = copy.deepcopy(
                self.plant.GetPositions(self.plant_context, self.robot.model_instance())
            )
            action_data[i] = copy.deepcopy(
                self._diagram.GetOutputPort("action").Eval(context)
            )

            # print(f"state: {state_data[i]}")
            # print(f"action: {action_data[i]}")
            # plt.imshow(rgb_image)
            # plt.show()

        return image_data, state_data, action_data

    # Not being used to generate data, only for debugging
    def _plot_logs(self, state_log, action_log) -> None:
        fig, axs = plt.subplots(3, 1, figsize=(16, 16))
        axis = axs[0]
        axis.step(state_log.sample_times(), state_log.data().transpose()[:, :4])
        axis.legend([r"$q_{bx}$", r"$q_{by}$", r"$q_{fx}$", r"$q_{fy}$"])
        axis.set_ylabel("state box")
        axis.set_xlabel("t")

        axis = axs[1]
        axis.step(state_log.sample_times(), state_log.data().transpose()[:, 4:])
        axis.legend([r"$v_{bx}$", r"$v_{by}$", r"$v_{fx}$", r"$v_{fy}$"])
        axis.set_ylabel("state finger")
        axis.set_xlabel("t")

        axis = axs[2]
        axis.step(action_log.sample_times(), action_log.data().transpose())
        axis.legend([r"$u_x$", r"$u_y$"])
        axis.set_ylabel("u")
        axis.set_xlabel("t")
        plt.show()
