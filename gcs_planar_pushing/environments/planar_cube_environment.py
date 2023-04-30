import atexit
from typing import List
from gcs_planar_pushing.images.image_generator import ImageGenerator
import matplotlib.pyplot as plt
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

    def setup(self) -> None:
        self._meshcat = StartMeshcat()

        # Setup environment
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=self._time_step
        )
        parser = get_parser(plant)
        parser.AddAllModelsFromFile(self._scene_directive_path)
        plant.Finalize()

        AddRgbdSensors(builder, plant, scene_graph)

        # Setup controller
        self._controller.add_meshcat(self._meshcat)
        self._controller.setup(builder, plant)

        visualizer_params = MeshcatVisualizerParams()
        visualizer_params.role = Role.kIllustration
        self._visualizer = MeshcatVisualizer.AddToBuilder(
            builder,
            scene_graph,
            self._meshcat,
            visualizer_params,
        )

        diagram = builder.Build()

        self._simulator = Simulator(diagram)

        # Set initial cube position
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

        # This could be refactored to be cleaner
        self.plant = plant
        self.plant_context = plant_context
        self.robot = sphere
        self.object = box

        # Set up image generator
        self._image_generator = ImageGenerator(
            max_depth_range=10.0, diagram=diagram, scene_graph=scene_graph
        )

        # Test image generator
        # (
        #     rgb_image,
        #     depth_image,
        #     object_labels,
        #     masks,
        # ) = self._image_generator.get_camera_data(
        #     camera_name="camera0", context=self._simulator.get_context()
        # )

        # plt.imshow(rgb_image)
        # plt.show()

    def simulate(self) -> None:
        print("Press 'Stop Simulation' in MeshCat to continue.")

        self._visualizer.StartRecording()

        print(f"Meshcat URL: {self._meshcat.web_url()}")

        sim_duration = self._controller.get_sim_duration()
        for t in np.arange(0.0, sim_duration, self._time_step):
            self._simulator.AdvanceTo(t)

        self._visualizer.StopRecording()
        self._visualizer.PublishRecording()
