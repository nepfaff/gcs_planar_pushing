import atexit
from typing import List

from pydrake.all import (
    StartMeshcat,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    RigidTransform,
    RotationMatrix,
    MeshcatVisualizer,
    Simulator,
)

from .environment_base import EnvironmentBase
from gcs_planar_pushing.controllers import ControllerBase
from gcs_planar_pushing.utils import get_parser


class PlanarCubeEnvironment(EnvironmentBase):
    def __init__(
        self,
        controller: ControllerBase,
        time_step: float,
        scene_directive_path: str,
        initial_cube_translation: List[float],
    ):
        super().__init__(controller, time_step, scene_directive_path)
        atexit.register(self._cleanup)

        self._initial_cube_translation = initial_cube_translation

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

        MeshcatVisualizer.AddToBuilder(builder, scene_graph, self._meshcat)

        # Setup controller
        self._controller.add_meshcat(self._meshcat)
        self._controller.setup(builder, plant)

        diagram = builder.Build()

        self._simulator = Simulator(diagram)

        # Set initial cube position
        context = self._simulator.get_mutable_context()
        plant_context = plant.GetMyMutableContextFromRoot(context)
        box = plant.GetBodyByName("box")
        plant.SetFreeBodyPose(
            plant_context,
            box,
            RigidTransform(RotationMatrix(), self._initial_cube_translation),
        )

    def simulate(self) -> None:
        print("Use the slider in the MeshCat controls to apply force to sphere.")
        print("Press 'Stop Simulation' in MeshCat to continue.")
        self._meshcat.AddButton("Stop Simulation")
        while self._meshcat.GetButtonClicks("Stop Simulation") < 1:
            self._simulator.AdvanceTo(self._simulator.get_context().get_time() + 1.0)

    def _cleanup(self):
        self._meshcat.DeleteAddedControls()
        self._meshcat.Delete()
