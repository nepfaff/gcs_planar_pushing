import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import (
    StartMeshcat,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    RigidTransform,
    RotationMatrix,
    MeshcatVisualizer,
    MultibodyPlant,
    InverseDynamicsController,
    Multiplexer,
    StateInterpolatorWithDiscreteDerivative,
    plot_system_graphviz,
    Simulator,
)
from underactuated.meshcat_utils import MeshcatSliders

meshcat = StartMeshcat()
time_step = 1e-3


def planar_pushing_demo():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    parser = Parser(plant)
    parser.package_map().AddPackageXml("./package.xml")
    file_name = "./gcs_planar_pushing/models/planar_pushing.dmd.yaml"
    parser.AddAllModelsFromFile(file_name)
    plant.Finalize()

    meshcat.Delete()
    X_WC = RigidTransform(RotationMatrix.MakeXRotation(-np.pi / 2), [0, 0, 3])
    # meshcat.Set2dRenderMode(X_WC=X_WC, xmin=-5, xmax=5, ymin=-5, ymax=5)
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    sphere_model_instance = plant.GetModelInstanceByName("sphere")

    # Sphere controller
    sphere_controller_plant = MultibodyPlant(time_step=time_step)
    sphere_controller_parser = Parser(sphere_controller_plant)
    sphere_controller_parser.package_map().AddPackageXml("./package.xml")
    sphere_controller_parser.AddModelsFromUrl(
        "package://gcs_planar_pushing/gcs_planar_pushing/models/actuated_sphere.urdf"
    )[0]
    sphere_controller_plant.set_name("sphere_controller_plant")
    sphere_controller_plant.Finalize()

    sphere_pid_gains = {"kp": 100, "kd": 20, "ki": 1}
    num_sphere_positions = 2

    sphere_controller = builder.AddSystem(
        InverseDynamicsController(
            sphere_controller_plant,
            kp=[sphere_pid_gains["kp"]] * num_sphere_positions,
            ki=[sphere_pid_gains["ki"]] * num_sphere_positions,
            kd=[sphere_pid_gains["kd"]] * num_sphere_positions,
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

    # Setup slider input
    input_limit = 10
    meshcat.AddSlider("x", min=-input_limit, max=input_limit, step=0.1, value=0.0)
    meshcat.AddSlider("y", min=-input_limit, max=input_limit, step=0.1, value=0.0)
    force_system = builder.AddSystem(MeshcatSliders(meshcat, ["x", "y"]))
    mux = builder.AddNamedSystem("Mux", Multiplexer(2))
    builder.Connect(force_system.get_output_port(0), mux.get_input_port(0))
    builder.Connect(force_system.get_output_port(1), mux.get_input_port(1))

    # Add discrete derivative to command velocities.
    desired_state_from_position = builder.AddSystem(
        StateInterpolatorWithDiscreteDerivative(
            num_sphere_positions, time_step, suppress_initial_transient=True
        )
    )
    desired_state_from_position.set_name("sphere_desired_state_from_position")
    builder.Connect(
        desired_state_from_position.get_output_port(),
        sphere_controller.get_input_port_desired_state(),
    )
    builder.Connect(mux.get_output_port(), desired_state_from_position.get_input_port())

    diagram = builder.Build()

    plt.figure(figsize=(20, 10))
    plot_system_graphviz(diagram)

    # Set up a simulator to run this diagram
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()

    plant_context = plant.GetMyMutableContextFromRoot(context)
    box = plant.GetBodyByName("box")
    plant.SetFreeBodyPose(
        plant_context, box, RigidTransform(RotationMatrix(), [1, 0.25, 0.5])
    )

    simulator.set_target_realtime_rate(1.0)

    print("Use the slider in the MeshCat controls to apply force to sphere.")
    print("Press 'Stop Simulation' in MeshCat to continue.")
    meshcat.AddButton("Stop Simulation")
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 1.0)

    meshcat.DeleteAddedControls()
    meshcat.Delete()


planar_pushing_demo()
