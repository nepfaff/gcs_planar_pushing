import os
import sys
from typing import List

from manipulation.utils import AddPackagePaths
from pydrake.all import (
    AbstractValue,
    BaseField,
    CameraInfo,
    ClippingRange,
    DepthImageToPointCloud,
    DepthRange,
    DepthRenderCamera,
    LeafSystem,
    MakeRenderEngineVtk,
    ModelInstanceIndex,
    MultibodyPlant,
    Parser,
    RenderCameraCore,
    RenderEngineVtkParams,
    RgbdSensor,
    RigidTransform,
    MultibodyPlant,
    Parser,
)


def get_parser(plant: MultibodyPlant) -> Parser:
    """Creates a parser for a plant and adds package paths to it."""
    parser = Parser(plant)
    AddPackagePaths(parser)
    parser.package_map().AddPackageXml(os.path.abspath("package.xml"))
    return parser


def AddRgbdSensors(
    builder,
    plant,
    scene_graph,
    also_add_point_clouds=True,
    model_instance_prefix="camera",
    depth_camera=None,
    renderer=None,
) -> List[RgbdSensor]:
    """
    Adds a RgbdSensor to the first body in the plant for every model instance
    with a name starting with model_instance_prefix.  If depth_camera is None,
    then a default camera info will be used.  If renderer is None, then we will
    assume the name 'my_renderer', and create a VTK renderer if a renderer of
    that name doesn't exist.
    """
    if sys.platform == "linux" and os.getenv("DISPLAY") is None:
        from pyvirtualdisplay import Display

        virtual_display = Display(visible=0, size=(1400, 900))
        virtual_display.start()

    if not renderer:
        renderer = "my_renderer"

    if not scene_graph.HasRenderer(renderer):
        scene_graph.AddRenderer(renderer, MakeRenderEngineVtk(RenderEngineVtkParams()))

    if not depth_camera:
        depth_camera = DepthRenderCamera(
            RenderCameraCore(
                renderer,
                CameraInfo(
                    width=480,
                    height=480,
                    focal_x=10000,
                    focal_y=10000,
                    center_x=239.5,
                    center_y=239.5,
                ),
                ClippingRange(near=0.1, far=150.0),
                RigidTransform(),
            ),
            DepthRange(0.1, 10.0),
        )

    rgbd_sensors = []
    for index in range(plant.num_model_instances()):
        model_instance_index = ModelInstanceIndex(index)
        model_name = plant.GetModelInstanceName(model_instance_index)

        if model_name.startswith(model_instance_prefix):
            body_index = plant.GetBodyIndices(model_instance_index)[0]
            rgbd = builder.AddSystem(
                RgbdSensor(
                    parent_id=plant.GetBodyFrameIdOrThrow(body_index),
                    X_PB=RigidTransform(),
                    depth_camera=depth_camera,
                    show_window=False,
                )
            )
            rgbd.set_name(model_name)
            rgbd_sensors.append(rgbd)

            builder.Connect(
                scene_graph.get_query_output_port(),
                rgbd.query_object_input_port(),
            )

            # Export the camera outputs
            builder.ExportOutput(
                rgbd.color_image_output_port(), f"{model_name}_rgb_image"
            )
            builder.ExportOutput(
                rgbd.depth_image_32F_output_port(), f"{model_name}_depth_image"
            )
            builder.ExportOutput(
                rgbd.label_image_output_port(), f"{model_name}_label_image"
            )

            if also_add_point_clouds:
                # Add a system to convert the camera output into a point cloud
                to_point_cloud = builder.AddSystem(
                    DepthImageToPointCloud(
                        camera_info=rgbd.depth_camera_info(),
                        fields=BaseField.kXYZs | BaseField.kRGBs,
                    )
                )
                builder.Connect(
                    rgbd.depth_image_32F_output_port(),
                    to_point_cloud.depth_image_input_port(),
                )
                builder.Connect(
                    rgbd.color_image_output_port(),
                    to_point_cloud.color_image_input_port(),
                )

                class ExtractBodyPose(LeafSystem):
                    def __init__(self, body_index):
                        LeafSystem.__init__(self)
                        self.body_index = body_index
                        self.DeclareAbstractInputPort(
                            "poses",
                            plant.get_body_poses_output_port().Allocate(),
                        )
                        self.DeclareAbstractOutputPort(
                            "pose",
                            lambda: AbstractValue.Make(RigidTransform()),
                            self.CalcOutput,
                        )

                    def CalcOutput(self, context, output):
                        poses = self.EvalAbstractInput(context, 0).get_value()
                        pose = poses[int(self.body_index)]
                        output.get_mutable_value().set(
                            pose.rotation(), pose.translation()
                        )

                camera_pose = builder.AddSystem(ExtractBodyPose(body_index))
                builder.Connect(
                    plant.get_body_poses_output_port(),
                    camera_pose.get_input_port(),
                )
                builder.Connect(
                    camera_pose.get_output_port(),
                    to_point_cloud.GetInputPort("camera_pose"),
                )

                # Export the point cloud output.
                builder.ExportOutput(
                    to_point_cloud.point_cloud_output_port(),
                    f"{model_name}_point_cloud",
                )
    return rgbd_sensors
