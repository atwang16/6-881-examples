import argparse

import numpy as np

from pydrake.common import FindResourceOrThrow
from pydrake.examples.manipulation_station import (
    ManipulationStation, ManipulationStationHardwareInterface)
from pydrake.multibody.multibody_tree.parsing import AddModelFromSdfFile
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.common.eigen_geometry import Isometry3
import pydrake.perception as mut

import meshcat.transformations as tf
from perception_tools.optimization_based_point_cloud_registration import (
    AlignSceneToModel)
from perception_tools.visualization_utils import ThresholdArray
from point_cloud_to_pose_system import PointCloudToPoseSystem
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors

# # L: frame the cupboard left door, whose origin is at the center of the door body.
# p_WL = np.array([0.7477, 0.1445, 0.4148]) #+ [-0.1, 0, 0]
# # center of the left hinge of the door in frame L and W
# p_LC_left_hinge = np.array([0.008, 0.1395, 0])
# p_WC_left_hinge = p_WL + p_LC_left_hinge

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def SegmentDoorSimulationLeft(scene_points, scene_colors):
    return SegmentDoorSimulation(scene_points, scene_colors, which_door="left")


def SegmentDoorSimulationRight(scene_points, scene_colors):
    return SegmentDoorSimulation(scene_points, scene_colors, which_door="right")


def SegmentDoorRealWorldLeft(scene_points, scene_colors):
    return SegmentDoorRealWorld(scene_points, scene_colors, which_door="left")


def SegmentDoorRealWorldRight(scene_points, scene_colors):
    return SegmentDoorRealWorld(scene_points, scene_colors, which_door="right")


def SegmentDoorSimulation(scene_points, scene_colors, which_door="left", center=False):
    """Removes all points that aren't a part of the foam brick.

    @param scene_points An Nx3 numpy array representing a scene.
    @param scene_colors An Nx3 numpy array of rgb values corresponding to the
        points in scene_points.

    @return door_points An Mx3 numpy array of points in the door.
    @return door_colors An Mx3 numpy array of the colors of the door points.
    """

    x_min = 0.4
    x_max = 0.75

    if which_door == "left":
        y_min = 0.0
        y_max = 0.5
    elif which_door == "right":
        y_min = -0.5
        y_max = 0.0
    else:
        raise ValueError(which_door + " is not valid. Must be \"left\" or \"right\"")

    z_min = 0.1
    z_max = 0.9

    x_indices = ThresholdArray(scene_points[:, 0], x_min, x_max)
    left_y_indices = ThresholdArray(scene_points[:, 1], y_min, y_max)
    # right_y_indices = ThresholdArray(scene_points[:, 1], y_min, y_center)
    z_indices = ThresholdArray(scene_points[:, 2], z_min, z_max)

    left_indices = reduce(np.intersect1d, (x_indices, left_y_indices, z_indices))
    # right_indices = reduce(np.intersect1d, (x_indices, right_y_indices, z_indices))

    table_points = scene_points[left_indices, :]
    table_colors = scene_colors[left_indices, :]

    # get only red points of door
    r_min = -1
    r_max = 1

    g_min = -1
    g_max = 0.25

    b_min = -1
    b_max = 0.25

    r_indices = ThresholdArray(table_colors[:, 0], r_min, r_max)
    g_indices = ThresholdArray(table_colors[:, 1], g_min, g_max)
    b_indices = ThresholdArray(table_colors[:, 2], b_min, b_max)

    indices = reduce(np.intersect1d, (r_indices, g_indices, b_indices))

    door_points = table_points[indices, :]
    door_colors = table_colors[indices, :]

    if center:
        x_mean = np.mean(door_points[:, 0])
        y_mean = np.mean(door_points[:, 1])

        door_points[:, 0] -= x_mean
        door_points[:, 1] -= y_mean

    door_points_x_y = door_points[:, :2]
    nbrs = NearestNeighbors(n_neighbors=60, algorithm='kd_tree', metric="euclidean").fit(door_points_x_y)
    distances, indices = nbrs.kneighbors(door_points_x_y)
    indices = np.where(distances[:, -1] <= 0.001)

    return door_points[list(indices[0]), :], door_colors[list(indices[0]), :]
    # return door_points, door_colors

def SegmentDoorRealWorld(scene_points, scene_colors, which_door="left", center=False):
    """Removes all points that aren't a part of the foam brick.

    @param scene_points An Nx3 numpy array representing a scene.
    @param scene_colors An Nx3 numpy array of rgb values corresponding to the
        points in scene_points.

    @return door_points An Mx3 numpy array of points in the door.
    @return door_colors An Mx3 numpy array of the colors of the door points.
    """
    x_min = 0.4
    x_max = 0.8

    if which_door == "left":
        y_min = 0.0
        y_max = 0.5
    elif which_door == "right":
        y_min = -0.5
        y_max = 0.0
    else:
        raise ValueError(which_door + " is not valid. Must be \"left\" or \"right\"")

    z_min = 0.1
    z_max = 0.9

    x_indices = ThresholdArray(scene_points[:, 0], x_min, x_max)
    left_y_indices = ThresholdArray(scene_points[:, 1], y_min, y_max)
    # right_y_indices = ThresholdArray(scene_points[:, 1], y_min, y_center)
    z_indices = ThresholdArray(scene_points[:, 2], z_min, z_max)

    left_indices = reduce(np.intersect1d, (x_indices, left_y_indices, z_indices))
    # right_indices = reduce(np.intersect1d, (x_indices, right_y_indices, z_indices))

    left_table_points = scene_points[left_indices, :]
    left_table_colors = scene_colors[left_indices, :]

    # get only red points of door
    r_min = 0.4
    r_max = 1

    g_min = 0.4
    g_max = 1

    b_min = 0.4
    b_max = 1

    r_indices = ThresholdArray(left_table_colors[:, 0], r_min, r_max)
    g_indices = ThresholdArray(left_table_colors[:, 1], g_min, g_max)
    b_indices = ThresholdArray(left_table_colors[:, 2], b_min, b_max)

    indices = reduce(np.intersect1d, (r_indices, g_indices, b_indices))

    left_door_points = left_table_points[indices, :]
    left_door_colors = left_table_colors[indices, :]

    if center:
        x_mean = np.mean(left_door_points[:, 0])
        y_mean = np.mean(left_door_points[:, 1])

        left_door_points[:, 0] -= x_mean
        left_door_points[:, 1] -= y_mean

    left_door_points_x_y = left_door_points[:, :2]
    nbrs = NearestNeighbors(n_neighbors=80, algorithm='kd_tree', metric="euclidean").fit(left_door_points_x_y)
    distances, indices = nbrs.kneighbors(left_door_points_x_y)
    indices = np.where(distances[:, -1] <= 0.001)

    return left_door_points[list(indices[0]), :], left_door_colors[list(indices[0]), :]


def ComputeLeftDoorPose(door_points, door_colors):
    return ComputeDoorPose(door_points, door_colors, model_path="models/left_door_model.npy")


def ComputeRightDoorPose(door_points, door_colors):
    transformation = ComputeDoorPose(door_points, door_colors, model_path="models/right_door_model.npy")
    transformation[0, 1] *= -1
    transformation[1, 0] *= -1
    return transformation


def ComputeDoorPose(door_points, door_colors, model_path):
    """Finds a good 4x4 pose of the brick from the segmented points.

    @param door_points An Nx3 numpy array of brick points.
    @param door_colors An Nx3 numpy array of corresponding brick colors.

    @return X_MS A 4x4 numpy array of the best-fit brick pose.
    """
    model_door = np.load(model_path)

    num_sample_points = min(door_points.shape[0], 250)
    X_MS, error = AlignSceneToModel(
        door_points, model_door, num_sample_points=num_sample_points)

    # if the best fit matrix isn't exactly an Isometry3, fix it
    try:
        Isometry3(X_MS)
    except:
        # make valid Isometry3
        sin_th = X_MS[1, 0]
        cos_th = X_MS[0, 0]

        if sin_th > 0:
            theta = np.arccos(np.clip(cos_th, -1.0, 1.0))
        else:
            theta = -np.arccos(np.clip(cos_th, -1.0, 1.0))

        X_MS[0, 0] = np.cos(theta)
        X_MS[1, 1] = np.cos(theta)
        X_MS[0, 1] = -np.sin(theta)
        X_MS[1, 0] = np.sin(theta)

    return X_MS

    # reg = LinearRegression().fit(door_points[:, :2], door_points[:, 2])
    # print reg.coef_
    # print reg.intercept_
    #
    # return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def build_system(builder, config_file, viz):
    # create the PointCloudToPoseSystem
    if "sim.yml" in config_file:
        pc_to_pose_left = builder.AddSystem(PointCloudToPoseSystem(
            config_file, viz, SegmentDoorSimulationLeft, ComputeLeftDoorPose, name="left_door"))
        pc_to_pose_right = builder.AddSystem(PointCloudToPoseSystem(
            config_file, viz, SegmentDoorSimulationRight, ComputeRightDoorPose, name="right_door"))
    elif "station_1.yml" in config_file or "station_2.yml" in config_file:
        pc_to_pose_left = builder.AddSystem(PointCloudToPoseSystem(
            config_file, viz, SegmentDoorRealWorldLeft, ComputeLeftDoorPose, name="left_door"))
        pc_to_pose_right = builder.AddSystem(PointCloudToPoseSystem(
            config_file, viz, SegmentDoorRealWorldRight, ComputeRightDoorPose, name="right_door"))
    else:
        raise ValueError("Could not detect whether config file is a simulation or station config.")

    # realsense serial numbers are >> 100
    use_hardware = int(pc_to_pose_left.camera_configs["left_camera_serial"]) > 100

    if use_hardware:
        camera_ids = [
            pc_to_pose_left.camera_configs["left_camera_serial"],
            pc_to_pose_left.camera_configs["middle_camera_serial"],
            pc_to_pose_left.camera_configs["right_camera_serial"]]
        station = builder.AddSystem(
            ManipulationStationHardwareInterface(camera_ids))
        station.Connect()
    else:
        station = builder.AddSystem(ManipulationStation())
        station.AddCupboard()
        object_file_path = \
            "drake/examples/manipulation_station/models/061_foam_brick.sdf"
        brick = AddModelFromSdfFile(
            FindResourceOrThrow(object_file_path),
            station.get_mutable_multibody_plant(),
            station.get_mutable_scene_graph())
        station.Finalize()

    # add systems to convert the depth images from ManipulationStation to
    # PointClouds
    left_camera_info = pc_to_pose_left.camera_configs["left_camera_info"]
    middle_camera_info = pc_to_pose_left.camera_configs["middle_camera_info"]
    right_camera_info = pc_to_pose_left.camera_configs["right_camera_info"]

    left_dut = builder.AddSystem(
        mut.DepthImageToPointCloud(camera_info=left_camera_info))
    middle_dut = builder.AddSystem(
        mut.DepthImageToPointCloud(camera_info=middle_camera_info))
    right_dut = builder.AddSystem(
        mut.DepthImageToPointCloud(camera_info=right_camera_info))

    left_name_prefix = \
        "camera_" + pc_to_pose_left.camera_configs["left_camera_serial"]
    middle_name_prefix = \
        "camera_" + pc_to_pose_left.camera_configs["middle_camera_serial"]
    right_name_prefix = \
        "camera_" + pc_to_pose_left.camera_configs["right_camera_serial"]

    # connect the depth images to the DepthImageToPointCloud converters
    builder.Connect(station.GetOutputPort(left_name_prefix + "_depth_image"),
                    left_dut.depth_image_input_port())
    builder.Connect(station.GetOutputPort(middle_name_prefix + "_depth_image"),
                    middle_dut.depth_image_input_port())
    builder.Connect(station.GetOutputPort(right_name_prefix + "_depth_image"),
                    right_dut.depth_image_input_port())

    for pc_to_pose in [pc_to_pose_left, pc_to_pose_right]:
        # connect the rgb images to the PointCloudToPoseSystem
        builder.Connect(station.GetOutputPort(left_name_prefix + "_rgb_image"),
                        pc_to_pose.GetInputPort("camera_left_rgb"))
        builder.Connect(station.GetOutputPort(middle_name_prefix + "_rgb_image"),
                        pc_to_pose.GetInputPort("camera_middle_rgb"))
        builder.Connect(station.GetOutputPort(right_name_prefix + "_rgb_image"),
                        pc_to_pose.GetInputPort("camera_right_rgb"))

        # connect the XYZ point clouds to PointCloudToPoseSystem
        builder.Connect(left_dut.point_cloud_output_port(),
                        pc_to_pose.GetInputPort("left_point_cloud"))
        builder.Connect(middle_dut.point_cloud_output_port(),
                        pc_to_pose.GetInputPort("middle_point_cloud"))
        builder.Connect(right_dut.point_cloud_output_port(),
                        pc_to_pose.GetInputPort("right_point_cloud"))

    return station, pc_to_pose_left, pc_to_pose_right, brick, use_hardware

def GetDoorPose(config_file, viz=False, left_door_angle=0.0, right_door_angle=0.0):
    """Estimates the pose of the foam brick in a ManipulationStation setup.

    @param config_file str. The path to a camera configuration file.
    @param viz bool. If True, save point clouds to numpy arrays.

    @return An Isometry3 representing the pose of the door.
    """
    builder = DiagramBuilder()

    station, pc_to_pose_left, pc_to_pose_right, brick, use_hardware = build_system(builder, config_file, viz)

    diagram = builder.Build()

    simulator = Simulator(diagram)

    if not use_hardware:
        X_WObject = Isometry3.Identity()
        X_WObject.set_translation([.6, 0, 0])
        station_context = diagram.GetMutableSubsystemContext(
            station, simulator.get_mutable_context())
        station.get_mutable_multibody_plant().tree().SetFreeBodyPoseOrThrow(
            station.get_mutable_multibody_plant().GetBodyByName(
                "base_link", brick),
                X_WObject,
                station.GetMutableSubsystemContext(
                    station.get_mutable_multibody_plant(), station_context))

        left_hinge_joint = station.get_mutable_multibody_plant().GetJointByName("left_door_hinge")
        left_hinge_joint.set_angle(
            station.GetMutableSubsystemContext(
                station.get_mutable_multibody_plant(), station_context),
            -left_door_angle)

        right_hinge_joint = station.get_mutable_multibody_plant().GetJointByName("right_door_hinge")
        right_hinge_joint.set_angle(
            station.GetMutableSubsystemContext(
                station.get_mutable_multibody_plant(), station_context),
            right_door_angle)

    left_context = diagram.GetMutableSubsystemContext(pc_to_pose_left, simulator.get_mutable_context())
    right_context = diagram.GetMutableSubsystemContext(pc_to_pose_right, simulator.get_mutable_context())

    # returns the pose of the brick, of type Isometry3
    return pc_to_pose_left.GetOutputPort("X_WObject").Eval(left_context), \
           pc_to_pose_right.GetOutputPort("X_WObject").Eval(right_context)
    # return pc_to_pose_right.GetOutputPort("X_WObject").Eval(right_context)


def get_door_angle(door_pose):
    sin_th = door_pose.rotation()[1, 0]
    cos_th = door_pose.rotation()[0, 0]

    if sin_th > 0:
        theta = np.arccos(np.clip(cos_th, -1.0, 1.0))
    else:
        theta = -np.arccos(np.clip(cos_th, -1.0, 1.0))

    if theta < -np.pi / 4:
        theta += np.pi
    elif theta > 3 * np.pi / 4:
        theta -= np.pi

    return theta


def compute_error(actual, guess):
    diff = actual - guess
    if diff < -np.pi:
        diff += 2 * np.pi
    elif diff > np.pi:
        diff -= 2 * np.pi
    return diff


if __name__ == "__main__":
    # DISPLAY=:100 python estimate_door_angle.py --config_file config/sim.yml --viz --left_door_angle 0.123
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
      "--config_file",
      required=True,
      help="The path to a .yml camera config file")
    parser.add_argument(
      "--viz",
      action="store_true",
      help="Save the aligned and segmented point clouds for visualization")
    parser.add_argument(
        "--left_door_angle",
        type=float,
        help="Angle of left door, from 0 to pi/2",
        default=0.0)
    parser.add_argument(
        "--right_door_angle",
        type=float,
        help="Angle of right door, from 0 to pi/2",
        default=0.0)
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test a range of angles",
        default=0.0)
    args = parser.parse_args()

    if args.test:
        with open("angle_tests.csv", "w") as f:
            f.write("angle,left_estimate,right_estimate,left_error,right_error\n")
            for angle in [i * np.pi/24 for i in range(25)]:
                print "Testing angle", angle
                left_door_pose, right_door_pose = GetDoorPose(args.config_file, args.viz,
                                                              left_door_angle=angle,
                                                              right_door_angle=angle)
                estimated_left_door_angle = get_door_angle(left_door_pose)
                estimated_right_door_angle = get_door_angle(right_door_pose)
                left_error = compute_error(angle, estimated_left_door_angle)
                right_error = compute_error(angle, estimated_right_door_angle)
                f.write(str(angle) + "," + str(estimated_left_door_angle) + "," + str(estimated_right_door_angle) +
                        "," + str(left_error) + "," + str(right_error) + "\n")
    else:
        left_door_pose, right_door_pose = GetDoorPose(args.config_file, args.viz,
                                                      left_door_angle=args.left_door_angle,
                                                      right_door_angle=args.right_door_angle)
        print "Left door pose:\n" + str(left_door_pose)
        print "Right door pose:\n" + str(right_door_pose)
        estimated_left_door_angle = get_door_angle(left_door_pose)
        estimated_right_door_angle = get_door_angle(right_door_pose)
        print "Estimated left door angle: " + str(estimated_left_door_angle)
        print "Estimated right door angle: " + str(estimated_right_door_angle)
        print "Left Door Error (simulation): " + str(compute_error(args.left_door_angle, estimated_left_door_angle))
        print "Right Door Error (simulation): " + str(compute_error(args.right_door_angle, estimated_right_door_angle))
