import argparse
import numpy as np

from pydrake.common import FindResourceOrThrow

from plan_runner.manipulation_station_simulator import ManipulationStationSimulator
from plan_runner.manipulation_station_plan_runner import *
from plan_runner.open_left_door import (GenerateOpenLeftDoorPlansByTrajectory,
                                        GenerateOpenLeftDoorPlansByImpedanceOrPosition,)
from perception.estimate_door_angle import GetDoorPose, get_door_angle

if __name__ == '__main__':
    # python run_open_left_door.py --controller=Position --left_door_angle_actual 0.2 --left_door_angle_guess 0.2 --open_fully
    # define command line arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hardware", action='store_true',
        help="Use the ManipulationStationHardwareInterface instead of an "
             "in-process simulation.")
    parser.add_argument(
        "--open_fully", action='store_true',
        help="Add additional plans to fully open the door after impedance/position plans.")
    parser.add_argument(
        "-c", "--controller", type=str, default="Impedance",
        choices=["Trajectory", "Impedance", "Position"],
        help="Specify the controller used to open the door. Its value should be: "
             "'Trajectory' (default), 'Impedance' or 'Position.")
    parser.add_argument(
        "--left_door_angle",
        type=float,
        help="Angle of left door, from 0 to pi/2",
        default=0.001)
    parser.add_argument(
        "--right_door_angle",
        type=float,
        help="Angle of right door, from 0 to pi/2",
        default=0.001)
    parser.add_argument(
        "--no_visualization", action="store_true", default=False,
        help="Turns off visualization")
    parser.add_argument(
        "--diagram_plan_runner", action="store_true", default=False,
        help="Use the diagram version of plan_runner")
    parser.add_argument(
        "--config_file",
        required=True,
        help="The path to a .yml camera config file")
    parser.add_argument(
        "--num_trials",
        type=int,
        help="set number of point cloud trials to run",
        default=1)
    args = parser.parse_args()
    is_hardware = args.hardware

    isometries = GetDoorPose(args.config_file,
                             viz=not args.no_visualization,
                             left_door_angle=args.left_door_angle,
                             right_door_angle=args.right_door_angle,
                             num_trials=args.num_trials)
    estimated_left_door_angle = sum(get_door_angle(left_door_pose) for left_door_pose in isometries["left_door"]) / args.num_trials
    estimated_right_door_angle = sum(get_door_angle(right_door_pose) for right_door_pose in isometries["right_door"]) / args.num_trials

    print "Estimated left door angle: " + str(estimated_left_door_angle)
    print "Estimated right door angle: " + str(estimated_right_door_angle)

    left_door_angle = -estimated_left_door_angle
    rotation_matrix = np.array([[np.cos(left_door_angle), -np.sin(left_door_angle), 0], [np.sin(left_door_angle), np.cos(left_door_angle), 0], [0, 0, 1]])
    p_handle_2_hinge_new = np.dot(rotation_matrix, p_handle_2_hinge)
    p_LC_handle_new = p_handle_2_hinge_new + p_LC_left_hinge
    p_WC_handle_new = p_WL + p_LC_handle_new
    theta0_hinge_new = np.arctan2(np.abs(p_handle_2_hinge_new[0]),
                              np.abs(p_handle_2_hinge_new[1]))  # = theta0_hinge - left_door_angle

    # Construct simulator system.
    object_file_path = FindResourceOrThrow(
        "drake/examples/manipulation_station/models/061_foam_brick.sdf")

    manip_station_sim = ManipulationStationSimulator(
        time_step=2e-3,
        object_file_path=object_file_path,
        object_base_link_name="base_link",)

    # Generate plans.
    plan_list = None
    gripper_setpoint_list = None
    if args.controller == "Trajectory":
        plan_list, gripper_setpoint_list = GenerateOpenLeftDoorPlansByTrajectory()
    elif args.controller == "Impedance" or args.controller == "Position":
        plan_list, gripper_setpoint_list = GenerateOpenLeftDoorPlansByImpedanceOrPosition(
            open_door_method=args.controller, is_open_fully=args.open_fully, handle_angle_start=theta0_hinge_new, handle_position=p_WC_handle_new)

    print type(gripper_setpoint_list), gripper_setpoint_list

    # Run simulator (simulation or hardware).
    if is_hardware:
        iiwa_position_command_log, iiwa_position_measured_log, iiwa_external_torque_log = \
            manip_station_sim.RunRealRobot(plan_list, gripper_setpoint_list, is_plan_runner_diagram=args.diagram_plan_runner)
        PlotExternalTorqueLog(iiwa_external_torque_log)
        PlotIiwaPositionLog(iiwa_position_command_log, iiwa_position_measured_log)
    else:
        q0 = [0, 0, 0, -1.75, 0, 1.0, 0]
        iiwa_position_command_log, iiwa_position_measured_log, iiwa_external_torque_log, \
            state_log = manip_station_sim.RunSimulation(
                plan_list, gripper_setpoint_list, extra_time=2.0, real_time_rate=1.0, q0_kuka=q0,
                is_visualizing=not args.no_visualization, left_door_angle=args.left_door_angle, right_door_angle=args.right_door_angle,
                is_plan_runner_diagram=args.diagram_plan_runner)
        PlotExternalTorqueLog(iiwa_external_torque_log)
        PlotIiwaPositionLog(iiwa_position_command_log, iiwa_position_measured_log)
