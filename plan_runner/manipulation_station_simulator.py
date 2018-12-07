import numpy as np
import time

from pydrake.examples.manipulation_station import (ManipulationStation,
                                    ManipulationStationHardwareInterface)
from pydrake.geometry import SceneGraph
from pydrake.multibody.multibody_tree.parsing import AddModelFromSdfFile
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator
from pydrake.common.eigen_geometry import Isometry3
from pydrake.systems.primitives import Demultiplexer, LogOutput


from underactuated.meshcat_visualizer import MeshcatVisualizer
# from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from plan_runner.manipulation_station_plan_runner import ManipStationPlanRunner
from plan_runner.manipulation_station_plan_runner_diagram import CreateManipStationPlanRunnerDiagram
from plan_runner.plan_utils import *
from perception import estimate_door_angle

X_WObject_default = Isometry3.Identity()
X_WObject_default.set_translation([.6, 0, 0])


class ManipulationStationSimulator:
    def __init__(self, time_step,
                 object_file_path=None,
                 object_base_link_name=None,
                 X_WObject=X_WObject_default,):
        self.object_base_link_name = object_base_link_name
        self.time_step = time_step

        # Finalize manipulation station by adding manipuland.
        self.station = ManipulationStation(self.time_step)
        self.station.AddCupboard()
        self.plant = self.station.get_mutable_multibody_plant()
        if object_file_path is not None:
            self.object = AddModelFromSdfFile(
                file_name=object_file_path,
                model_name="object",
                plant=self.station.get_mutable_multibody_plant(),
                scene_graph=self.station.get_mutable_scene_graph() )
        self.station.Finalize()

        self.simulator = None
        self.plan_runner = None

        # Initial pose of the object
        self.X_WObject = X_WObject

    def RunSimulation(self, plan_list, gripper_setpoint_list,
                      extra_time=0, real_time_rate=1.0, q0_kuka=np.zeros(7), is_visualizing=True, sim_duration=None,
                      is_plan_runner_diagram=False):
        """
        Constructs a Diagram that sends commands to ManipulationStation.
        @param plan_list: A list of Plans to be executed.
        @param gripper_setpoint_list: A list of gripper setpoints. Each setpoint corresponds to a Plan.
        @param extra_time: the amount of time for which commands are sent,
            in addition to the sum of the durations of all plans.
        @param real_time_rate: 1.0 means realtime; 0 means as fast as possible.
        @param q0_kuka: initial configuration of the robot.
        @param is_visualizing: if true, adds MeshcatVisualizer to the Diagram. It should be set to False
            when running tests.
        @param sim_duration: the duration of simulation in seconds. If unset, it is set to the sum of the durations of all
            plans in plan_list plus extra_time.
        @param is_plan_runner_diagram: True: use the diagram version of PlanRunner; False: use the leaf version
            of PlanRunner.
        @return: logs of robot configuration and MultibodyPlant, generated by simulation.
            Logs are SignalLogger systems, whose data can be accessed by SignalLogger.data().
        """
        builder = DiagramBuilder()
        builder.AddSystem(self.station)

        # Add plan runner.
        if is_plan_runner_diagram:
            plan_runner = CreateManipStationPlanRunnerDiagram(
                station=self.station,
                kuka_plans=plan_list,
                gripper_setpoint_list=gripper_setpoint_list)
        else:
            plan_runner = ManipStationPlanRunner(
                station=self.station,
                kuka_plans=plan_list,
                gripper_setpoint_list=gripper_setpoint_list)
        self.plan_runner = plan_runner

        builder.AddSystem(plan_runner)
        builder.Connect(plan_runner.GetOutputPort("gripper_setpoint"),
                        self.station.GetInputPort("wsg_position"))
        builder.Connect(plan_runner.GetOutputPort("force_limit"),
                        self.station.GetInputPort("wsg_force_limit"))


        demux = builder.AddSystem(Demultiplexer(14, 7))
        builder.Connect(
            plan_runner.GetOutputPort("iiwa_position_and_torque_command"),
            demux.get_input_port(0))
        builder.Connect(demux.get_output_port(0),
                        self.station.GetInputPort("iiwa_position"))
        builder.Connect(demux.get_output_port(1),
                        self.station.GetInputPort("iiwa_feedforward_torque"))
        builder.Connect(self.station.GetOutputPort("iiwa_position_measured"),
                        plan_runner.GetInputPort("iiwa_position"))
        builder.Connect(self.station.GetOutputPort("iiwa_velocity_estimated"),
                        plan_runner.GetInputPort("iiwa_velocity"))

        # Add meshcat visualizer
        if is_visualizing:
            scene_graph = self.station.get_mutable_scene_graph()
            viz = MeshcatVisualizer(scene_graph,
                                    is_drawing_contact_force = True,
                                    plant = self.plant)
            builder.AddSystem(viz)
            builder.Connect(self.station.GetOutputPort("pose_bundle"),
                            viz.GetInputPort("lcm_visualization"))
            builder.Connect(self.station.GetOutputPort("contact_results"),
                            viz.GetInputPort("contact_results"))

        # Add logger
        iiwa_position_command_log = LogOutput(demux.get_output_port(0), builder)
        iiwa_position_command_log._DeclarePeriodicPublish(0.005)

        iiwa_external_torque_log = LogOutput(
            self.station.GetOutputPort("iiwa_torque_external"), builder)
        iiwa_external_torque_log._DeclarePeriodicPublish(0.005)

        iiwa_position_measured_log = LogOutput(
            self.station.GetOutputPort("iiwa_position_measured"), builder)
        iiwa_position_measured_log._DeclarePeriodicPublish(0.005)

        plant_state_log = LogOutput(
            self.station.GetOutputPort("plant_continuous_state"), builder)
        plant_state_log._DeclarePeriodicPublish(0.005)

        # build diagram
        diagram = builder.Build()
        if is_visualizing:
            viz.load()
            time.sleep(2.0)
            RenderSystemWithGraphviz(diagram)

        # construct simulator
        simulator = Simulator(diagram)
        self.simulator = simulator

        context = diagram.GetMutableSubsystemContext(
            self.station, simulator.get_mutable_context())

        # set initial state of the robot
        self.station.SetIiwaPosition(q0_kuka, context)
        self.station.SetIiwaVelocity(np.zeros(7), context)
        self.station.SetWsgPosition(0.05, context)
        self.station.SetWsgVelocity(0, context)

        # set initial hinge angles of the cupboard.
        # setting hinge angle to exactly 0 or 90 degrees will result in intermittent contact
        # with small contact forces between the door and the cupboard body.
        left_hinge_joint = self.plant.GetJointByName("left_door_hinge")
        left_hinge_joint.set_angle(
            context=self.station.GetMutableSubsystemContext(self.plant, context), angle=-0.001)

        right_hinge_joint = self.plant.GetJointByName("right_door_hinge")
        right_hinge_joint.set_angle(
            context=self.station.GetMutableSubsystemContext(self.plant, context), angle=0.001)

        # set initial pose of the object
        if self.object_base_link_name is not None:
            self.plant.tree().SetFreeBodyPoseOrThrow(
               self.plant.GetBodyByName(self.object_base_link_name, self.object),
                self.X_WObject, self.station.GetMutableSubsystemContext(self.plant, context))

        simulator.set_publish_every_time_step(False)
        simulator.set_target_realtime_rate(real_time_rate)

        # calculate starting time for all plans.
        t_plan = GetPlanStartingTimes(plan_list)
        if sim_duration is None:
            sim_duration = t_plan[-1] + extra_time
        print "simulation duration", sim_duration
        simulator.Initialize()
        simulator.StepTo(sim_duration)

        return iiwa_position_command_log, iiwa_position_measured_log, \
            iiwa_external_torque_log, plant_state_log

    def RunRealRobot(self, plan_list, gripper_setpoint_list, sim_duration=None, extra_time=2.0,
                     is_plan_runner_diagram=False,):
        """
        Constructs a Diagram that sends commands to ManipulationStationHardwareInterface.
        @param plan_list: A list of Plans to be executed.
        @param gripper_setpoint_list: A list of gripper setpoints. Each setpoint corresponds to a Plan.
        @param sim_duration: the duration of simulation in seconds. If unset, it is set to the sum of the durations of all
            plans in plan_list plus extra_time.
        @param extra_time: the amount of time for which commands are sent, in addition to the duration of all plans.
        @param is_plan_runner_diagram: True: use the diagram version of PlanRunner; False: use the leaf version
            of PlanRunner.
        @return: logs of robot configuration and torque, decoded from LCM messges sent by the robot's driver.
            Logs are SignalLogger systems, whose data can be accessed by SignalLogger.data().
        """
        builder = DiagramBuilder()
        camera_ids = ["805212060544"]
        station_hardware = ManipulationStationHardwareInterface(camera_ids)
        station_hardware.Connect(wait_for_cameras=False)
        builder.AddSystem(station_hardware)

        # Add plan runner.
        if is_plan_runner_diagram:
            plan_runner = CreateManipStationPlanRunnerDiagram(
                station=self.station,
                kuka_plans=plan_list,
                gripper_setpoint_list=gripper_setpoint_list,
                print_period=0,)
        else:
            plan_runner = ManipStationPlanRunner(
                station=self.station,
                kuka_plans=plan_list,
                gripper_setpoint_list=gripper_setpoint_list,
                print_period=0,)

        builder.AddSystem(plan_runner)
        builder.Connect(plan_runner.GetOutputPort("gripper_setpoint"),
                        station_hardware.GetInputPort("wsg_position"))
        builder.Connect(plan_runner.GetOutputPort("force_limit"),
                        station_hardware.GetInputPort("wsg_force_limit"))

        demux = builder.AddSystem(Demultiplexer(14, 7))
        builder.Connect(
            plan_runner.GetOutputPort("iiwa_position_and_torque_command"),
            demux.get_input_port(0))
        builder.Connect(demux.get_output_port(0),
                        station_hardware.GetInputPort("iiwa_position"))
        builder.Connect(demux.get_output_port(1),
                        station_hardware.GetInputPort("iiwa_feedforward_torque"))
        builder.Connect(station_hardware.GetOutputPort("iiwa_position_measured"),
                        plan_runner.GetInputPort("iiwa_position"))
        builder.Connect(station_hardware.GetOutputPort("iiwa_velocity_estimated"),
                        plan_runner.GetInputPort("iiwa_velocity"))

        # Add logger
        iiwa_position_command_log = LogOutput(demux.get_output_port(0), builder)
        iiwa_position_command_log._DeclarePeriodicPublish(0.005)

        iiwa_position_measured_log = LogOutput(
            station_hardware.GetOutputPort("iiwa_position_measured"), builder)
        iiwa_position_measured_log._DeclarePeriodicPublish(0.005)

        iiwa_external_torque_log = LogOutput(
            station_hardware.GetOutputPort("iiwa_torque_external"), builder)
        iiwa_external_torque_log._DeclarePeriodicPublish(0.005)

        # build diagram
        diagram = builder.Build()
        # RenderSystemWithGraphviz(diagram)

        # construct simulator
        simulator = Simulator(diagram)
        self.simulator = simulator

        simulator.set_target_realtime_rate(1.0)
        simulator.set_publish_every_time_step(False)

        t_plan = GetPlanStartingTimes(plan_list)
        if sim_duration is None:
            sim_duration = t_plan[-1] + extra_time

        print "sending trajectories in 2 seconds..."
        time.sleep(1.0)
        print "sending trajectories in 1 second..."
        time.sleep(1.0)
        print "sending trajectories now!"
        simulator.StepTo(sim_duration)

        return iiwa_position_command_log, \
               iiwa_position_measured_log, iiwa_external_torque_log
