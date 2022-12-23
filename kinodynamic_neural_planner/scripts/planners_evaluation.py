#!/usr/bin/env python
import os.path
import shlex
import signal
import subprocess

import psutil
import rospy
import tf

import numpy as np
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, DeleteModel, GetModelState, SetModelConfiguration

from kinodynamic_msgs.msg import PlannerRequest, PlannerStatus

from air_hockey_puck_tracker.srv import GetPuckState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from manifold_planning.utils.data import unpack_data_kinodynamic
from manifold_planning.utils.constants import Cup
from gazebo_ros import gazebo_interface
from geometry_msgs.msg import Pose, Quaternion
from tf.transformations import quaternion_from_euler



def print_(x, N=5):
    for i in range(N):
        print()
    print(x)
    for i in range(N):
        print()



class PlannersEvaluationNode:
    def __init__(self):
        rospy.init_node("planners_evaluation", anonymous=True)
        self.tf_listener = tf.TransformListener()
        self.planner_request_publisher = rospy.Publisher("/neural_planner/plan_trajectory", PlannerRequest,
                                                         queue_size=5)
        self.robot_state_subscriber = rospy.Subscriber("/joint_states", JointState, self.set_robot_state)
        self.planner_status_subscriber = rospy.Subscriber(f"/neural_planner/status", PlannerStatus, self.kill_rosbag_proc)
        self.iiwa_publisher = rospy.Publisher(f"/bspline_ff_kino_joint_trajectory_controller/command",
        # self.iiwa_publisher = rospy.Publisher(f"/bspline_joint_trajectory_controller/command",
                                                    JointTrajectory,
                                                    queue_size=5)
        self.robot_joint_pose = None
        self.robot_joint_velocity = None
        self.rosbag_proc = None
        self.is_moving = False
        #self.pause_physics_srv = rospy.ServiceProxy('/gazebo/pause_physics', )
        #self.unpause_physics_srv = rospy.ServiceProxy('/gazebo/unpause_physics', None)
        self.delete_model_srv = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.get_model_srv = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_model_configration_srv = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
        self.delete_model("ground_plane")
        package_path = os.path.join(os.path.dirname(__file__), "..")
        file_name = os.path.join(package_path, "data/kinodynamic7fixed_12kg_validated/test/data.tsv")
        self.data = np.loadtxt(file_name, delimiter="\t").astype(np.float32)#[:5]
        #self.method = "sst"
        #self.method = "mpcmpnet"
        #self.method = "ours"
        #self.method = "ours_long"
        #self.method = "ours_n10"
        #self.method = "ours_n20"
        #self.method = "ours_l64_long"
        #self.method = "ours_l128_long"
        #self.method = "ours_l256_long"
        #self.method = "ours_l512_long"
        #self.method = "ours_l1024_long"
        self.method = "ours_l2048_long"
        #self.method = "ours_l3072_long"
        #self.method = "ours_l2048s"
        #self.method = "ours_l2048sl"
        #self.method = "ours_l128"
        #self.method = "ours_l256"
        #self.method = "ours_l512"
        #self.method = "ours_l1024"
        #self.method = "iros"
        #self.method = "nlopt"
        #self.method = "cbirrt"

    def move_to(self, q):
        iiwa_front_msg = JointTrajectory()
        pt = JointTrajectoryPoint()
        pt.positions = q
        pt.velocities = [0.] * 7
        pt.accelerations = [0.] * 7
        pt.time_from_start = rospy.Duration(3.0)
        iiwa_front_msg.points.append(pt)
        iiwa_front_msg.header.stamp = rospy.Time.now() + rospy.Duration(0.1)
        iiwa_front_msg.joint_names = [f"F_joint_{i}" for i in range(1, 8)]
        self.iiwa_publisher.publish(iiwa_front_msg)
        rospy.sleep(5.)

    def kill_rosbag_proc(self, msg):
        print("XDDDDD")
        rospy.sleep(max(4., msg.planned_motion_time + 2.))
        #rospy.sleep(max(4., msg.planned_hitting_time + 1.))
        if self.rosbag_proc is not None:
            os.kill(self.rosbag_proc.pid, signal.SIGTERM)
        rospy.sleep(1.)
        self.rosbag_proc = None
        self.is_moving = False

    def set_robot_state(self, msg):
        self.robot_joint_pose = msg.position[:7]
        self.robot_joint_velocity = msg.velocity[:7]

    def delete_model(self, name):
        if self.get_model_srv(model_name=name).success:
            self.delete_model_srv(model_name=name)

    def record_rosbag(self, i):
        name = f"data/kino_exp/{self.method}/{i:03d}.bag"
        command = "rosbag record " \
                  f"-O {name} " \
                  "/joint_states /tf " \
                  "/bspline_ff_kino_joint_trajectory_controller/state " \
                  "/neural_planner/status /neural_planner/plan_trajectory"
        # "/bspline_joint_trajectory_controller/state " \
        command = shlex.split(command)
        self.rosbag_proc = subprocess.Popen(command)
        rospy.sleep(1.0)

    def evaluate(self):
        q0, qk, xyz0, xyzk, q_dot_0, q_dot_k, q_ddot_0 = unpack_data_kinodynamic(self.data, 7)
        #n = 17
        for i in range(0, len(q0)):
        #for i in range(n, n+1):
        #for i in [16, 22, 23, 24, 42, 44, 65, 72, 93]:
        ##for i in [1, 12, 16, 24, 39, 42, 44, 65, 74, 91]:
        #for i in [0, 6, 16, 21, 23, 24, 42, 44, 65, 66, 67, 72, 92]:
        #for i in [67, 72]:
        #for i in [24, 44, 53, 80, 93, 94]:
        #for i in range(37, 44):
            print("Q0:", q0[i])
            print("XYZ0:", xyz0[i])
            print("XYZK:", xyzk[i])
            self.delete_boxes()
            self.move_to(q0[i])
            self.create_boxes(xyz0[i, -1] - Cup.height, xyzk[i, -1] - Cup.height)
            #self.record_rosbag(i)
            self.request_plan(qk[i])
            self.is_moving = True
            k = 0
            while self.is_moving:
                print("XD", self.is_moving, k)
                rospy.sleep(0.1)
                k += 1
                pass
            rospy.sleep(1.)

    def delete_boxes(self):
        self.delete_model("box_1")
        self.delete_model("box_2")

    def create_boxes(self, h1, h2):
        def model_xml(x, y, h):
            return f"""
            <robot name="box">
              <link name="world"/>
              <joint name="base_joint" type="fixed">
                <parent link="world"/>
                <child link="box"/>
                <origin xyz="{x} {y} {h/2.}" rpy="0 0 0"/>
              </joint>
              <link name="box">
                <inertial>
                  <origin xyz="0 0 0" />
                  <mass value="100.0" />
                  <inertia  ixx="100.0" ixy="0.0"  ixz="0.0"  iyy="100.0"  iyz="0.0"  izz="100.0" />
                </inertial>
                <visual>
                  <origin xyz="0 0 0"/>
                  <geometry>
                    <box size="0.4 0.3 {h}" />
                  </geometry>
                </visual>
              </link>
              <gazebo reference="box">
                <material>Gazebo/Blue</material>
              </gazebo>
            </robot>
            """
        gazebo_interface.spawn_urdf_model_client("box_1", model_xml(0.4, -0.45, h1), "", Pose(), 'world', "/gazebo")
        gazebo_interface.spawn_urdf_model_client("box_2", model_xml(0.4, 0.45, h2), "", Pose(), 'world', "/gazebo")

    def prepare_planner_request(self, q_d):
        pr = PlannerRequest()
        pr.q_0 = self.robot_joint_pose
        pr.q_dot_0 = np.zeros(7)
        pr.q_d = q_d
        pr.q_dot_0 = np.zeros(7)
        pr.header.stamp = rospy.Time.now()
        return pr

    def request_plan(self, q_d):
        pr = self.prepare_planner_request(q_d)
        if pr is None:
            return False
        self.planner_request_publisher.publish(pr)
        return True


if __name__ == '__main__':
    node = PlannersEvaluationNode()
    node.evaluate()