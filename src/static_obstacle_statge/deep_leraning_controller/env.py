#! /usr/bin/env python

import rospy
import rospkg
import tf
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist, Point, Quaternion
import math

from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from std_srvs.srv import Empty
from turtlebot3_waypoint_navigation.srv import PheroReset, PheroResetResponse

import time
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
import random

class InfoGetter:
    def __init__(self):
        """
        Initialize an InfoGetter instance.
        """
        self._event = threading.Event()
        self._msg = None

    def __call__(self, msg):
        """
        Store the received message and trigger the event.
        This method allows the instance itself to act as the callback.
        
        Args:
        - msg: The received message.
        """
        self._msg = msg
        self._event.set()

    def get_msg(self, timeout=None):
        """
        Blocks until the data is received with an optional timeout.
        
        Args:
        - timeout (float, optional): The timeout duration in seconds.
        
        Returns:
        - The received message.
        """
        self._event.wait(timeout)
        return self._msg
    

class Env:

    def __init__(self):
        """
        クラスのインスタンスが作成されるときに実行されるメソッド。
        各種属性の初期化や、ROSノードの設定、移動設定などを行う。
        """
        self._init_settings()
        self._init_ros_nodes()
        self._init_movement_settings()
        self._init_environment_settings()
        self._init_reinforcement_learning_setting()

    def _init_settings(self):
        """環境の基本的な設定を初期化するメソッド"""
        self.num_robots = 1  # Robotの数

    def _init_ros_nodes(self):
        """ROSのノードやトピックの初期化を行うメソッド"""
        self.node = rospy.init_node('environment', anonymous=True)
        self.pose_ig = InfoGetter()
        self.phero_ig = InfoGetter()

        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.pose_info = rospy.Subscriber("/gazebo/model_states", ModelStates, self.pose_ig)
        self.phero_info = rospy.Subscriber("/pheromone_value", Float32MultiArray, self.phero_ig)
        self.rate = rospy.Rate(100) # 一秒間に100回, 10ミリ秒ごと
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)

    def _init_movement_settings(self):
        """ロボットの移動に関する初期設定を行うメソッド"""
        self.position = Point()
        self.move_cmd = Twist()
        self.move_cmd.linear.x = 0.1
        self.move_cmd.angular.z = 0.0
        self.is_collided = False # 衝突判定

    def _init_environment_settings(self): 
        """環境に関する初期設定を行うメソッド"""  
        # Targetの位置の初期化
        self.target_x = 4.0
        self.target_y = 0.0
        self.target_index = 0
        self.target_radius = 4 # Targetの再配置のための円周半径

        self.last_robot_pos_x = 0.0
        self.last_robot_pos_y = 0.0
        self.stuck_indicator = 0
        self.robot_model_index = -1

    def _init_reinforcement_learning_setting(self):
        """強化学習に関する初期設定を行うメソッド"""
        self.state_num = 13 # 9 : フェロモン, 1 : ゴールまでの距離, 1 : ゴールの方向とロボットの現在の方向の角度の差, 2 : ロボットの速度と角速度 
        self.action_num = 2 # 行動 速度と角速度
        self.observation_space = np.empty(self.state_num)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.ep_len_counter = 0
        self.num_experiments = 20 # エピソード回数
        self.dis_rwd_norm = 7 # 報酬の正規化



    #//////////////////////////////////////////////////////
    #                   RESET
    #//////////////////////////////////////////////////////
    def reset(self):
        self.reset_collision_status()
        self.reset_target_position()
        self.reset_robot_position()
        self.reset_target_position()
        # self.reset_pheromone_grid()  # 当該機能が実装されたらコメントアウトを外す
        self.reset_simulation()
        self.reset_initial_state()
        return range(0, self.num_robots), np.zeros(self.state_num)

    def reset_collision_status(self):
        self.is_collided = False

    def reset_target_position(self):
        angle_target = self.target_index * 2.0 * math.pi / self.num_experiments        
        self.target_x = self.target_radius * cos(angle_target)
        self.target_y = self.target_radius * sin(angle_target)
        if self.target_index < self.num_experiments-1:
            self.target_index += 1
        else:
            self.target_index = 0

    def reset_robot_position(self):
        state_msg = self.get_default_state_msg('hero_0')
        self.set_model_state(state_msg)

    def reset_target_position(self):
        state_target_msg = self.get_default_state_msg('unit_sphere_0_0')
        state_target_msg.pose.position.x = self.target_x
        state_target_msg.pose.position.y = self.target_y
        state_target_msg.pose.orientation.z = -0.2
        self.set_model_state(state_target_msg)

    def get_default_state_msg(self, model_name):
        state_msg = ModelState()
        state_msg.model_name = model_name
        state_msg.pose.position.z = 0.0
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 0
        return state_msg

    def set_model_state(self, state_msg):
        rospy.wait_for_service('/gazebo/set_model_state')
        try: 
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
        except rospy.ServiceException as e:
            print("Service Call Failed: %s" % e)

    def reset_simulation(self):
        rospy.wait_for_service('gazebo/reset_simulation')
       

    def reset_initial_state(self):
        self.move_cmd.linear.x = 0.0
        self.move_cmd.angular.z = 0.0
        self.pub.publish(self.move_cmd)
        time.sleep(1)
        self.pub.publish(self.move_cmd)
        self.rate.sleep()


    #//////////////////////////////////////////////////////
    #                   STEP
    #//////////////////////////////////////////////////////
    def rescale_action(self, linear_x, angular_z):
        linear_x = min(1.0, max(-1.0, linear_x)) # only forward motion
        angular_z = min(1.0, max(-1.0, angular_z))
        return linear_x, angular_z

    def get_robot_pose(self):
        model_state = self.pose_ig.get_msg()
        pose = model_state.pose[self.robot_model_index]
        x = pose.position.x
        y = pose.position.y
        ori = pose.orientation
        angles = tf.transformations.euler_from_quaternion((ori.x, ori.y, ori.z, ori.w))
        theta = angles[2]
        return x, y, theta

    def calculate_distance_angle_to_goal(self, x, y, theta):
        distance_to_goal = math.sqrt((x-self.target_x)**2 + (y-self.target_y)**2)
        global_angle = math.atan2(self.target_y - y, self.target_x - x)
        angle_diff = global_angle - theta
        if angle_diff < -math.pi:
            angle_diff += 2*math.pi
        if angle_diff > math.pi:
            angle_diff -= 2*math.pi
        return distance_to_goal, angle_diff

    def calculate_rewards(self, distance_to_goal, distance_to_goal_prv, phero_vals, linear_x_rsc, angular_z_rsc):
        goal_progress = distance_to_goal_prv - distance_to_goal
        distance_reward = goal_progress
        phero_reward = 0.0
        goal_reward = 50.0 if distance_to_goal <= 0.3 else 0.0
        angular_punish_reward = -1 if abs(angular_z_rsc) > 0.8 else 0.0
        linear_punish_reward = -1 if linear_x_rsc < 0.2 else 0.0
        collision_reward = self.check_collision(distance_to_goal)
        total_reward = (distance_reward * (4/self.time_step) + phero_reward + goal_reward + 
                        angular_punish_reward + linear_punish_reward + collision_reward)
        return total_reward

    def check_collision(self, distance_to_goal):
        obs_pos = [[2, 0],[-2,0],[0,2],[0,-2]]
        x, y, _ = self.get_robot_pose()
        dist_obs = [math.sqrt((x-obs_pos[i][0])**2+(y-obs_pos[i][1])**2) for i in range(len(obs_pos))]
        for dist in dist_obs:
            if dist < 0.3:
                return -50
        return 0
    
    def step(self, time_step=0.1, linear_x=0.2, angular_z=0.0):
        # 0. Initiliasation
        start_time = time.time()
        record_time = start_time
        record_time_step = 0

        linear_x_rsc, angular_z_rsc = self.rescale_action(linear_x,angular_z)

        self.move_cmd.linear.x = linear_x_rsc
        self.move_cmd.angular.z = angular_z_rsc

        x_previous, y_previous, _ = self.get_robot_pose()
        distance_to_goal_prv = math.sqrt((x_previous-self.target_x)**2 + (y_previous-self.target_y)**2)
        
        while time.time() - start_time < time_step:
            self.pub.publish(self.move_cmd)
            self.rate.sleep()

        x, y, theta = self.get_robot_pose()
        distance_to_goal, angle_diff = self.calculate_distance_angle_to_goal(x, y, theta)

        state = self.phero_ig.get_msg().data
        state = np.concatenate([state, [distance_to_goal, linear_x, angular_z, angle_diff]])
        
        reward = self.calculate_rewards(distance_to_goal, distance_to_goal_prv, state, linear_x_rsc, angular_z_rsc)
        done = distance_to_goal <= 0.3 or self.check_collision(distance_to_goal) == -50

        if done:
            self.reset()
            time.sleep(1)

        info = [{"episode": {"l": self.ep_len_counter, "r": reward}}]
        self.ep_len_counter += 1
        return range(0, self.num_robots), state, reward, done, info
    
if __name__ == '__main__':
    try:
        env = Env()
    except rospy.ROSInterruptException:
        pass