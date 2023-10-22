import random
import numpy as np
import tf
from std_msgs.msg import Float32MultiArray

import rospy
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3

import time

class GazeboEnvironment:
    def __init__(self, robot_name):
        self.robot_name = robot_name # hero_0
        self.state = None
        self.reward = 0
        self.done = False
        self.robot_position = None
        self.robot_angle = None
        self.pheromone_value = None
        self.goal_pos_x = 0
        self.goal_pos_y = 0

        # ROSのノードの初期化
        rospy.init_node(f'{self.robot_name}_gazebo_environment', anonymous=True)

        # ロボットをコントロールするためのパブリッシャの設定
        self.cmd_vel_pub = rospy.Publisher(f'/{self.robot_name}/cmd_vel', Twist, queue_size=1)

        # Gazeboのモデルの状態を設定するためのサービスの設定
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        # Gazeboのモデルの状態を取得するためのサブスクライバの設定
        self.gazebo_model_state_sub = rospy.Subscriber(
            '/gazebo/model_states', ModelStates, self.gazebo_model_state_callback)
        
        # pheromoneの値を取得するためのサブスクライバの設定
        self.sub_phero = rospy.Subscriber(
            '/pheromone_value', Float32MultiArray, self.pheromone_callback)

    # Gazeboのモデルの状態を取得するためのコールバック関数
    def gazebo_model_state_callback(self, model_states):
        robot_index = model_states.name.index('hero_0')
        pose = model_states.pose[robot_index]
        pos = pose.position
        self.robot_position = pos
        ori = pose.orientation

        angles = tf.transformations.euler_from_quaternion(
            (ori.x, ori.y, ori.z, ori.w))

        self.robot_angle = angles[2]    

    # pheromoneの値を取得するためのコールバック関数
    def pheromone_callback(self, phero):
        self.pheromone_value = phero.data

    # 環境のリセット
    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)

        # ロボットの初期位置を設定
        state_msg = ModelState()
        state_msg.model_name = 'hero_0'
        state_msg.pose.position.x = 0.0
        state_msg.pose.position.y = 0.0
        state_msg.pose.position.z = 0.1
        state_msg.pose.orientation.x = 0.0
        state_msg.pose.orientation.y = 0.0
        state_msg.pose.orientation.z = 0.0
        state_msg.pose.orientation.w = 0.0


        rospy.wait_for_service('gazebo/reset_simulation')

        self.cmdmsg.linear.x = 0.0
        self.cmdmsg.angular.z = 0.0
        self.cmd_vel_pub.publish(self.cmdmsg)

        # ゴールの初期位置をランダムに設定
        goal_x = random.uniform(-5, 5)
        goal_y = random.uniform(-5, 5)
        goal_state = ModelState(model_name='goal', pose=Pose(position=Point(x=goal_x, y=goal_y, z=0), orientation=Quaternion(x=0, y=0, z=0, w=1)))
        self.set_model_state(goal_state)

        # 状態の初期化
        self.state = [robot_x, robot_y, goal_x, goal_y]
        self.reward = 0
        self.done = False

        return np.array(self.state)

    def step(self, action): # action = [v, w]
        # ロボットに速度を設定
        v, w = action
        twist = Twist()
        twist.linear = Vector3(x=v, y=0, z=0)
        twist.angular = Vector3(x=0, y=0, z=w)
        self.cmd_vel_pub.publish(twist)
        time.sleep(1)

        print('action: ', action)
        print("self.robot_position: ", self.robot_position)
        # # 状態の更新
        # # ここでは簡単のため、ランダムに状態が変化するものとします
        # self.state[0] += v * np.cos(self.state[2]) * 0.1
        # self.state[1] += v * np.sin(self.state[2]) * 0.1
        # self.state[2] += w * 0.1

        # # 報酬の計算
        # distance_to_goal = np.sqrt((self.state[0] - self.state[2])**2 + (self.state[1] - self.state[3])**2)
        # self.reward = -distance_to_goal

        # # タスクの終了判定
        # if distance_to_goal < 0.5:
        #     self.done = True

        # return np.array(self.state), self.reward, self.done
    def calucurate_rewards(self, state):
        distance_to_goal = np.sqrt((state[0] - state[2])**2 + (state[1] - state[3])**2)
        reward = -distance_to_goal
        return reward
