import random
import numpy as np
import tf
import math
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty
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
        self.robot_linear_velocity = None
        self.robot_angular_velocity = None
        self.robot_angle = None
        self.pheromone_value = None
        self.goal_pos_x = 0
        self.goal_pos_y = 0
        self.distance_to_goal = 0

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
        ori = pose.orientation
        angles = tf.transformations.euler_from_quaternion(
            (ori.x, ori.y, ori.z, ori.w))
        
        robot_twist = model_states.twist[robot_index]

        self.robot_position = pose.position
        self.robot_angle = angles[2]    
        self.robot_linear_velocity = robot_twist.linear
        self.robot_angular_velocity = robot_twist.angular
    # pheromoneの値を取得するためのコールバック関数
    def pheromone_callback(self, phero):
        self.pheromone_value = phero.data

    # 環境のステップを実行する
    def step(self, action): # action = [v, w]
        # ロボットに速度を設定
        v, w = action
        twist = Twist()
        twist.linear = Vector3(x=v, y=0, z=0)
        twist.angular = Vector3(x=0, y=0, z=w)
        self.cmd_vel_pub.publish(twist)
        time.sleep(0.2)

        print('Gazebo Env : action: ', action)
        print("Gazebo Env : self.robot_position( after step ): \n", self.robot_position)

        # 状態の更新
        next_state = self.get_next_state()

        reward = self.calculate_rewards(next_state)

        return next_state, reward, self.done

        # # 報酬の計算
        # distance_to_goal = np.sqrt((self.state[0] - self.state[2])**2 + (self.state[1] - self.state[3])**2)
        # self.reward = -distance_to_goal

        # # タスクの終了判定
        # if distance_to_goal < 0.5:
        #     self.done = True

        # return np.array(self.state), self.reward, self.done



        
    
    def calculate_rewards(self, next_state):
        Rw = -1.0  # angular velocity penalty constant
        Ra = 30.0  # goal reward constant
        Rc = -30.0 # collision penalty constant
        w_m = 0.2  # maximum allowable angular velocity
        wd_p = 4.0 # weight for positive distance
        wd_n = 6.0 # weight for negative distance

        next_state_robot_pos_x, next_state_robot_pos_y, next_state_robot_angle, next_state_distance_to_goal, next_state_angle_to_goal, next_state_linear_x, next_state_angular_z = next_state

        goal_to_distance_diff = self.distance_to_goal - next_state_distance_to_goal

        r_g = Ra if next_state_distance_to_goal < 0.5 else 0  # goal reward
        r_c = 0  # collision penalty（衝突検出ロジックを追加する場合は、この部分を更新してください。
        r_d = wd_p * goal_to_distance_diff if goal_to_distance_diff > 0 else wd_n * goal_to_distance_diff
        r_w = Rw if abs(next_state_angular_z) > w_m else 0  # angular velocity penalty

        return r_g + r_c + r_d + r_w

    def get_next_state(self):
        # 現在の変数の格納
        next_state_robot_pos_x = self.robot_position.x
        next_state_robot_pos_y = self.robot_position.y
        next_state_robot_angle = self.robot_angle
        next_state_linear_x = self.robot_linear_velocity.x
        next_state_angular_z = self.robot_angular_velocity.z

        # ゴールまでの距離
        next_state_distance_to_goal = math.sqrt((next_state_robot_pos_x-self.goal_pos_x)**2
                             + (next_state_robot_pos_y-self.goal_pos_y)**2)
        
        # ロボットの現在の体の向きのベクトルとロボットの現在の位置からゴールまでのベクトルのなす角度
        next_state_angle_to_goal = math.atan2(self.goal_pos_y - next_state_robot_pos_y,
                                   self.goal_pos_x - next_state_robot_pos_x) - next_state_robot_angle


        return [next_state_robot_pos_x,
                next_state_robot_pos_y,
                next_state_robot_angle,
                next_state_distance_to_goal,
                next_state_angle_to_goal,
                next_state_linear_x,
                next_state_angular_z]
    


    # 環境のリセット
    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)

        # # ゴールの初期位置をランダムに設定
        goal_r = 0.8
        goal_radius = 2.0 * math.pi * random.random()
        self.goal_pos_x = goal_r * math.cos(goal_radius)
        self.goal_pos_y = goal_r * math.sin(goal_radius)

        # シミュレーションをリセット
        rospy.wait_for_service('/gazebo/reset_simulation')
        reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)  # Emptyは適切なメッセージタイプに置き換えてください
        reset_simulation()
        before = self.robot_position

        # シミュレーションが初期化されるまで待機
        while before == self.robot_position:
            time.sleep(1)

        # 変数の初期化
        self.reward = 0
        self.distance_to_goal = math.sqrt((self.robot_position.x-self.goal_pos_x)**2
                             + (self.robot_position.y-self.goal_pos_y)**2)
        # ロボットの現在の体の向きのベクトルとロボットの現在の位置からゴールまでのベクトルのなす角度
        angle_to_goal = math.atan2(self.goal_pos_y - self.robot_position.y,
                                   self.goal_pos_x - self.robot_position.x) - self.robot_angle

        self.done = False
        self.state = [self.distance_to_goal, angle_to_goal,self.robot_linear_velocity.x, self.robot_angular_velocity.z]
        return self.state