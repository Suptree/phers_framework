import random
import numpy as np
import tf
import math
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty
from std_msgs.msg import Empty as EmptyMsg
from std_msgs.msg import ColorRGBA

import rospy
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from visualization_msgs.msg import Marker

import time

class GazeboEnvironment:
    def __init__(self, robot_name):
        self.robot_name = robot_name # hero_0
        self.state = None
        self.reward = 0
        self.done = False

        # 非同期更新
        self.robot_position = None
        self.robot_linear_velocity = None
        self.robot_angular_velocity = None
        self.robot_angle = None
        self.pheromone_value = None

        # 障害物の位置
        self.obstacle = [[0.4, 0.0], [-0.4, 0.0], [0.0, 0.4], [0.0, -0.4]]

        # ゴールの位置
        self.goal_pos_x = 0
        self.goal_pos_y = 0
        self.prev_distance_to_goal = 0

        # Robotの状態
        self.robot_color = "CYEAN"


        # Flags
        self.is_collided = False # 衝突したかどうか
        self.is_goal = False   # ゴールしたかどうか
        self.is_timeout = False # タイムアウトしたかどうか
        


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
        # pheromoneの値をリセットするためのパブリッシャの設定
        self.reset_pheromone_pub = rospy.Publisher('/pheromone_reset_signal', EmptyMsg, queue_size=1)

        # マーカーを表示するためのパブリッシャの設定        
        self.marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
        
        self.pub_led = rospy.Publisher(f'/{self.robot_name}/led', ColorRGBA, queue_size=1)


        # Initialise simulation
        self.reset_timer = rospy.get_time()

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
        # vの値を0から0.2の範囲に収める
        v = max(min(v, 0.2), 0,0)
        # wの値を-1.0から1.0の範囲に収める
        w = max(min(w, 1.0), -1.0)
        twist = Twist()
        twist.linear = Vector3(x=v, y=0, z=0)
        twist.angular = Vector3(x=0, y=0, z=w)
        self.cmd_vel_pub.publish(twist)
        time.sleep(0.1)

        # アクション後の環境の状態を取得, 衝突判定やゴール判定も行う
        next_state_pheromone_value, next_state_distance_to_goal, next_state_angle_to_goal, next_state_robot_linear_velocity_x, next_state_robot_angular_velocity_z = self.get_next_state()
        
        # ゴールとの角度を度数法にしてプリント

        # フェロモンを読み込んだ場合は緑色にする
        if sum(next_state_pheromone_value) > 0:
            if self.robot_color != "GREEN":
                self.robot_color = "GREEN"
                color = ColorRGBA()
                color.r = 0
                color.g = 255
                color.b = 0
                color.a = 255
                self.pub_led.publish(color)
        else: # pheromone == 0
            if self.robot_color != "CYEAN":
                self.robot_color = "CYEAN"
                color = ColorRGBA()
                color.r = 0
                color.g = 160
                color.b = 233
                color.a = 255
                self.pub_led.publish(color)

        
        # 報酬の計算
        reward, baseline_reward = self.calculate_rewards(next_state_distance_to_goal,next_state_robot_angular_velocity_z)
        # print("reward: ", reward)
        # タスクの終了判定
        self.done = self.is_collided or self.is_goal or self.is_timeout
        # print("self.done: ", self.done)
        if self.is_goal:
            print("/////// GOAL ///////")
        # 状態の更新
        self.prev_distance_to_goal = next_state_distance_to_goal

        # 観測情報をstateに格納
        self.state = list(next_state_pheromone_value) + [next_state_distance_to_goal, next_state_angle_to_goal, next_state_robot_linear_velocity_x, next_state_robot_angular_velocity_z]

        return self.state, reward, self.done, baseline_reward


    def calculate_rewards(self, next_state_distance_to_goal,next_state_robot_angular_velocity_z):
        Rw = -1.0  # angular velocity penalty constant
        Ra = 30.0  # goal reward constant
        Rc = -30.0 # collision penalty constant
        Rt = -1.0  # time penalty
        w_m = 0.8  # maximum allowable angular velocity
        wd_p = 4.0 # weight for positive distance
        wd_n = 6.0 # weight for negative distance

        # アクション後のロボットとゴールまでの距離の差分
        goal_to_distance_diff = self.prev_distance_to_goal - next_state_distance_to_goal

        r_g = Ra if self.is_goal else 0 # goal reward
        r_c = Rc if self.is_collided else 0  # collision penalty
        r_d = wd_p * goal_to_distance_diff if goal_to_distance_diff > 0 else wd_n * goal_to_distance_diff
        r_w = Rw if abs(next_state_robot_angular_velocity_z) > w_m else 0  # angular velocity penalty
        r_t = Rt

        reward = r_g + r_c + r_d + r_w + r_t

        base_r_d = 4.0 * goal_to_distance_diff if goal_to_distance_diff > 0 else wd_n * goal_to_distance_diff
        baseline_reward = r_g + r_c + base_r_d + r_w

        return reward, baseline_reward

    def get_next_state(self):

        # ゴールまでの距離
        next_state_distance_to_goal = math.sqrt((self.robot_position.x - self.goal_pos_x)**2
                             + (self.robot_position.y - self.goal_pos_y)**2)
        
        # ロボットの現在の体の向きのベクトルとロボットの現在の位置からゴールまでのベクトルのなす角度
        next_state_angle_to_goal = math.atan2(self.goal_pos_y - self.robot_position.y,
                                   self.goal_pos_x - self.robot_position.x) - self.robot_angle
        ## 角度を-πからπの範囲に正規化
        if next_state_angle_to_goal < -math.pi:
            next_state_angle_to_goal += 2 * math.pi
        elif next_state_angle_to_goal > math.pi:
            next_state_angle_to_goal -= 2 * math.pi

        # 障害物との衝突判定
        self.is_collided = self.check_collision_to_obstacle()
        
        # ゴール判定
        self.is_goal = self.check_goal()
        # if next_state_distance_to_goal <= 0.02:
        #     self.is_goal = True

        # タイムアウト判定
        self.is_timeout = rospy.get_time() - self.reset_timer > 40.0

        return self.pheromone_value, next_state_distance_to_goal, next_state_angle_to_goal, self.robot_linear_velocity.x, self.robot_angular_velocity.z
    
    # ゴールに到達したかどうか
    def check_goal(self):
        distance_to_goal =  math.sqrt((self.robot_position.x - self.goal_pos_x)**2
                             + (self.robot_position.y - self.goal_pos_y)**2)
        dx = self.goal_pos_x - self.robot_position.x
        dy = self.goal_pos_y - self.robot_position.y
        angle_to_goal = math.atan2(dy, dx)  # ロボットから見たゴールの角度
        angle_robot = self.robot_angle  # ロボットの向き
        relative_angle = abs(angle_to_goal - angle_robot)
        if relative_angle <= math.radians(5):
            if distance_to_goal <= 0.0714:
                # print("ゴール\n")
                return True
        else:
            if distance_to_goal <= 0.059:
                # print("ゴール\n")
                return True
        return False
    def check_collision_to_obstacle(self):
        for obs in self.obstacle:
            distance_to_obstacle = math.sqrt((self.robot_position.x - obs[0])**2 + (self.robot_position.y - obs[1])**2)
            dx = obs[0] - self.robot_position.x
            dy = obs[1] - self.robot_position.y
            angle_to_obstacle = math.atan2(dy, dx)  # ロボットから見た障害物の角度
            angle_robot = self.robot_angle  # ロボットの向き
            relative_angle = abs(angle_to_obstacle - angle_robot)
            
            if relative_angle <= math.radians(5):  # 障害物が正面にある場合
                if distance_to_obstacle <= 0.0714:
                    # 障害物の座標と正面という情報と角度をプリント
                    # print("正面\n")
                    # print("obs: ", obs)
                    # print("relative_angle: ", math.degrees(relative_angle))
                    # print("angle_robot: ", math.degrees(angle_robot))
                    # print("angle_to_obstacle: ", math.degrees(angle_to_obstacle))
                    # print("distance_to_obstacle: ", distance_to_obstacle)
                    return True
            else:  # 障害物が横や後ろにある場合
                if distance_to_obstacle <= 0.059:
                    # 障害物の座標と正面という情報と角度をプリント
                    # print("正面以外\n")
                    # print("obs: ", obs)
                    # print("relative_angle: ", math.degrees(relative_angle))
                    # print("angle_robot: ", math.degrees(angle_robot))
                    # print("angle_to_obstacle: ", math.degrees(angle_to_obstacle))
                    # print("distance_to_obstacle: ", distance_to_obstacle)
                    return True
        return False

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
        
        # フラグのリセット
        self.is_collided = False
        self.is_goal = False
        self.is_timeout = False

        # 変数の初期化
        self.reward = 0
        self.distance_to_goal = math.sqrt((self.robot_position.x-self.goal_pos_x)**2
                             + (self.robot_position.y-self.goal_pos_y)**2)
        self.reset_timer = rospy.get_time()

        # ロボットの現在の体の向きのベクトルとロボットの現在の位置からゴールまでのベクトルのなす角度
        angle_to_goal = math.atan2(self.goal_pos_y - self.robot_position.y,
                                   self.goal_pos_x - self.robot_position.x) - self.robot_angle
        ## 角度を-πからπの範囲に正規化
        if angle_to_goal < -math.pi:
            angle_to_goal += 2 * math.pi
        elif angle_to_goal > math.pi:
            angle_to_goal -= 2 * math.pi
        # フェロモンマップをリセット
        self.reset_pheromone_pub.publish(EmptyMsg())
        # ゴールのマーカーを表示
        self.set_goal_marker(self.goal_pos_x, self.goal_pos_y)

        self.done = False
        self.state = list(self.pheromone_value) + [self.distance_to_goal, angle_to_goal,self.robot_linear_velocity.x, self.robot_angular_velocity.z]

        return self.state
    

    def set_goal_marker(self, x, y):
        """
        Set a goal marker in the Gazebo world.
        """
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.04
        marker.scale.y = 0.04
        marker.scale.z = 0.1

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.marker_pub.publish(marker)
