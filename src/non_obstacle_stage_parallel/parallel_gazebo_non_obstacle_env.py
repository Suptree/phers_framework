import random
import numpy as np
import tf
import math
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
    def __init__(self, id):
        self.robot_name = f"hero_{id}" # hero_0
        self.state = None
        self.reward = 0
        self.done = False

        # 非同期更新
        self.id = id
        self.robot_position = None
        self.robot_linear_velocity = None
        self.robot_angular_velocity = None
        self.robot_angle = None

        # ゴールの位置
        self.goal_pos_x = 0
        self.goal_pos_y = 0
        self.prev_distance_to_goal = 0

        # Robotの状態
        self.robot_color = "CYEAN"

        # Flags
        self.is_goal = False   # ゴールしたかどうか
        self.is_timeout = False # タイムアウトしたかどうか


        # ROSのノードの初期化
        rospy.init_node(f'{self.robot_name}_gazebo_environment', anonymous=True, disable_signals=True)

        # ロボットをコントロールするためのパブリッシャの設定
        self.cmd_vel_pub = rospy.Publisher(f'/{self.robot_name}/cmd_vel', Twist, queue_size=1)

        # # Gazeboのモデルの状態を設定するためのサービスの設定
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        # Gazeboのモデルの状態を取得するためのサブスクライバの設定
        self.gazebo_model_state_sub = rospy.Subscriber(
            '/gazebo/model_states', ModelStates, self.gazebo_model_state_callback)
        
        # マーカーを表示するためのパブリッシャの設定        
        self.marker_pub = rospy.Publisher(f'/{self.robot_name}/visualization_marker', Marker, queue_size=10)
        
        self.pub_led = rospy.Publisher(f'/{self.robot_name}/led', ColorRGBA, queue_size=1)

        self.last_time = rospy.Time.now()

        # Initialise simulation
        self.reset_timer = rospy.get_time()

    # Gazeboのモデルの状態を取得するためのコールバック関数
    def gazebo_model_state_callback(self, model_states):
        robot_index = model_states.name.index(self.robot_name)

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
        while self.last_time == rospy.Time.now():
            # print("waiting for update")
            rospy.sleep(0.01)
        

        # ロボットに速度を設定
        v, w = action
        # vの値を0から0.2の範囲に収める
        v = max(min(v, 0.2), 0.0)
        # wの値を-1.0から1.0の範囲に収める
        w = max(min(w, 1.0), -1.0)
        twist = Twist()
        twist.linear = Vector3(x=v, y=0, z=0)
        twist.angular = Vector3(x=0, y=0, z=w)
        
        current_time = rospy.Time.now()
        self.cmd_vel_pub.publish(twist)
        # while rospy.Time.now() < current_time + rospy.Duration(0.1):
        #     pass
        rospy.sleep(0.1)

        # アクション後の環境の状態を取得, 衝突判定やゴール判定も行う
        next_state_distance_to_goal, next_state_angle_to_goal, next_state_robot_linear_velocity_x, next_state_robot_angular_velocity_z = self.get_next_state()
        
        # 報酬の計算
        reward, baseline_reward = self.calculate_rewards(next_state_distance_to_goal,next_state_robot_angular_velocity_z)
        # タスクの終了判定
        self.done = self.is_collided or self.is_goal or self.is_timeout
        # print("self.done: ", self.done)
        if self.is_goal:
            print("/////// GOAL ///////")
        # 状態の更新
        self.prev_distance_to_goal = next_state_distance_to_goal


        # 観測情報をstateに格納
        self.state = [next_state_distance_to_goal, next_state_angle_to_goal, next_state_robot_linear_velocity_x, next_state_robot_angular_velocity_z]
        self.last_time = rospy.Time.now()
        
        # 訓練したNNモデルを評価するための情報を返す
        info = None
        # infoの設定
        if self.done:
            task_time = rospy.get_time() - self.reset_timer
            if self.is_goal:
                done_category = 0
            elif self.is_collided:
                done_category = 1
            else: # self.is_timeout
                done_category = 2
            info = {"task_time": task_time, "done_category": done_category}
 

        return self.state, reward, self.done, baseline_reward, info


    def calculate_rewards(self, next_state_distance_to_goal,next_state_robot_angular_velocity_z):
        Rw = -1.0  # angular velocity penalty constant
        Ra = 30.0  # goal reward constant
        Rc = -30.0 # collision penalty constant
        # Rt = -1.0  # time penalty
        w_m = 0.8  # maximum allowable angular velocity
        wd_p = 4.0 # weight for positive distance
        wd_n = 6.0 # weight for negative distance

        # アクション後のロボットとゴールまでの距離の差分
        goal_to_distance_diff = 100.0 * (self.prev_distance_to_goal - next_state_distance_to_goal)
        
        r_g = Ra if self.is_goal else 0 # goal reward
        r_c = Rc if self.is_collided else 0  # collision penalty
        if goal_to_distance_diff > 0:
            r_d = wd_p * goal_to_distance_diff
        else:
            r_d = wd_n * goal_to_distance_diff
        r_w = Rw if abs(next_state_robot_angular_velocity_z) > w_m else 0  # angular velocity penalty

        reward = r_g + r_c + r_d + r_w

        return reward, reward

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

        # ゴール判定
        self.is_goal = self.check_goal()

        # タイムアウト判定
        self.is_timeout = rospy.get_time() - self.reset_timer > 40.0

        return next_state_distance_to_goal, next_state_angle_to_goal, self.robot_linear_velocity.x, self.robot_angular_velocity.z
    
    # ゴールに到達したかどうか
    def check_goal(self):
        distance_to_goal =  math.sqrt((self.robot_position.x - self.goal_pos_x)**2
                             + (self.robot_position.y - self.goal_pos_y)**2)
        
        if distance_to_goal <= 0.0535:
            return True

        return False
    

    # 環境のリセット
    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)

        # # ゴールの初期位置をランダムに設定
        goal_r = 0.8
        goal_radius = 2.0 * math.pi * random.random()

        self.goal_pos_x = float(int(int(self.id) / 4) * 20.0) + goal_r * math.cos(goal_radius)
        self.goal_pos_y = float((int(self.id) % 4) * 20.0) + goal_r * math.sin(goal_radius)

        # ロボット停止命令
        twist = Twist()
        twist.linear = Vector3(x=0, y=0, z=0)
        twist.angular = Vector3(x=0, y=0, z=0)
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(1.0)

        # 新しいゴールマーカーを設定
        self.set_goal_marker(self.goal_pos_x, self.goal_pos_y)

        # ロボットの位置をリセット
        before = self.robot_position
        self.set_robot()
        # # ロボットの位置がリセットされるまで待機
        # while before == self.robot_position:
            # print("waiting for reset robot position")
        rospy.sleep(3.0)
        
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

        # previous distance to goal の初期値をidごとの原点からゴールまでの距離に設定
        self.prev_distance_to_goal = self.distance_to_goal

        self.done = False
        self.state = [self.distance_to_goal, angle_to_goal,self.robot_linear_velocity.x, self.robot_angular_velocity.z]

        return self.state
    def set_robot(self):

        state_msg = ModelState()
        state_msg.model_name = self.robot_name
        
        state_msg.pose.position.x = int(int(self.id) / 4) * 20.0
        state_msg.pose.position.y = (int(self.id) % 4) * 20.0

        state_msg.pose.position.z = 0.2395
        state_msg.pose.orientation.x = 0.0
        state_msg.pose.orientation.y = 0.0
        state_msg.pose.orientation.z = 0.0
        state_msg.pose.orientation.w = 0.0
        state_msg.twist.linear.x = 0.0
        state_msg.twist.linear.y = 0.0
        state_msg.twist.linear.z = 0.0
        state_msg.twist.angular.x = 0.0
        state_msg.twist.angular.y = 0.0
        state_msg.twist.angular.z = 0.0
        

        try:
            self.set_model_state(state_msg)
        except rospy.ServiceException as e:
            print("Spawn URDF service call failed: {0}".format(e))


    def set_goal_marker(self, x, y):
        """
        Set a goal marker in the Gazebo world.
        """
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "goal"
        marker.id = self.id
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
    def shutdown(self):
        """
        Shuts down the ROS node.
        """
        twist = Twist()
        twist.linear = Vector3(x=0, y=0, z=0)
        twist.angular = Vector3(x=0, y=0, z=0)
        self.cmd_vel_pub.publish(twist)


        rospy.signal_shutdown("Closing Gazebo environment")
        rospy.spin()