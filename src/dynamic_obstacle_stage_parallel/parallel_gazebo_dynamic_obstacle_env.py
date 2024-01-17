import random
import numpy as np
import tf
import math
from std_msgs.msg import ColorRGBA
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Empty as EmptyMsg
from geometry_msgs.msg import Pose

import rospy
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import DeleteModel
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from visualization_msgs.msg import Marker
import time
import os
import slackweb

class GazeboEnvironment:
    def __init__(self, id):
        self.id = id

        # 原点座標
        self.origin_x = float((int(self.id) % 4) * 20.0) 
        self.origin_y = float(int(int(self.id) / 4) * 20.0)

        self.id = id * 2

        self.robot_num = 2

        self.robot_name = [f"hero_{self.id}", f"hero_{self.id + 1}"]

        self.state = [None] * self.robot_num
        self.reward = [None] * self.robot_num
        self.done = [None] * self.robot_num

        # 非同期更新
        self.robot_position = [None] * self.robot_num
        self.robot_linear_velocity = [None] * self.robot_num
        self.robot_angular_velocity = [None] * self.robot_num
        self.robot_angle = [None] * self.robot_num
        self.pheromone_value = [None] * self.robot_num

        # ゴールの位置
        self.goal_pos_x = [None] * self.robot_num
        self.goal_pos_y = [None] * self.robot_num
        self.prev_distance_to_goal = [None] * self.robot_num


        # Robotの状態
        self.robot_color = ["CYEAN"] * self.robot_num
        print(self.robot_color)

        # Flags
        self.is_collided = [False] * self.robot_num
        print(self.is_collided)
        self.is_goal = [False] * self.robot_num   # ゴールしたかどうか
        self.is_timeout = [False] * self.robot_num # タイムアウトしたかどうか


        # ROSのノードの初期化
        rospy.init_node(f'{self.id}_gazebo_environment', anonymous=True, disable_signals=True)

        # ロボットをコントロールするためのパブリッシャの設定
        self.cmd_vel_pub = [rospy.Publisher(f'/{self.robot_name[0]}/cmd_vel', Twist, queue_size=1),
                            rospy.Publisher(f'/{self.robot_name[1]}/cmd_vel', Twist, queue_size=1)]

        # # Gazeboのモデル(Robot)の状態を設定するためのサービスの設定
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        # Gazeboのモデルの状態を取得するためのサブスクライバの設定
        self.gazebo_model_state_sub = rospy.Subscriber(
            '/gazebo/model_states', ModelStates, self.gazebo_model_state_callback)

        # pheromoneの値を取得するためのサブスクライバの設定
        self.sub_phero = rospy.Subscriber(
            f'/{self.id}/pheromone_value', Float32MultiArray, self.pheromone_callback)
        
        # pheromoneの値をリセットするためのパブリッシャの設定
        self.reset_pheromone_pub = rospy.Publisher(f'/{self.id}/pheromone_reset_signal', EmptyMsg, queue_size=1)
        
        # マーカーを表示するためのパブリッシャの設定        
        self.marker_pub = rospy.Publisher(f'/{self.id}/visualization_marker', Marker, queue_size=10)
        
        self.pub_led = [rospy.Publisher(f'/{self.robot_name[0]}/led', ColorRGBA, queue_size=1),
                        rospy.Publisher(f'/{self.robot_name[1]}/led', ColorRGBA, queue_size=1)]

        self.last_time = rospy.Time.now()

        # Initialise simulation
        self.reset_timer = rospy.get_time()

    # Gazeboのモデルの状態を取得するためのコールバック関数
    def gazebo_model_state_callback(self, model_states):
        for i in range(self.robot_num):
            # 各ロボットのインデックスを取得
            robot_index = model_states.name.index(self.robot_name[i])

            # 各ロボットのposeとtwist情報を取得
            pose = model_states.pose[robot_index]
            twist = model_states.twist[robot_index]
            ori = pose.orientation
            angles = tf.transformations.euler_from_quaternion(
                (ori.x, ori.y, ori.z, ori.w))

            # 各ロボットの情報をクラス変数に格納
            self.robot_position[i] = pose.position
            self.robot_angle[i] = angles[2]
            self.robot_linear_velocity[i] = twist.linear
            self.robot_angular_velocity[i] = twist.angular

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
        
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(0.1)

        # アクション後の環境の状態を取得, 衝突判定やゴール判定も行う
        next_state_pheromone_value, next_state_distance_to_goal, next_state_angle_to_goal, next_state_robot_linear_velocity_x, next_state_robot_angular_velocity_z = self.get_next_state()
        
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
        # タスクの終了判定
        self.done = self.is_collided or self.is_goal or self.is_timeout
        # print("self.done: ", self.done)
        if self.is_goal:
            print(f"\033[1;36m[HERO_{self.id}]\033[0m : \033[32m///////   GOAL    ///////\033[0m")
        elif self.is_collided:
            print(f"\033[1;36m[HERO_{self.id}]\033[0m : \033[38;5;214m/////// COLLISION ///////\033[0m")
        # 状態の更新
        self.prev_distance_to_goal = next_state_distance_to_goal

        # 観測情報をstateに格納
        self.state = list(next_state_pheromone_value) + [next_state_distance_to_goal, next_state_angle_to_goal, next_state_robot_linear_velocity_x, next_state_robot_angular_velocity_z]
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
            info = {"task_time": task_time, "done_category": done_category, "angle_to_goal": math.degrees(next_state_angle_to_goal),
                    "pheromone_mean": np.mean(next_state_pheromone_value),
                    "pheromone_value": next_state_pheromone_value,
                    "pheromone_left_value" : (next_state_pheromone_value[0] + next_state_pheromone_value[3] + next_state_pheromone_value[6])/3.0,
                    "pheromone_right_value" : (next_state_pheromone_value[2] + next_state_pheromone_value[5] + next_state_pheromone_value[8])/3.0,
                    }
        else:
            info = {"task_time": None, "done_category": None, "angle_to_goal": math.degrees(next_state_angle_to_goal),
                    "pheromone_mean": np.mean(next_state_pheromone_value),
                    "pheromone_value": next_state_pheromone_value,
                    "pheromone_left_value" : (next_state_pheromone_value[0] + next_state_pheromone_value[3] + next_state_pheromone_value[6])/3.0,
                    "pheromone_right_value" : (next_state_pheromone_value[2] + next_state_pheromone_value[5] + next_state_pheromone_value[8])/3.0,
                    }
        return self.state, reward, self.done, baseline_reward, info


    def calculate_rewards(self, next_state_distance_to_goal,next_state_robot_angular_velocity_z):
        Rw = -1.0  # angular velocity penalty constant
        Ra = 300.0  # goal reward constant
        Rc = -300.0 # collision penalty constant
        Rt = -0.1  # time penalty
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
        r_t = Rt
        reward = r_g + r_c + r_d + r_w + r_t
        baseline_reward = r_g + r_c + r_d + r_w

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

        # 衝突判定
        self.is_collided = self.check_collision_to_obstacle()

        # ゴール判定
        self.is_goal = self.check_goal()

        # タイムアウト判定
        self.is_timeout = rospy.get_time() - self.reset_timer > 40.0

        return self.pheromone_value, next_state_distance_to_goal, next_state_angle_to_goal, self.robot_linear_velocity.x, self.robot_angular_velocity.z
    
    def normalize_distance_to_goal(self, distance_to_goal):
        max_distance = 20.0
        normalize_distance_to_goal = distance_to_goal / max_distance

        return normalize_distance_to_goal
    
    def normalize_angle_to_goal(self, angle_to_goal):
        normalize_angle_to_goal = angle_to_goal / math.pi

        return normalize_angle_to_goal
    
    # ゴールに到達したかどうか
    def check_goal(self):
        distance_to_goal =  math.sqrt((self.robot_position.x - self.goal_pos_x)**2
                             + (self.robot_position.y - self.goal_pos_y)**2)
        
        if distance_to_goal <= (0.02):
            return True

        return False
    
    def check_collision_to_obstacle(self):
        for obs in self.obstacle:
            distance_to_obstacle = math.sqrt((self.robot_position.x - obs[0])**2 + (self.robot_position.y - obs[1])**2)
            if distance_to_obstacle <= 0.04408 + 0.02:
                # print(f"obstacle(x, y): {obs[0]}, {obs[1]}")
                # print("distance_to_obstacle: ", distance_to_obstacle)
                return True
            # if distance_to_obstacle <= 0.08:
                # print(f"0.8 以下 obstacle(x, y): {obs[0]}, {obs[1]}")
                # print("distance_to_obstacle: ", distance_to_obstacle)

        return False
    
    # 環境のリセット
    def reset(self, seed=None):
        rospy.sleep(0.01)
        if seed is not None:
            random.seed(seed)
        
        # ロボットの速度を停止
        twist = Twist()
        twist.linear = Vector3(x=0, y=0, z=0)
        twist.angular = Vector3(x=0, y=0, z=0)
        try:
            for i in range(self.robot_num):
                self.cmd_vel_pub[i].publish(twist)
        except rospy.ServiceException as e:
            print("[def reset] : {0}".format(e))
        rospy.sleep(1.0)


        # マーカーを削除
        self.delete_all_markers()

        # ロボットの位置リセット
        self.set_initialize_robots()
        self.respawn_robots()

        # ロボットの色をリセット
        self.set_initialize_robots_color()

        ## ロボットの位置がリセットされるまで待機
        rospy.sleep(1.0)

        # ゴールの初期位置を設定
        self.set_goals()

        # 新しいゴールマーカーを設定
        self.set_goal_marker()


        # フェロモンマップをリセット
        self.reset_pheromone_pub.publish(EmptyMsg())
        rospy.sleep(2.0)
        
        # フラグのリセット
        self.is_collided = [False] * self.robot_num
        self.is_goal = [False] * self.robot_num
        self.is_timeout = [False] * self.robot_num

        # 変数の初期化
        self.reward = [None] * self.robot_num
        # self.distance_to_goal = math.sqrt((self.robot_position.x-self.goal_pos_x)**2
        #                      + (self.robot_position.y-self.goal_pos_y)**2)
        # self.reset_timer = rospy.get_time()

        # # ロボットの現在の体の向きのベクトルとロボットの現在の位置からゴールまでのベクトルのなす角度
        # angle_to_goal = math.atan2(self.goal_pos_y - self.robot_position.y,
        #                            self.goal_pos_x - self.robot_position.x) - self.robot_angle
        # ## 角度を-πからπの範囲に正規化
        # if angle_to_goal < -math.pi:
        #     angle_to_goal += 2 * math.pi
        # elif angle_to_goal > math.pi:
        #     angle_to_goal -= 2 * math.pi

        # # previous distance to goal の初期値をidごとの原点からゴールまでの距離に設定
        # self.prev_distance_to_goal = self.distance_to_goal

        # self.done = False
        # self.state = list(self.pheromone_value) + [self.distance_to_goal, angle_to_goal,self.robot_linear_velocity.x, self.robot_angular_velocity.z]

        # return self.state
    

    def set_goals(self):
        for i in range(self.robot_num):
            # ゴールを他のロボットの位置に設定
            other_robot_index = (i + 1) % self.robot_num  # 他のロボットのインデックスを取得
            self.goal_pos_x[i] = self.robot_position[other_robot_index].x
            self.goal_pos_y[i] = self.robot_position[other_robot_index].y
    
    
    def set_initialize_robots(self):
        robot_r = 0.8  # 半径 0.8 の円

        # 最初のロボットの位置をランダムに選択
        angle = 2.0 * math.pi * random.random()
        x1 = robot_r * math.cos(angle) + self.origin_x
        y1 = robot_r * math.sin(angle) + self.origin_y
        self.robot_position[0] = Vector3(x=x1, y=y1, z=0)

        # 二番目のロボットの位置を選ぶ
        min_distance = 0.3  # 最小距離
        while True:
            angle = 2.0 * math.pi * random.random()
            x2 = robot_r * math.cos(angle) + self.origin_x
            y2 = robot_r * math.sin(angle) + self.origin_y

            # 二つのロボットの距離を計算
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if distance >= min_distance:
                break  # 条件を満たした場合はループを抜ける

        self.robot_position[1] = Vector3(x=x2, y=y2, z=0)

    def respawn_robots(self):
        for i in range(self.robot_num):
            state_msg = ModelState()
            state_msg.model_name = self.robot_name[i]
            state_msg.pose.position.x = self.robot_position[i].x
            state_msg.pose.position.y = self.robot_position[i].y
            state_msg.pose.position.z = 0.2395

            # ランダムな角度をラジアンで生成
            yaw = random.uniform(0, 2 * math.pi)

            # 四元数への変換
            qx = 0.0
            qy = 0.0
            qz = math.sin(yaw / 2)
            qw = math.cos(yaw / 2)

            state_msg.pose.orientation.x = 0.0
            state_msg.pose.orientation.y = 0.0
            state_msg.pose.orientation.z = qz
            state_msg.pose.orientation.w = qw
            state_msg.twist.linear.x = 0.0
            state_msg.twist.linear.y = 0.0
            state_msg.twist.linear.z = 0.0
            state_msg.twist.angular.x = 0.0
            state_msg.twist.angular.y = 0.0
            state_msg.twist.angular.z = 0.0

            try:
                self.set_model_state(state_msg)
            except rospy.ServiceException as e:
                print("[def respawn_robots]: {0}".format(e))

    def set_initialize_robots_color(self):
        for i in range(self.robot_num):
            # ロボットの色をリセット
            self.robot_color[i] = "CYEAN"
            color = ColorRGBA()
            color.r = 0
            color.g = 160
            color.b = 233
            color.a = 255
            try:
                self.pub_led[i].publish(color)
            except rospy.ServiceException as e:
                print("[def set_initialize_robots_color]: {0}".format(e))


    def shutdown(self):
        """
        Shuts down the ROS node.
        """

        twist = Twist()
        twist.linear = Vector3(x=0, y=0, z=0)
        twist.angular = Vector3(x=0, y=0, z=0)
        try:
            for i in range(self.robot_num):
                self.cmd_vel_pub[i].publish(twist)
        except rospy.ServiceException as e:
            print("[def shutdown]: {0}".format(e))


        rospy.signal_shutdown("Closing Gazebo environment")
        rospy.spin()



    def set_goal_marker(self):
        """
        Set a goal marker in the Gazebo world and Rviz.
        """

        for i in range(self.robot_num):
            # Rviz
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()

            marker.ns = f"goal_{self.id+i}"
            marker.id = self.id * 1000 + 100 + i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            marker.pose.position.x = self.goal_pos_x[i]
            marker.pose.position.y = self.goal_pos_y[i]
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



    def delete_all_markers(self):
        # すべてのマーカーを削除するためのマーカーメッセージを作成
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL

        # マーカーをパブリッシュ
        self.marker_pub.publish(delete_marker)

    def stop_robot(self):
        # ロボットの速度を停止
        twist = Twist()
        twist.linear = Vector3(x=0, y=0, z=0)
        twist.angular = Vector3(x=0, y=0, z=0)
        try:
            for i in range(self.robot_num):
                self.cmd_vel_pub[i].publish(twist)
        except rospy.ServiceException as e:
            print("[def stop_robot]: {0}".format(e))
