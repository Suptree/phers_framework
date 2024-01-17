#! /usr/bin/env python3

import rospy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Empty
import math
from gazebo_msgs.msg import ModelStates
import numpy as np
import time
from pathlib import Path
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import tf
import sys

class PheromoneFramework:
    def __init__(self, id):
        self.id = id
        self.robot_name = f"hero_{id}" # hero_0
        self.robot_radius = 0.04408

        # 原点座標
        self.origin_x = float((int(self.id) % 4) * 20.0) 
        self.origin_y = float(int(int(self.id) / 4) * 20.0)

        # pheromonクラスのインスタンス生成
        self.pheromone = Pheromone(
            grid_map_size=40, resolution=50, evaporation=0.0, diffusion=0.0
        )

        # フェロモンの最大値と最小値を設定
        self.max_pheromone_value = 1.0
        self.min_pheromone_value = 0.0

        # 経過時間
        self.start_time = rospy.get_time()
        # 最後に受け取った model_states のデータを保存する属性
        self.latest_model_states = None
        
        # オブジェクトの周りにフェロモンを配置
        # Refactor repetitive code
        # 障害物の名前リスト


        self.obstacles = [
            (self.origin_x + 0.0,    self.origin_y + 0.4),
            (self.origin_x + 0.0,    self.origin_y + (-0.4)),
            (self.origin_x + 0.4,    self.origin_y + 0.0),
            (self.origin_x + (-0.4), self.origin_y + 0.0)
        ]
        
        # for obs in self.obstacles:
        #     x_index, y_index = self.posToIndex(obs[0], obs[1])
        #     self.pheromone.injectionCircle(x_index, y_index, self.max_pheromone_value, 0.06)

        # Publisher & Subscriber
        # フェロモン値を送信
        self.publish_pheromone = rospy.Publisher(
            f"/{self.robot_name}/pheromone_value", Float32MultiArray, queue_size=1
        )
        # gazeboの環境上にあるオブジェクトの状態を取得
        # 取得すると, pheromoneCallback関数が呼び出される
        self.subscribe_pose = rospy.Subscriber(
            "/gazebo/model_states", ModelStates, self.pheromoneCallback
        )
        # リセット信号のSubscriber
        self.subscribe_reset = rospy.Subscriber(
            f"/{self.robot_name}/pheromone_reset_signal", Empty, self.resetPheromoneMap
        )

        self.marker_pub = rospy.Publisher(
            f"/{self.robot_name}/visualization_marker_array", MarkerArray, queue_size=10
        )

    
    def publish_markers(self):
        markerArray = MarkerArray()
        id = 1
        for i in range(self.pheromone.num_cell):
            for j in range(self.pheromone.num_cell):
                
                if self.pheromone.grid[i][j] > 0:
                    marker = Marker()
                    marker.header.frame_id = "world"
                    marker.header.stamp = rospy.Time.now()
                    marker.ns = "pheromone"
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD

                    marker.pose.position.z = 0.02
                    marker.pose.orientation.x = 0.0
                    marker.pose.orientation.y = 0.0
                    marker.pose.orientation.z = 0.0
                    marker.pose.orientation.w = 1.0

                    marker.scale.x = 1.0 / self.pheromone.resolution
                    marker.scale.y = 1.0 / self.pheromone.resolution
                    marker.scale.z = 0.01

                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                    marker.color.a = self.pheromone.grid[i][j]

                    (
                        marker.pose.position.x,
                        marker.pose.position.y,
                    ) = self.indexToPos(i, j)
                    # marker.pose.position.x += 0.01
                    # marker.pose.position.y += 0.01
                    marker.id = self.id * 1000 + 100 + id
                    id += 1
                    markerArray.markers.append(marker)

        self.marker_pub.publish(markerArray)

    # 座標からフェロモングリッドへ変換
    def posToIndex(self, x, y):
        # x_index = math.floor((x - self.origin_x + self.pheromone.grid_map_size/2.0) * self.pheromone.resolution)
        # y_index = math.floor((y - self.origin_y + self.pheromone.grid_map_size/2.0) * self.pheromone.resolution)
        x_index = math.floor((x - self.origin_x + ((1.0/self.pheromone.resolution)/2.0) + self.pheromone.grid_map_size/2.0) * self.pheromone.resolution)
        y_index = math.floor((y - self.origin_y + ((1.0/self.pheromone.resolution)/2.0) + self.pheromone.grid_map_size/2.0) * self.pheromone.resolution)
        # print("x: {}, y: {}".format(x, y))
        # print("x - origin_x: {}, y - origin_y: {}".format(x - self.origin_x, y - self.origin_y))
        # print("x_index: {}, y_index: {}".format(x_index, y_index))
        if (
            x_index < 0
            or y_index < 0
            or x_index > self.pheromone.num_cell - 1
            or y_index > self.pheromone.num_cell - 1
        ):
            raise Exception("The pheromone matrix index is out of range.")
        return x_index, y_index    # フェロモングリッドから座標へ変換
    
    
    # セルの中央を返す
    def indexToPos(self, x_index, y_index): 
        # x = (x_index / self.pheromone.resolution) + self.origin_x -  (self.pheromone.grid_map_size / 2.0)
        # y = (y_index / self.pheromone.resolution) + self.origin_y - (self.pheromone.grid_map_size / 2.0)
        x = (x_index -  (self.pheromone.resolution * self.pheromone.grid_map_size / 2.0)) * (1.0 / self.pheromone.resolution) + self.origin_x
        y = (y_index -  (self.pheromone.resolution * self.pheromone.grid_map_size / 2.0)) * (1.0 / self.pheromone.resolution) + self.origin_y

        return x, y
    
    def pheromoneCallback(self, model_status):

        try:
            self.latest_model_states = model_status
            # Reading from arguments
            robot_index = model_status.name.index(self.robot_name)
            # print(model_status.pose)
            # print(model_status.twist)
            pose = model_status.pose[robot_index]
            # twist = model_status.twist[robot_index]
            pos = pose.position
            ori = pose.orientation

            angles = tf.transformations.euler_from_quaternion(
                (ori.x, ori.y, ori.z, ori.w))
            self.theta = angles[2]
            pheromone_value = Float32MultiArray()

            # ロボットの現在の位置
            current_x = pos.x
            current_y = pos.y
            angles = [math.pi/4, 0, -math.pi/4, math.pi/2, -math.pi/2, 3*math.pi/4, math.pi, -3*math.pi/4]
            # 8方向の座標を計算
            directions = []
            debug_index = []
            for i, angle in enumerate(angles):
                # ロボットの向きに応じた角度を考慮
                adjusted_angle = self.theta + angle

                # X, Y座標を計算
                dir_x = current_x + self.robot_radius * math.cos(adjusted_angle)
                dir_y = current_y + self.robot_radius * math.sin(adjusted_angle)
                # print(f"angle degree: {angle}, adjusted_angle: {adjusted_angle}, dir_x: {dir_x}, dir_y: {dir_y}")
                #　度数法でprint
                # print(f"angle degree: {angle * 180 / math.pi}, adjusted_angle: {adjusted_angle * 180 / math.pi}, dir_x: {dir_x}, dir_y: {dir_y}")
                

                directions.append((dir_x, dir_y))
                x_index, y_index = self.posToIndex(dir_x, dir_y)
                debug_index.append((x_index, y_index))
                pheromone_value.data.append(
                    self.pheromone.getPheromone(x_index, y_index)
                )

                # 現在のロボットの座標を4番目と5番目の要素の間に挿入
                if i == 3:
                    directions.append((current_x, current_y))
                    x_index, y_index = self.posToIndex(dir_x, dir_y)
                    debug_index.append((x_index, y_index))
                    pheromone_value.data.append(
                        self.pheromone.getPheromone(x_index, y_index)
                    )



            # x_index, y_index = self.posToIndex(pos.x, pos.y)
            # for j in reversed(range(3)):
            #     for i in range(3):
            #         pheromone_value.data.append(
            #             self.pheromone.getPheromone(x_index + i - 1, y_index + j - 1)
            #         )
        except Exception as e:
            # print("Error occured in pheromoneCallback")
            # print(e)
            pheromone_value = Float32MultiArray()
            for _ in range(9):
                pheromone_value.data.append(0.0)

        if not rospy.is_shutdown():
            self.publish_pheromone.publish(pheromone_value)

    def resetPheromoneMap(self, msg):
        if self.latest_model_states:
            try:
                obstacle_names = [f"obs_{self.id}1", f"obs_{self.id}2", f"obs_{self.id}3", f"obs_{self.id}4"]
                self.obstacles = []  # 障害物リストをリセット
                for obs_name in obstacle_names:
                    if obs_name in self.latest_model_states.name:
                        obs_index = self.latest_model_states.name.index(obs_name)
                        obs_pose = self.latest_model_states.pose[obs_index]
                        obs_pos = obs_pose.position
                        self.obstacles.append((obs_pos.x, obs_pos.y))

            except Exception as e:
                rospy.logerr("Error updating obstacles in resetPheromoneMap: {}".format(e))

        self.pheromone.reset()
        # 経過時間
        self.start_time = rospy.get_time()

        # オブジェクトの周りにフェロモンを配置
        # Refactor repetitive code
        for obs in self.obstacles:
            x_index, y_index = self.posToIndex(obs[0], obs[1])
            self.pheromone.injectionCircle(x_index, y_index, self.max_pheromone_value, 0.30)

        self.publish_markers()


        
class Pheromone:
    def __init__(self, grid_map_size=0, resolution=0, evaporation=0.0, diffusion=0.0):
        # グリッド地図の生成
        # map size = 1 m * size
        self.grid_map_size = grid_map_size
        # grid cell size = 1 m / resolution
        self.resolution = resolution
        # 1辺におけるグリッドセルの合計個数
        self.num_cell = self.resolution * self.grid_map_size + 1
        # 例外処理 : 一辺におけるグリッドセルの合計個数を必ず奇数でなければならない
        if self.num_cell % 2 == 0:
            raise Exception("Number of cell is even. It needs to be an odd number")
        self.grid = np.zeros((self.num_cell, self.num_cell))
        self.grid_copy = np.zeros((self.num_cell, self.num_cell))

        # 蒸発パラメータの設定
        self.evaporation = evaporation
        if self.evaporation == 0.0:
            self.isEvaporation = False
        else:
            self.isEvaporation = True

        # 拡散パラメータの設定
        self.diffusion = diffusion
        if self.diffusion == 0.0:
            self.isDiffusion = False
        else:
            self.isDiffusion = True

        # Timers
        self.update_timer = rospy.get_time()
        self.step_timer = rospy.get_time()
        self.injection_timer = rospy.get_time()
        self.save_timer = rospy.get_time()
        self.reset_timer = rospy.get_time()

    # 指定した座標(x, y)からフェロモンを取得
    def getPheromone(self, x, y):
        return self.grid[x, y]

    # 指定した座標(x, y)へフェロモンを配置
    def setPheromone(self, x, y, value):
        self.grid[x, y] = value

    # 正方形の形でフェロモンの射出する
    def injectionSquare(
        self, x, y, pheromone_value, injection_size, max_pheromone_value
    ):
        # 例外処理 : フェロモンを射出するサイズはかならず奇数
        if injection_size % 2 == 0:
            raise Exception("Pheromone injection size must be an odd number.")
        # 現在時刻を取得
        current_time = rospy.get_time()
        # フェロモンを射出する間隔を0.1s間隔に設定
        if current_time - self.injection_timer > 0.1:
            for i in range(injection_size):
                for j in range(injection_size):
                    # 指定した範囲のセルにフェロモンを配置
                    self.grid[
                        x - (injection_size - 1) / 2 + i,
                        y - (injection_size - 1) / 2 + j,
                    ] += pheromone_value
                    # 配置するフェロモン値がフェロモンの最大値より大きければカット
                    if (
                        self.grid[
                            x - (injection_size - 1) / 2 + i,
                            y - (injection_size - 1) / 2 + j,
                        ]
                        >= max_pheromone_value
                    ):
                        self.grid[
                            x - (injection_size - 1) / 2 + i,
                            y - (injection_size - 1) / 2 + j,
                        ] = max_pheromone_value
            # フェロモンを射出した時間を記録
            self.injection_timer = current_time

    def injectionCircle(self, x_index, y_index, max_value, radius):
        radius = int(radius * self.resolution)
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                distance_squared = i**2 + j**2
                if distance_squared <= radius**2:
                    # 中心からの距離を計算
                    distance = np.sqrt(distance_squared)

                    # 中心に近いほど値が大きく、外縁に近いほど値が小さくなるようにする
                    # 例: 線形減衰（max_value から 0 まで線形に減少）
                    value = max_value * (1 - distance / radius)

                    self.grid[x_index + i, y_index + j] = self.grid[x_index + i, y_index + j] + value
    def update(self, min_pheromone_value, max_pheromone_value):
        current_time = rospy.get_time()
        time_elapsed = current_time - self.update_timer
        self.update_timer = current_time

        # 拡散を行うかどうかの判定
        if self.isDiffusion is True:
            # Diffusion
            for i in range(self.num_cell):
                for j in range(self.num_cell):
                    self.grid_copy[i, j] += 0.9 * self.grid[i, j]
                    if i >= 1:
                        self.grid_copy[i - 1, j] += 0.025 * self.grid[i, j]
                    if j >= 1:
                        self.grid_copy[i, j - 1] += 0.025 * self.grid[i, j]
                    if i < self.num_cell - 1:
                        self.grid_copy[i + 1, j] += 0.025 * self.grid[i, j]
                    if j < self.num_cell - 1:
                        self.grid_copy[i, j + 1] += 0.025 * self.grid[i, j]
            # 最大と最小を丸める
            for i in range(self.num_cell):
                for j in range(self.num_cell):
                    if self.grid_copy[i, j] < min_pheromone_value:
                        self.grid_copy[i, j] = min_pheromone_value
                    elif self.grid_copy[i, j] > max_pheromone_value:
                        self.grid_copy[i, j] = max_pheromone_value
            # グリッドセルを更新
            self.grid = np.copy(self.grid_copy)
            # 複製したグリッドセルを初期化
            self.grid_copy = np.zeros((self.num_cell, self.num_cell))

        # 蒸発を行うかどうかの判定
        if self.isEvaporation is True:
            # Evaporation
            decay = 2 ** (-time_elapsed / self.evaporation)
            for i in range(self.num_cell):
                for j in range(self.num_cell):
                    self.grid[i, j] = decay * self.grid[i, j]

    def save(self, file_name):
        parent = Path(__file__).resolve().parent
        with open(
            parent.joinpath("pheromone_saved/" + file_name + ".pheromone"), "wb"
        ) as f:
            np.save(f, self.grid)
        # print("The pheromone matrix {} is successfully saved".
        # format(file_name))

    def load(self, file_name):
        parent = Path(__file__).resolve().parent
        with open(parent.joinpath("pheromone_saved/" + file_name + ".npy"), "rb") as f:
            self.grid = np.load(f)
        # print("The pheromone matrix {} is successfully loaded".
        #       format(file_name))
    def reset(self):
        self.grid = np.zeros((self.num_cell, self.num_cell))
        self.grid_copy = np.zeros((self.num_cell, self.num_cell))


if __name__ == "__main__":
    # コマンドライン引数を確認
    if len(sys.argv) < 2:
        print("Please provide the number of parallel environments")
        sys.exit(1)

    # 引数を取得
    num_env = int(sys.argv[1])

    rospy.init_node(f"hero_{num_env}_Pheromone_Framework", anonymous=True,)

    node = PheromoneFramework(id=num_env)
    rospy.spin()
