#! /usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan
import math
import numpy as np

class LaserProcessingNode:
    def __init__(self):
        rospy.init_node('laser_processing_node', anonymous=True)
        self.laser_sub = rospy.Subscriber('/hero_0/laser', LaserScan, self.laser_callback)

    def laser_callback(self, data):
        angles = [math.pi/4, 0, -math.pi/4, math.pi/2, -math.pi/2, 3*math.pi/4, math.pi, -3*math.pi/4]
        angle_range = 22.5 * (math.pi / 180)  # ±22.5度をラジアンに変換

        avg_distances = []
        for base_angle in angles:
            sector_start = base_angle - angle_range
            sector_end = base_angle + angle_range
            distances = self.get_sector_distances(data, sector_start, sector_end)
            avg_distance = sum(distances) / len(distances)
            avg_distances.append(avg_distance)

        # 結果の表示（デバッグ用）
        print("Average distances for each sector:", avg_distances)

    def get_sector_distances(self, data, start_angle, end_angle):
        # スキャンデータからセクターの距離を抽出
        num_points = len(data.ranges)
        distances = []
        for i in range(num_points):
            current_angle = data.angle_min + i * data.angle_increment
            if start_angle <= current_angle <= end_angle or start_angle <= current_angle + 2 * math.pi <= end_angle:
                distance = data.ranges[i]
                if distance == float('inf'):
                    distance = 0
                distances.append(distance)
        return distances

if __name__ == '__main__':
    try:
        node = LaserProcessingNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
