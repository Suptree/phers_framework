#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import ColorRGBA
from gazebo_msgs.msg import ModelStates
import math
import tf
import time
from geometry_msgs.msg import Twist
import random
from visualization_msgs.msg import Marker
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
import statistics

class WaypointNavigation:
    # 最大速度 0.1 m/s
    MAX_FORWARD_SPEED = 0.1
    MAX_ROTATION_SPEED = 1.0

    # ロボットに発信する速度・角速度変数を定義
    cmdmsg = Twist()
    index = 0

    # Tunable parameters
    wGain = 1.0
    vConst = 0.06
    distThr = 0.02
    pheroThr = 0.3

    def __init__(self):
        # Initialise pheromone values
        self.pheromone_value = [0.0] * 9
        self.sum_pheromone_value = 0.0

        self.robot_theta = 0
        self.color = "BLUE"

        # Goalをランダムで決定する
        # goal_r = 0.8
        # goal_radius = 2.0 * math.pi * random.random()
        # print("goal_raius = {}".format(math.degrees(goal_radius)))
        # self.goal_pos_x = goal_r * math.cos(goal_radius)
        # self.goal_pos_y = goal_r * math.sin(goal_radius)
        # print("Goal Position = ({}, {})".format(
        #     self.goal_pos_x, self.goal_pos_y))
        # Goalを手動で決める
        self.goal_pos_x = -1.0
        self.goal_pos_y = 0.0

        # 障害物の位置
        self.obstacle = [[0.4, 0.0], [-0.4, 0.0], [0.0, 0.4], [0.0, -0.4]]

        # 実験に必要な変数
        self.counter_step = 0       # 実験の試行回数
        self.counter_collision = 0  # 衝突回数
        self.counter_success = 0    # タスク成功回数
        self.arrival_time = []
        # Flags
        self.is_collided = False
        self.is_goal = False
        self.is_timeout = False



        # ゴールしたかどうか判定
        # self.arrived = False
        # self.collied = False
        # self.timeuped = False
        # self.counted = False
        # self.reseted = False

        self.marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)


        self.sub_phero = rospy.Subscriber(
            '/pheromone_value', Float32MultiArray, self.ReadPheromone)
        self.sub = rospy.Subscriber(
            '/gazebo/model_states', ModelStates, self.Callback)
        self.pub = rospy.Publisher('/hero_0/cmd_vel', Twist, queue_size=1)
        self.pub_led = rospy.Publisher('/hero_0/led', ColorRGBA, queue_size=1)
        self.beta_const = 1.2
        self.sensitivity = 1.2
        self.BIAS = 0.25
        self.V_COEF = 0.6  # self.v_range[0]
        self.W_COEF = 0.4  # self.w_range[0]

        self.set_goal_marker(self.goal_pos_x, self.goal_pos_y)
        # Initialise simulation
        self.reset_timer = rospy.get_time()
        self.reset()
        self.reset_flag = False


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

    def ReadPheromone(self, pheromone_message):
        pheromone_data = pheromone_message.data
        self.pheromone_value = pheromone_data
        self.sum_pheromone_value = np.sum(np.asarray(pheromone_data))

    def Callback(self, model_status):
        # if( self.is_goal == True):
        # if(self.counter_step >= 20 and self.arrived == True):
        # print("counter_step : {}, counter_collision : {}, counter_success: {}, average_time : {}, times : {}".format(
        #     self.counter_step,
        #     self.counter_collision,
        #     self.counter_success,
        #     self.arrival_times.mean(),
        #     self.arrival_times

        # ))
            
            # return
        # if(rospy.get_time() - self.arrival_time > 40.0):
        #     if(self.counted == False):
        #         self.counter_step += 1
        #         self.counted = True
        #     self.reset()
        robot_index = model_status.name.index('hero_0')

        pose = model_status.pose[robot_index]

        pos = pose.position
        # if(pos.x >= -0.01 and pos.x <= 0.01 and pos.y >= -0.01 and pos.y <= 0.01):
        #     self.is_goal = False
        #     print("genntenn modotta!!")
        ori = pose.orientation

        angles = tf.transformations.euler_from_quaternion(
            (ori.x, ori.y, ori.z, ori.w))

        self.theta = angles[2]

        # P controller
        v = 0
        w = 0
        distance = math.sqrt((pos.x-self.goal_pos_x)**2
                             + (pos.y-self.goal_pos_y)**2)

        # Reset condition reset (to prevent unwanted reset due to delay of position message subscription)
        step_timer = rospy.get_time()
        reset_time = step_timer - self.reset_timer

        msg = Twist()
        if(distance <= self.distThr and reset_time > 1):
            if self.color != "YELLOW":
                self.color = "YELLOW"
                color = ColorRGBA()
                color.r = 255
                color.g = 255
                color.b = 0
                color.a = 255
                self.pub_led.publish(color)

            self.is_goal = True
            msg.linear.x = 0.0
            msg.angular.z = 1.0
            
            # if(self.counted == False):
            #     self.counted = True
            #     np.append(self.arrival_times,self.arrival_time)
            #     self.counter_step +=1
            #     self.counter_success +=1
            self.reset()

        elif (self.sum_pheromone_value > self.pheroThr):
            if self.color != "GREEN":
                self.color = "GREEN"
                color = ColorRGBA()
                color.r = 0
                color.g = 255
                color.b = 0
                color.a = 255
                self.pub_led.publish(color)

            msg = self.PheroOA(self.pheromone_value)
            v = msg.linear.x
            w = msg.angular.z

        # Adjust velocities
        elif (distance > self.distThr):
            # print("not pheromone\n")
            if self.color != "CYAN":
                self.color = "CYAN"
                color = ColorRGBA()
                color.r = 0
                color.g = 100
                color.b = 100
                color.a = 255
                self.pub_led.publish(color)


            v = self.vConst
            msg.linear.x = v

            yaw = math.atan2(self.goal_pos_y-pos.y, self.goal_pos_x-pos.x)
            u = yaw - self.theta

            # 追試の文章
            bound = math.atan2(math.sin(u), math.cos(u))
            w = min(1.0, max(-1.0, self.wGain*bound))
            msg.angular.z = w

            self.reset_flag = False

        # 衝突判定
        distance_to_obs = [1.0]*len(self.obstacle)
        for i in range(len(distance_to_obs)):
            distance_to_obs[i] = math.sqrt((pos.x-self.obstacle[i][0])**2+(pos.y-self.obstacle[i][1])**2)
        if (distance_to_obs[0] < 0.059 or distance_to_obs[1] < 0.059 or distance_to_obs[2] < 0.059 or distance_to_obs[3] < 0.059) and reset_time > 1:
            msg = Twist()
            self.is_collided = True
            print(distance_to_obs)
            # if(self.counted == False):
            #     self.counted = True
            #     self.counter_collision +=1
            #     self.counter_step +=1
            self.reset()

        if reset_time > 40.0:
            print(reset_time)
            print("Times up!")
            self.is_timeout = True
            self.reset()

        if msg.linear.x > self.MAX_FORWARD_SPEED:
            msg.linear.x = self.MAX_FORWARD_SPEED
        # Publish velocity
        self.pub.publish(msg)

        self.prev_x = pos.x
        self.prev_y = pos.y

        # Reporting
        # print("Distance to goal {}".format(distance))
        # print('Callback: x=%2.2f, y=%2.2f, dist=%4.2f, cmd.v=%2.2f, cmd.w=%2.2f' %(pos.x,pos.y,distance,v,w))

    def velCoef(self, value1, value2):
        '''
        - val_avg (0, 1)
        - val_dif (-1, 1)
        - dif_coef (1, 2.714)
        - coefficient (-2.714, 2.714)
        '''
        val_avg = (value1 + value2)/2
        val_dif = value1 - value2
        dif_coef = math.exp(val_avg)

        return dif_coef*val_dif

    def PheroOA(self, phero):
        '''
        Pheromone-based obstacle avoidance algorithm
        - Input: 9 cells of pheromone
        - Output: Twist() to avoid obstacle
        '''
        # Constants:
        # Constants:
        BIAS = self.BIAS
        V_COEF = self.V_COEF
        W_COEF = self.W_COEF
        # BIAS = 0.25
        # V_COEF = 0.2
        # W_COEF = 0.3

        # Initialise values
        # values are assigned from the top left (135 deg) to the bottom right (-45 deg) ((0,1,2),(3,4,5),(6,7,8))
        # print("pheromone_value = {}".format(phero))
        avg_phero = np.average(np.asarray(phero))
        # print("pheromone_average : %f" % avg_phero)
        unit_vecs = np.asarray(
            [[1, 0], [math.sqrt(2)/2, math.sqrt(2)/2], [0, 1], [-math.sqrt(2)/2, math.sqrt(2)/2]])
        vec_coefs = [0.0] * 4
        twist = Twist()

        # Calculate vector weights
        vec_coefs[0] = self.velCoef(phero[5], phero[3])
        vec_coefs[1] = self.velCoef(phero[2], phero[6])
        vec_coefs[2] = self.velCoef(phero[1], phero[7])
        vec_coefs[3] = self.velCoef(phero[0], phero[8])

        # print("vec_corf[0] : {}, vec_corf[0] : {}, vec_corf[0] : {}, vec_corf[0] : {}".format(
        # vec_coefs[0], vec_coefs[1], vec_coefs[2], vec_coefs[3]))
        vec_coefs = np.asarray(vec_coefs).reshape(4, 1)
        # print("vec_coefs = {}".format(vec_coefs))
        vel_vecs = np.multiply(unit_vecs, vec_coefs)
        # print("vel_vecs = {}".format(vel_vecs))
        vel_vec = np.sum(vel_vecs, axis=0)
        # print("vel_vec = {}".format(vel_vec))

        ang_vel = math.atan2(vel_vec[1], vel_vec[0])
        # print("angle_vel = {}".format(math.degrees(ang_vel)))

        # Velocity assignment
        twist.linear.x = BIAS + V_COEF*avg_phero
        twist.angular.z = ang_vel

        return twist

    def PheroResponse(self, phero):
        # takes two pheromone input from antennae
        avg_phero = np.average(np.asarray(phero))
        beta = self.beta_const - avg_phero
        s_l = beta + (phero[0] - phero[1])*self.sensitivity
        s_r = beta + (phero[1] - phero[0])*self.sensitivity
        twist = Twist()

        twist.linear.x = (s_l + s_r)/2
        twist.angular.z = (s_l - s_r)

        return twist
    
    def reset_goal(self):
        # Goalをランダムで決定する
        goal_r = 0.8
        goal_radius = 2.0 * math.pi * random.random()
        print("goal_raius = {}".format(math.degrees(goal_radius)))
        self.goal_pos_x = goal_r * math.cos(goal_radius)
        self.goal_pos_y = goal_r * math.sin(goal_radius)
        print("Goal Position = ({}, {})".format(
            self.goal_pos_x, self.goal_pos_y))
        # Goalを手動で決める
        # self.goal_pos_x = 0.0
        # self.goal_pos_y = 1.5

        self.set_goal_marker(self.goal_pos_x, self.goal_pos_y)
    
    def b_reset(self):
        # if(self.reseted == True):
        #     return
        self.reseted = True
        self.is_goal = True
        # self.arrival_time = rospy.get_time()
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

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy(
                '/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
            if resp.success:
                self.counted = False
                self.reseted = False

        except rospy.ServiceException as e:
            print("Service Call Failed: %s" % e)
        
        self.cmdmsg.linear.x = 0.0
        self.cmdmsg.angular.z = 0.0
        self.pub.publish(self.cmdmsg)
        self.reset_goal()


    def reset(self):
        
        '''
        Resettng the Experiment
        1. Update the counter based on the flag from step
        2. Assign next positions and reset
        3. Log the result in every selected time-step
        '''


        # ========================================================================= #
	    #                           COUNTER UPDATE                                  #
	    # ========================================================================= #

        # Increment Collision Counter
        if self.is_collided == True:
            print("Collision!")
            self.counter_collision += 1
            self.counter_step += 1

        # Increment Arrival Counter and store the arrival time
        if self.is_goal == True:
            print("Arrived goal!")
            self.counter_success += 1
            self.counter_step += 1
            arrived_timer = rospy.get_time()
            art = arrived_timer-self.reset_timer
            self.arrival_time.append(art)
            print("Episode time: %0.2f"%art)

        if self.is_timeout == True:
            self.counter_collision += 1
            self.counter_step += 1
            print("Timeout!")

        print("counter_step: {}, counter_success: {}, counter_collision: {}, Episode mean time: {}".format(self.counter_step, self.counter_success, self.counter_collision, statistics.mean(self.arrival_time)))
        # ========================================================================= #
	    #                                  RESET                                    #
	    # ========================================================================= #
            
        # Reset the flags
        self.is_collided = False
        self.is_goal = False
        self.is_timeout = False
        # self.arrival_time = rospy.get_time()
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

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy(
                '/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
            if resp.success:
                self.counted = False
                self.reseted = False

        except rospy.ServiceException as e:
            print("Service Call Failed: %s" % e)
        
        self.cmdmsg.linear.x = 0.0
        self.cmdmsg.angular.z = 0.0
        self.pub.publish(self.cmdmsg)
        self.reset_goal()
        
        self.reset_timer = rospy.get_time()
        self.reset_flag = True


if __name__ == '__main__':
    rospy.init_node('pose_reading')
    wayN = WaypointNavigation()
    # wayN.InformationMaker()
    rospy.spin()
