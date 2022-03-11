#! /usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64, Bool
from geometry_msgs.msg import Twist
import lane_detection_lib
import numpy as np
import cv2
from cv_bridge import CvBridge


HEADERS = ""
bridge = CvBridge()
IMAGE_TOPIC_NAME = "/camera/color/image_raw"
FIXED_LINEAR_VEL = 0.9
rospy.init_node("lane_follower_node", anonymous=False)

move_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=5)
img_pub = rospy.Publisher("/lanes", Image, queue_size=2)

pid_state_pub = rospy.Publisher("/lane_following_pid/state", Float64, queue_size=2)
pid_setpoint_pub = rospy.Publisher("/lane_following_pid/setpoint", Float64, queue_size=2)
pid_enable = rospy.Publisher("/lane_following_pid/pid_enable", Bool, queue_size=1)

def control_callback(data: Float64):
    twist_msg = Twist()
    twist_msg.linear.x = FIXED_LINEAR_VEL
    twist_msg.angular.z = data.data
    move_pub.publish(twist_msg)

def img_recvd(data: Image):
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    img, ang, lane_found = lane_detection_lib.parse_image(cv_image)
    image_message = bridge.cv2_to_imgmsg(img, encoding="passthrough")
    image_message.header = data.header
    img_pub.publish(image_message)
    enable_msg = Bool()
    if lane_found:
        enable_msg.data = True
        state_msg = Float64()
        setpoint_msg = Float64()
        state_msg.data = float(ang)
        setpoint_msg.data = 90.0
        pid_state_pub.publish(state_msg)
        pid_setpoint_pub.publish(setpoint_msg)
    else:
        enable_msg.data = False
        twist_msg = Twist()
        move_pub.publish(twist_msg)
    
    pid_enable.publish(enable_msg)


if __name__ == "__main__":
    rospy.Subscriber(IMAGE_TOPIC_NAME, Image, img_recvd)
    rospy.Subscriber("/lane_following_pid/control_effort", Float64, control_callback)
    rospy.spin()    