#!/usr/bin/env python3

import queue
import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float64, Bool
from geometry_msgs.msg import Twist
import lane_detection_lib
import numpy as np
import cv2
from cv_bridge import CvBridge


HEADERS = ""
bridge = CvBridge()
IMAGE_TOPIC_NAME = "/camera/color/image_raw"
DEPTH_IMAGE_TOPIC_NAME = "/camera/aligned_depth_to_color/image_raw"
# attested that both topics publish image of same shape


rospy.init_node("lane_follower_node", anonymous=False)

img_pub = rospy.Publisher("/lanes", Image, queue_size=2)
info_pub = rospy.Publisher("/filtered_points/camera_info", CameraInfo, queue_size=2)
mask = None

camera_info_message = None
info_found = False
def camera_info_recvd(data: CameraInfo):
    global camera_info_message, info_found
    info_found = True
    camera_info_message = data

def depth_img_recvd(data: Image):
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

    if mask is None:
        return

    print(cv_image.shape)
    print(mask.shape)

    img = cv2.bitwise_and(cv_image, cv_image, mask=mask)
    image_message = bridge.cv2_to_imgmsg(img)
    image_message.header = data.header
    img_pub.publish(image_message)
    info_pub.publish(camera_info_message)

def img_recvd(data: Image):
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    img, ang, lane_found = lane_detection_lib.parse_image(cv_image)

    img = lane_detection_lib.inverse_perspective_change(img)
    (thresh, img) = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    global mask
    mask = np.copy(img)
    if lane_found:
        # do inverse transform on mask and apply it to depth image and publish it to /lanes
        pass

if __name__ == "__main__":
    info_sub = rospy.Subscriber("/camera/aligned_depth_to_color/camera_info", CameraInfo, camera_info_recvd)
    if info_found:
        info_sub.unregister()
    rospy.Subscriber(IMAGE_TOPIC_NAME, Image, img_recvd)
    rospy.Subscriber(DEPTH_IMAGE_TOPIC_NAME, Image, depth_img_recvd)
    rospy.spin()    