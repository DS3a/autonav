#!/usr/bin/env python3

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
DEPTH_IMAGE_TOPIC_NAME = "/camera/depth/image_rect_raw"
# attested that both topics publish image of same shape


rospy.init_node("lane_follower_node", anonymous=False)

img_pub = rospy.Publisher("/lanes", Image, queue_size=2)
mask = None

def depth_img_recvd(data: Image):
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

    img = cv2.bitwise_and(cv_image, cv_image, mask=mask)
    image_message = bridge.cv2_to_imgmsg(img)
    image_message.header = data.header
    img_pub.publish(image_message)

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
    rospy.Subscriber(IMAGE_TOPIC_NAME, Image, img_recvd)
    rospy.Subscriber(DEPTH_IMAGE_TOPIC_NAME, Image, depth_img_recvd)
    rospy.spin()    