#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def main():
    rospy.init_node('realsense_rgb_v4l2_publisher')
    pub = rospy.Publisher('camera/rgb/image_raw', Image, queue_size=10)
    bridge = CvBridge()

    # Replace 2 with the actual device index for your RealSense color camera
    cap = cv2.VideoCapture(4, cv2.CAP_V4L2)
    if not cap.isOpened():
        rospy.logerr("Failed to open /dev/video2")
        return

    # Ensure OpenCV hands you RGB rather than BGR (some kernels convert automatically)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rospy.loginfo(f"V4L2 camera opened at {width}×{height}")

    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rospy.logwarn("Empty frame; retrying")
            continue

        try:
            # frame is already RGB if CAP_PROP_CONVERT_RGB=1; otherwise you'd cvtColor here
            img_msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            img_msg.header.stamp = rospy.Time.now()
            pub.publish(img_msg)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

        rate.sleep()

    cap.release()
    rospy.loginfo("Shutting down.")

if __name__ == '__main__':
    main()