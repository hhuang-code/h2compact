#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import WrenchStamped
from collections import deque
import time

class ATIZeroOffset:
    def __init__(self):
        rospy.init_node('ati_zero_offset', anonymous=True)
        
        # Parameters
        self.calibration_time = 3.0  # seconds
        self.sample_window = 3000    # number of samples to collect
        
        # Initialize offset values for each sensor
        self.mini40_force_offset = np.zeros(3)
        self.mini40_torque_offset = np.zeros(3)
        self.mini45_force_offset = np.zeros(3)
        self.mini45_torque_offset = np.zeros(3)
        
        # Initialize sample buffers for each sensor
        self.mini40_force_samples = deque(maxlen=self.sample_window)
        self.mini40_torque_samples = deque(maxlen=self.sample_window)
        self.mini45_force_samples = deque(maxlen=self.sample_window)
        self.mini45_torque_samples = deque(maxlen=self.sample_window)
        
        # Flag to track calibration state
        self.is_calibrated = False
        self.calibration_start_time = None
        
        # Create subscribers and publishers for both sensors
        # Mini 40
        self.mini40_sub = rospy.Subscriber('/ati_ros_ati_mini40/ati_mini40/ft_meas', WrenchStamped, 
                                         self.mini40_callback)
        self.mini40_pub = rospy.Publisher('/ati_ros_ati_mini40/ati_mini40/ft_meas_zeroed', 
                                        WrenchStamped, queue_size=10)
        
        # Mini 45
        self.mini45_sub = rospy.Subscriber('/ati_ros_ati_mini45/ati_mini45/ft_meas', WrenchStamped, 
                                         self.mini45_callback)
        self.mini45_pub = rospy.Publisher('/ati_ros_ati_mini45/ati_mini45/ft_meas_zeroed', 
                                        WrenchStamped, queue_size=10)
        
        rospy.loginfo("ATI Zero Offset node initialized. Starting calibration...")
        self.calibration_start_time = time.time()

    def mini40_callback(self, msg):
        if not self.is_calibrated:
            self.collect_samples(msg, self.mini40_force_samples, self.mini40_torque_samples)
        else:
            self.publish_zeroed_data(msg, self.mini40_pub, "ati_mini40", 
                                   self.mini40_force_offset, self.mini40_torque_offset)

    def mini45_callback(self, msg):
        if not self.is_calibrated:
            self.collect_samples(msg, self.mini45_force_samples, self.mini45_torque_samples)
        else:
            self.publish_zeroed_data(msg, self.mini45_pub, "ati_mini45", 
                                   self.mini45_force_offset, self.mini45_torque_offset)

    def collect_samples(self, msg, force_samples, torque_samples):
        # Collect force and torque samples
        force = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
        torque = np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
        
        force_samples.append(force)
        torque_samples.append(torque)
        
        # Check if calibration time has elapsed
        if time.time() - self.calibration_start_time >= self.calibration_time:
            self.calculate_offset()
            self.is_calibrated = True
            rospy.loginfo("Calibration complete! Offsets calculated.")

    def calculate_offset(self):
        # Calculate mean of collected samples for Mini 40
        mini40_force_array = np.array(list(self.mini40_force_samples))
        mini40_torque_array = np.array(list(self.mini40_torque_samples))
        
        self.mini40_force_offset = np.mean(mini40_force_array, axis=0)
        self.mini40_torque_offset = np.mean(mini40_torque_array, axis=0)
        
        # Calculate mean of collected samples for Mini 45
        mini45_force_array = np.array(list(self.mini45_force_samples))
        mini45_torque_array = np.array(list(self.mini45_torque_samples))
        
        self.mini45_force_offset = np.mean(mini45_force_array, axis=0)
        self.mini45_torque_offset = np.mean(mini45_torque_array, axis=0)
        
        rospy.loginfo("Mini 40 Offsets:")
        rospy.loginfo(f"Force offset: {self.mini40_force_offset}")
        rospy.loginfo(f"Torque offset: {self.mini40_torque_offset}")
        
        rospy.loginfo("Mini 45 Offsets:")
        rospy.loginfo(f"Force offset: {self.mini45_force_offset}")
        rospy.loginfo(f"Torque offset: {self.mini45_torque_offset}")

    def publish_zeroed_data(self, msg, publisher, frame_id, force_offset, torque_offset):
        # Create new message with zeroed values
        zeroed_msg = WrenchStamped()
        zeroed_msg.header = msg.header
        zeroed_msg.header.frame_id = frame_id
        
        # Apply offset to force
        zeroed_msg.wrench.force.x = msg.wrench.force.x - force_offset[0]
        zeroed_msg.wrench.force.y = msg.wrench.force.y - force_offset[1]
        zeroed_msg.wrench.force.z = msg.wrench.force.z - force_offset[2]
        
        # Apply offset to torque
        zeroed_msg.wrench.torque.x = msg.wrench.torque.x - torque_offset[0]
        zeroed_msg.wrench.torque.y = msg.wrench.torque.y - torque_offset[1]
        zeroed_msg.wrench.torque.z = msg.wrench.torque.z - torque_offset[2]
        
        # Publish zeroed data
        publisher.publish(zeroed_msg)

if __name__ == '__main__':
    try:
        ati_zero = ATIZeroOffset()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 