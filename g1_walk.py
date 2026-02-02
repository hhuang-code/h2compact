#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
import sys
import time
import yaml

# Import the Unitree SDK functions and classes
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

def cmd_vel_callback(msg, sport_client):
    """
    Callback function that reads the Twist message and sends motion commands
    to the Unitree robot via the sport_client.
    """
    # Extract linear and angular velocities
    forward = msg.linear.x    # Forward/backward motion
    lateral = msg.linear.y    # Lateral (sideways) motion
    rotate  = msg.angular.z   # Rotational velocity

    rospy.loginfo(f"Received cmd_vel: forward={forward}, lateral={lateral}, rotate={rotate}")

    # Apply a deadzone if needed
    if abs(forward) < 1e-3 and abs(lateral) < 1e-3 and abs(rotate) < 1e-3:
        # No significant command: stop the robot (or use a damp command)
        print("no cmd")
    else:
        # Send the motion command to the robot.
        # Note: The Move() function parameters are assumed to be in the order (forward, lateral, rotation)
        sport_client.Move(forward, lateral, rotate)

def main():
    if len(sys.argv) < 2:
        print(f"Usage: rosrun <your_package> {sys.argv[0]} <networkInterface>")
        sys.exit(-1)

    # Initialize ROS node
    rospy.init_node('unitree_cmd_vel_listener', anonymous=True)

    # Get network interface from command line argument
    network_interface = sys.argv[1]

    rospy.loginfo("Initializing channel and Unitree robot client...")
    # Initialize the channel (make sure to pass the correct network interface)
    ChannelFactoryInitialize(0, network_interface)

    # Create and initialize the LocoClient
    sport_client = LocoClient()
    sport_client.SetTimeout(10.0)
    sport_client.Init()

    rospy.loginfo("Unitree cmd_vel listener node started. Waiting for /cmd_vel messages...")

    # Subscribe to the /cmd_vel topic.
    # The lambda is used here to pass the sport_client to the callback.
    rospy.Subscriber('/cmd_vel', Twist, lambda msg: cmd_vel_callback(msg, sport_client))

    # Keep the node running
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
