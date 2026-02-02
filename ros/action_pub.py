import rospy
from geometry_msgs.msg import Twist


output = {'x': 0.5, 'y': 0.0, 'yaw': 0.1}  # example values


def publish_cmd_vel():
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rospy.init_node('cmd_vel_publisher', anonymous=True)
    rate = rospy.Rate(10)  # 10 Hz

    rospy.loginfo("Publishing /cmd_vel using output: %s", output)

    while not rospy.is_shutdown():
        twist_msg = Twist()
        twist_msg.linear.x = output['x']
        twist_msg.linear.y = output['y']
        twist_msg.angular.z = output['yaw']
        pub.publish(twist_msg)
        rate.sleep()


if __name__ == '__main__':
    try:
        publish_cmd_vel()
    except rospy.ROSInterruptException:
        pass