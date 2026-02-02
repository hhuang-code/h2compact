#!/usr/bin/env python
import rospy
from geometry_msgs.msg import WrenchStamped

# Module‐level vars
force_m40 = None  # (fx, fy, fz)
torque_m40 = None  # (tx, ty, tz)
time_m40 = None  # float seconds since epoch

force_m45 = None
torque_m45 = None
time_m45 = None


def wrench_callback(msg, topic_name):
    global force_m40, torque_m40, time_m40
    global force_m45, torque_m45, time_m45

    # unpack timestamp
    secs = msg.header.stamp.secs
    nsecs = msg.header.stamp.nsecs
    timestamp = secs + nsecs * 1e-9

    # unpack force & torque
    fx, fy, fz = (
        msg.wrench.force.x,
        msg.wrench.force.y,
        msg.wrench.force.z
    )
    tx, ty, tz = (
        msg.wrench.torque.x,
        msg.wrench.torque.y,
        msg.wrench.torque.z
    )

    if 'mini40' in topic_name:
        force_m40 = (fx, fy, fz)
        torque_m40 = (tx, ty, tz)
        time_m40 = timestamp
        rospy.loginfo("m40 @%.6f → fx=%.3f, fy=%.3f, fz=%.3f; tx=%.3f, ty=%.3f, tz=%.3f",
                      timestamp, fx, fy, fz, tx, ty, tz)
    elif 'mini45' in topic_name:
        force_m45 = (fx, fy, fz)
        torque_m45 = (tx, ty, tz)
        time_m45 = timestamp
        rospy.loginfo("m45 @%.6f → fx=%.3f, fy=%.3f, fz=%.3f; tx=%.3f, ty=%.3f, tz=%.3f",
                      timestamp, fx, fy, fz, tx, ty, tz)


def main():
    rospy.init_node('wrench_subscriber', anonymous=True)

    topics = [
        '/ati_ros_ati_mini40/ati_mini40/ft_meas_zeroed',
        '/ati_ros_ati_mini45/ati_mini45/ft_meas_zeroed'
    ]
    for t in topics:
        rospy.Subscriber(t, WrenchStamped, wrench_callback, callback_args=t)

    rospy.loginfo("Wrench subscriber started, listening to %d topics.", len(topics))
    rospy.spin()


if __name__ == '__main__':
    main()
