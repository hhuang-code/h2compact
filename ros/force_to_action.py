#!/usr/bin/env python3
"""
force_to_action_node.py  –cache wrench data and feed the whole batch to the
model at 5Hz, then clear the cache for the next round.
"""
import threading
from typing import Dict, List, Tuple

import rospy
from geometry_msgs.msg import WrenchStamped, Twist

import pdb


# ───────────────────────────────────────────────────────────────────────────
# Shared state (guarded by cache_lock)
# ───────────────────────────────────────────────────────────────────────────
movement_cmd: Dict[str, float] = {"x": 0.0, "y": 0.0, "yaw": 0.0}
wrench_cache: List[Tuple[str, Tuple[float, ...]]] = []   # [(sensor, 6‑tuple), …]
cache_lock = threading.Lock()


# ───────────────────────────────────────────────────────────────────────────
# Dummy batch model: list[(sensor, 6‑D wrench)] → command
# Replace with real NN inference (e.g. Torch) if needed.
# ───────────────────────────────────────────────────────────────────────────
def dummy_model(batch: List[Tuple[str, Tuple[float, ...]]]) -> Dict[str, float]:
    """
    Very simple example: average all forces and z‑torques, then scale.
    `batch` contains one entry per wrench message received since the last call.
    """
    if not batch:                       # safety guard
        return movement_cmd.copy()

    sum_fx = sum_fy = sum_fz = sum_tz = 0.0
    for _sensor, (fx, fy, fz, tx, ty, tz) in batch:
        sum_fx += fx
        sum_fy += fy
        sum_fz += fz
        sum_tz += tz                   # use z‑torque for yaw

    n = len(batch)
    return {
        "x"  : 0.005 * (sum_fx / n),
        "y"  : 0.005 * (sum_fy / n),
        "yaw": 0.0005 * (sum_tz / n)
    }


# ───────────────────────────────────────────────────────────────────────────
# Subscriber callback – add each sample to the cache
# ───────────────────────────────────────────────────────────────────────────
def wrench_callback(msg: WrenchStamped, sensor_name: str) -> None:
    fx, fy, fz = msg.wrench.force.x,  msg.wrench.force.y,  msg.wrench.force.z
    tx, ty, tz = msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z

    with cache_lock:
        wrench_cache.append((sensor_name, (fx, fy, fz, tx, ty, tz)))

    # rospy.loginfo_throttle(
    #     1.0,
    #     "[%s] cached  F=(%.2f,%.2f,%.2f) τ=(%.2f,%.2f,%.2f)   (cache len=%d)",
    #     sensor_name, fx, fy, fz, tx, ty, tz, len(wrench_cache)
    # )

    # rospy.loginfo("[%s] cached  F=(%.2f,%.2f,%.2f) τ=(%.2f,%.2f,%.2f)   (cache len=%d)",
    #     sensor_name, fx, fy, fz, tx, ty, tz, len(wrench_cache))


# ───────────────────────────────────────────────────────────────────────────
# Thread 1 : subscribe & spin
# ───────────────────────────────────────────────────────────────────────────
def force_subscriber_thread() -> None:
    sensor_topics = {
        "mini40": "/ati_ros_ati_mini40/ati_mini40/ft_meas_zeroed",
        "mini45": "/ati_ros_ati_mini45/ati_mini45/ft_meas_zeroed",
    }
    for name, topic in sensor_topics.items():
        rospy.Subscriber(topic, WrenchStamped, wrench_callback, callback_args=name)

    rospy.loginfo("Subscriber thread listening on both ATI sensors.")
    rospy.spin()        # callback handling stays here


# ───────────────────────────────────────────────────────────────────────────
# Thread 2 : publish at 5 Hz
# ───────────────────────────────────────────────────────────────────────────
def publisher_thread() -> None:
    pub  = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    rate = rospy.Rate(5)      # 5Hz

    rospy.loginfo("Publisher thread started at 5Hz.")

    while not rospy.is_shutdown():
        # ── Copy & clear cache atomically ──────────────────────────────────
        with cache_lock:
            batch = wrench_cache[:]
            wrench_cache.clear()

        # ── Run model on the whole batch (may be empty) ────────────────────
        new_cmd = dummy_model(batch)
        movement_cmd.update(new_cmd)

        # ── Publish ────────────────────────────────────────────────────────
        twist = Twist()
        twist.linear.x  = movement_cmd["x"]
        twist.linear.y  = movement_cmd["y"]
        twist.angular.z = movement_cmd["yaw"]
        pub.publish(twist)

        rospy.loginfo(
            "Published cmd: x=%.6f, y=%.6f, yaw=%.6f  (batchsize=%d)",
            movement_cmd["x"],
            movement_cmd["y"],
            movement_cmd["yaw"],
            len(batch)
        )
        rate.sleep()


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rospy.init_node("force_to_action_node", anonymous=True)

    threading.Thread(target=force_subscriber_thread, daemon=True).start()
    threading.Thread(target=publisher_thread, daemon=True).start()

    rospy.loginfo("force_to_action_node running (cached batch → model every 0.2s).")
    rospy.spin()
