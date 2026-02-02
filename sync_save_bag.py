#!/usr/bin/env python3
import os
import glob
import bisect
import argparse

import rosbag
from cv_bridge import CvBridge
import cv2

def find_nearest_index(sorted_list, value):
    """
    Given a sorted list and a value, return the index of the element
    nearest to value.
    """
    idx = bisect.bisect_left(sorted_list, value)
    if idx == 0:
        return 0
    if idx >= len(sorted_list):
        return len(sorted_list) - 1
    before = sorted_list[idx - 1]
    after = sorted_list[idx]
    # pick the closer of the two
    if abs(after - value) < abs(value - before):
        return idx
    else:
        return idx - 1

def extract_and_sync(bag_path):
    """
    Extract force readings from two ATI sensors and RGB images from a ROS bag,
    align them in time, and write out synchronized data to disk.
    """
    base = os.path.splitext(os.path.basename(bag_path))[0]
    out_root = os.path.join(os.getcwd(), base)
    forces_dir = os.path.join(out_root, 'forces')
    images_dir = os.path.join(out_root, 'images')
    os.makedirs(forces_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # containers for raw data
    force_data = {'ati_mini40': [], 'ati_mini45': []}
    image_data = []  # list of (timestamp, sensor_msgs/Image)

    print(f"[INFO] Reading bag: {bag_path}")
    bag = rosbag.Bag(bag_path, 'r')
    for topic, msg, _ in bag.read_messages():
        # Force measurements
        if topic.endswith('ft_meas_zeroed'):
            sensor = 'ati_mini40' if 'mini40' in topic else 'ati_mini45'
            ts = msg.header.stamp.to_sec()
            w = msg.wrench
            force_data[sensor].append((
                ts,
                w.force.x, w.force.y, w.force.z,
                w.torque.x, w.torque.y, w.torque.z
            ))

        # RGB image
        elif topic == '/camera/rgb/image_raw':
            ts = msg.header.stamp.to_sec()
            image_data.append((ts, msg))
    bag.close()
    print(f"[INFO] Finished reading bag. "
          f"Found {len(force_data['ati_mini40'])} mini40 samples, "
          f"{len(force_data['ati_mini45'])} mini45 samples, "
          f"{len(image_data)} images.")

    # Sort each by timestamp
    for key in force_data:
        force_data[key].sort(key=lambda x: x[0])
    image_data.sort(key=lambda x: x[0])

    # Split timestamps and values
    ts40   = [t for t, *_ in force_data['ati_mini40']]
    vals40 = [v for _, *v in force_data['ati_mini40']]
    ts45   = [t for t, *_ in force_data['ati_mini45']]
    vals45 = [v for _, *v in force_data['ati_mini45']]

    # Align mini45 readings to mini40 timestamps
    aligned45 = [ vals45[find_nearest_index(ts45, t)] for t in ts40 ]
    print(f"[INFO] Synchronized mini45: {len(vals45)} → {len(aligned45)} entries")

    # Write out force data
    header = "timestamp,fx,fy,fz,tx,ty,tz\n"
    for sensor, timestamps, values in [
        ('ati_mini40', ts40, vals40),
        ('ati_mini45', ts40, aligned45)
    ]:
        sensor_dir = os.path.join(forces_dir, sensor)
        os.makedirs(sensor_dir, exist_ok=True)
        out_file = os.path.join(sensor_dir, f"{sensor}.txt")
        with open(out_file, 'w') as f:
            f.write(header)
            for t, (fx, fy, fz, tx, ty, tz) in zip(timestamps, values):
                f.write(f"{t:.6f},{fx:.6f},{fy:.6f},{fz:.6f},{tx:.6f},{ty:.6f},{tz:.6f}\n")
        print(f"[INFO] Wrote {len(values)} lines to {out_file}")

    # Save images, naming by closest force timestamp
    bridge = CvBridge()
    for img_ts, img_msg in image_data:
        idx = find_nearest_index(ts40, img_ts)
        matched_t = ts40[idx]
        cv_img = bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        fn = os.path.join(images_dir, f"{matched_t:.6f}.jpg")
        cv2.imwrite(fn, cv_img)
    print(f"[INFO] Saved {len(image_data)} images to {images_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract and sync force & image data from one or more ROS bag files."
    )
    parser.add_argument(
        'input_path',
        help="Path to a single .bag file or a directory containing .bag files"
    )
    args = parser.parse_args()

    # Determine list of .bag files to process
    if os.path.isdir(args.input_path):
        bag_paths = glob.glob(os.path.join(args.input_path, '*.bag'))
        if not bag_paths:
            parser.error(f"No .bag files found in directory: {args.input_path}")
    else:
        bag_paths = [args.input_path]

    # Process each bag
    for bag_path in bag_paths:
        if not bag_path.endswith('.bag'):
            print(f"[WARN] Skipping non-.bag file: {bag_path}")
            continue
        try:
            extract_and_sync(bag_path)
        except Exception as e:
            print(f"[ERROR] Failed to process {bag_path}: {e}")

if __name__ == '__main__':
    main()

