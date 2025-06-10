# scripts/export_bags.py
import numpy as np
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

def read_ros2_bag(bag_path):
    """
    Reads a ROS 2 bag via rosbag2_py and returns
    lidar scans, steering (angular.z), speeds (linear.x), and timestamps.
    """
    storage_opts = StorageOptions(uri=bag_path, storage_id='sqlite3')
    conv_opts    = ConverterOptions(input_serialization_format='', output_serialization_format='')
    reader = SequentialReader()
    reader.open(storage_opts, conv_opts)

    lidar_data, servo_data, speed_data, timestamps = [], [], [], []

    while reader.has_next():
        topic, serialized_msg, t_ns = reader.read_next()
        # Convert nanoseconds to seconds
        t = t_ns * 1e-9

        if topic == 'scan':
            msg = deserialize_message(serialized_msg, LaserScan)
            cleaned = np.nan_to_num(msg.ranges, posinf=0.0, neginf=0.0)
            lidar_data.append(cleaned[::2])
            timestamps.append(t)

        elif topic == 'odom':
            msg = deserialize_message(serialized_msg, Odometry)
            servo_data.append(msg.twist.twist.angular.z)
            speed_data.append(msg.twist.twist.linear.x)
            # align timestamp for control measurements too
            # (you can choose to append t here or ignore if you only need LIDAR dt)
            # timestamps.append(t)

    return (
        np.array(lidar_data),
        np.array(servo_data),
        np.array(speed_data),
        np.array(timestamps)
    )

all_lidar, all_servo, all_speed, all_ts = read_ros2_bag('/home/shirin/lab_ws/RecurrentLidarNet/car_Dataset/controller_slow_5min/controller_slow_5min_0.db3')
np.savez('slow_5min.npz',
         lidar=all_lidar,
         servo=all_servo,
         speed=all_speed,
         ts=all_ts)
# repeat for each bag
