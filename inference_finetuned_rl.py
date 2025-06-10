#!/usr/bin/env python3
import os, time
from collections import deque
from threading import Lock

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

import tensorflow as tf
from sensor_msgs.msg import LaserScan, Joy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from vesc_msgs.msg import VescImuStamped
from ackermann_msgs.msg import AckermannDriveStamped

from stable_baselines3 import PPO


class FinetunedNode(Node):
    def __init__(self):
        super().__init__('finetuned_node')

        self.declare_parameter('is_joy', False)
        self.prev_button = 0
        self.buffer_lock = Lock()

        sup_model_path = os.path.join(os.path.dirname(__file__),
                                      'Models/sup_controller.keras')
        self.get_logger().info(f'Loading SUP model from {sup_model_path}')
        self.controller = tf.keras.models.load_model(sup_model_path)
        final = self.controller.get_layer('final_dense')
        self.orig_W, self.orig_b = final.get_weights()
        self.final_layer = final

        rl_model_path = os.path.join(os.path.dirname(__file__),
                                     'Models/ppo_rltuner.zip')
        self.get_logger().info(f'Loading RL policy from {rl_model_path}')
        self.rl = PPO.load(rl_model_path)

        _, seq_len, num_ranges, num_ch = self.controller.input_shape
        self.seq_len = int(seq_len)
        self.num_ranges = int(num_ranges)
        self.buff_scans = deque(maxlen=self.seq_len)
        self.buff_ts    = deque(maxlen=self.seq_len)

        lidar_qos = QoSProfile(depth=10,
                              reliability=QoSReliabilityPolicy.BEST_EFFORT,
                              durability=QoSDurabilityPolicy.VOLATILE)
        self.create_subscription(Joy, '/joy',
                                 self.button_callback, 10)
        self.create_subscription(LaserScan, '/scan',
                                 self.lidar_callback, lidar_qos)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)

        self.hz = 40.0
        self.period = 1.0 / self.hz
        self.create_timer(self.period, self.control_loop)

        self.get_logger().info(
            f'Node ready: seq_len={self.seq_len}, ranges={self.num_ranges}, channels={num_ch}'
        )

    def button_callback(self, msg: Joy):
        curr = msg.buttons[0]
        if curr == 1 and curr != self.prev_button:
            new_joy = not self.get_parameter('is_joy').value
            self.set_parameters([Parameter('is_joy',
                                           Parameter.Type.BOOL,
                                           new_joy)])
        self.prev_button = curr

    def lidar_callback(self, msg: LaserScan):
        cleaned = np.nan_to_num(msg.ranges,
                                nan=0.0, posinf=0.0, neginf=0.0)
        idx = np.linspace(0, len(cleaned)-1,
                          self.num_ranges, dtype=int)
        scan = cleaned[idx].astype(np.float32)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self.buffer_lock:
            self.buff_scans.append(scan)
            self.buff_ts.append(t)

    def linear_map(self, x, x_min, x_max, y_min, y_max):
        if x_max == x_min:
            return 0.5*(y_min + y_max)
        return (x - x_min)/(x_max - x_min)*(y_max - y_min) + y_min

    def control_loop(self):
        joy = self.get_parameter('is_joy').value
        if joy:
            return  # manual control; do nothing here

        with self.buffer_lock:
            if len(self.buff_scans) < self.seq_len:
                return  # not enough data yet
            scans = np.stack(self.buff_scans, axis=0)
            ts = np.array(self.buff_ts, dtype=np.float32)

        diffs = np.diff(ts, prepend=ts[0])
        dt = np.repeat(diffs[:,None], self.num_ranges, axis=1).astype(np.float32)

        nn_in = np.stack([scans, dt], axis=2)
        flat_in = nn_in.flatten().astype(np.float32)

        action, _ = self.rl.predict(flat_in, deterministic=True)
        dW = action[:self.orig_W.size].reshape(self.orig_W.shape)
        db = action[self.orig_W.size:].reshape(self.orig_b.shape)

        self.final_layer.set_weights([self.orig_W + dW,
                                      self.orig_b + db])

        pred = self.controller.predict(nn_in[None,...],
                                       verbose=0)[0]

        self.final_layer.set_weights([self.orig_W, self.orig_b])
        steer = self.linear_map(pred[0], -1.0, 1.0, -0.34, 0.34)
        speed = self.linear_map(pred[1], -1.0, 1.0, -0.5, 7.0)

        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(steer)
        msg.drive.speed = float(speed * 1.2)
        self.drive_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FinetunedNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
