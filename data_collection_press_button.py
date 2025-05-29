#!/usr/bin/env python3
"""
PS4-triggered data collection
Hold **L1**  (button index 4) -> write messages
Release L1                  -> pause, keep bag open
Ctrl-C or 5-min timeout     -> finalize and close bag
"""
import rclpy, time
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.serialization import serialize_message
import message_filters

from sensor_msgs.msg  import LaserScan, Imu, Joy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg     import Odometry
from vesc_msgs.msg    import VescImuStamped
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata


class DataCollectionNode(Node):
    RECORD_BUTTON = 4          # PS4 L1
    MAX_DURATION  = 10 * 60     # seconds

    def __init__(self):
        super().__init__('ps4_triggered_collector')

        # ---------- bag state ----------
        self.bag_open         = False
        self.writer           = None
        self.recording_active = False   # True while L1 is held
        self.msg_counter      = 0
        self.start_time       = None

        # ---------- QoS ----------
        qos_sensor = QoSProfile(
            depth=20,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        # ---------- data subscribers ----------
        self.odom_sub    = message_filters.Subscriber(self, Odometry,   '/odom')
        self.pf_sub      = message_filters.Subscriber(self, Odometry,   '/pf/pose/odom')
        self.scan_sub    = message_filters.Subscriber(self, LaserScan,  '/scan', qos_profile=qos_sensor)
        self.pose_sub    = message_filters.Subscriber(self, PoseStamped,'/pf/viz/inferred_pose')
        self.imu_raw_sub = message_filters.Subscriber(self, Imu,        '/sensors/imu/raw')
        self.imu_sub     = message_filters.Subscriber(self, VescImuStamped,'/sensors/imu')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.odom_sub, self.pf_sub, self.scan_sub,
             self.pose_sub, self.imu_raw_sub, self.imu_sub],
            queue_size=20, slop=0.05, allow_headerless=True)
        self.ts.registerCallback(self.sensor_cb)

        # ---------- joystick ----------
        self.create_subscription(Joy, '/joy', self.joy_cb, 10)

        # ---------- timeout watchdog ----------
        self.create_timer(1.0, self._timeout_check)

        self.get_logger().info('💾 Ready – hold L1 to record')

    # ------------------------------------------------------------------ Joystick
    def joy_cb(self, joy: Joy):
        pressed = joy.buttons[self.RECORD_BUTTON] == 1

        if pressed and not self.bag_open:
            self._open_bag()

        # toggle recording flag only (bag already open)
        self.recording_active = pressed

    # ------------------------------------------------------------------ Bag I/O
    def _open_bag(self):
        stamp = int(time.time())
        uri   = f'car_Dataset/fast_sequential'
        self.get_logger().info(f'🎬 Bag opened → {uri}')

        self.writer = SequentialWriter()
        self.writer.open(
            StorageOptions(uri=uri, storage_id='sqlite3'),
            ConverterOptions('', '')
        )
        for name, mtype in [
                ('odom',    'nav_msgs/msg/Odometry'),
                ('pf_odom', 'nav_msgs/msg/Odometry'),
                ('scan',    'sensor_msgs/msg/LaserScan'),
                ('pose',    'geometry_msgs/msg/PoseStamped'),
                ('imu_raw', 'sensor_msgs/msg/Imu'),
                ('imu',     'vesc_msgs/msg/VescImuStamped')]:
            self.writer.create_topic(
                TopicMetadata(name=name, type=mtype, serialization_format='cdr'))

        self.bag_open    = True
        self.start_time  = self.get_clock().now()
        self.msg_counter = 0
        self.recording_active = True   # because button is currently pressed

    def _close_bag(self):
        if self.writer:
            self.writer.close()
            self.get_logger().info(f'⏹️  Bag closed (total {self.msg_counter} msgs)')
        self.bag_open = self.recording_active = False
        self.writer = None

    # -------------------------------------------------------------- Data writer
    def sensor_cb(self, odom, pf_odom, scan, pose, imu_raw, imu):
        if not self.recording_active:
            return

        t = self.get_clock().now().nanoseconds
        try:
            self.writer.write('odom',    serialize_message(odom),    t)
            self.writer.write('pf_odom', serialize_message(pf_odom), t)
            self.writer.write('scan',    serialize_message(scan),    t)
            self.writer.write('pose',    serialize_message(pose),    t)
            self.writer.write('imu_raw', serialize_message(imu_raw), t)
            self.writer.write('imu',     serialize_message(imu),     t)

            self.msg_counter += 1
            if self.msg_counter % 10 == 0:
                self.get_logger().info(f'… {self.msg_counter} msgs')
        except Exception as e:
            self.get_logger().error(f'Write error: {e}')

    # ------------------------------------------------------------- Watchdog
    def _timeout_check(self):
        if self.bag_open and self.start_time:
            elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
            if elapsed >= self.MAX_DURATION:
                self.get_logger().warn('⏰ Max duration reached – closing bag')
                self._close_bag()

    # ------------------------------------------------------------- Shutdown
    def destroy_node(self):
        self._close_bag()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DataCollectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Ctrl-C – shutting down')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
