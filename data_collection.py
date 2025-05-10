import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata
from rclpy.serialization import serialize_message
import message_filters
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from sensor_msgs.msg import LaserScan, Imu
from sick_scan_xd.msg import SickImu
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from vesc_msgs.msg import VescImuStamped   # ① import real type



class DataCollectionNode(Node):

    STOP_AFTER_SEC = 5 * 60 #time in sec 

    def __init__(self):
        super().__init__('sim_data_collector')

        #Time flag 
        self.start_time = self.get_clock().now()
        self.recording_active = True

        self.msg_counter = 0  # Add message counter
        
        # Initialize bag writer immediately
        self.writer = None
        self.init_bag_writer()
        
        # Configure QoS for sensor data
        sensor_qos = QoSProfile(
            depth=20,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        
        # Initialize synchronized subscribers
        self.odom_sub = message_filters.Subscriber(self, Odometry, '/odom') #/pf/pose/odom' doesn't record the steering angle ()
        self.odom_pf_sub = message_filters.Subscriber(self, Odometry, '/pf/pose/odom') 
        self.lidar_sub = message_filters.Subscriber(self, LaserScan, '/scan', qos_profile=sensor_qos)
        self.position_sub = message_filters.Subscriber(self, PoseStamped, '/pf/viz/inferred_pose')
        self.imu_raw_sub = message_filters.Subscriber(self, Imu, '/sensors/imu/raw')
        self.imu_sub = message_filters.Subscriber(self, VescImuStamped, '/sensors/imu')
        #self.imu_sub = message_filters.Subscriber(self, VescImuStamped, '/sensors/imu', qos_profile=sensor_qos)

        # Setup time synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            #[self.odom_sub, self.odom_pf_sub, self.lidar_sub, self.position_sub, self.imu_sub],
            #[self.odom_sub, self.lidar_sub, self.position_sub, self.imu_sub],
            [self.odom_sub, self.odom_pf_sub, self.lidar_sub, self.position_sub, self.imu_raw_sub, self.imu_sub],
            queue_size=20, #20
            slop=0.05,
            allow_headerless=True
        )
        self.ts.registerCallback(self.sensor_callback)

        #Check every sec if 10 min have elapsed
        self.stop_timer = self.create_timer(1.0, self._check_time)
        
        self.get_logger().info('Data collector node initialized!')

    def init_bag_writer(self):
        storage_options = StorageOptions(
            uri='car_Dataset/pure_pursuit_slow_5min',
            storage_id='sqlite3'
        )
        converter_options = ConverterOptions('', '')
        
        self.writer = SequentialWriter()
        self.writer.open(storage_options, converter_options)
        
        # Register topics
        topics = [
            TopicMetadata(name='odom', type='nav_msgs/msg/Odometry', serialization_format='cdr'),
            TopicMetadata(name='pf_odom', type='nav_msgs/msg/Odometry', serialization_format='cdr'),
            TopicMetadata(name='scan', type='sensor_msgs/msg/LaserScan', serialization_format='cdr'),
            TopicMetadata(name='pose', type='geometry_msgs/msg/PoseStamped', serialization_format='cdr'),
            TopicMetadata(name='imu_raw', type='sensor_msgs/msg/Imu', serialization_format='cdr'),
            TopicMetadata(name='imu', type='vesc_msgs/msg/VescImuStamped', serialization_format='cdr')
            #TopicMetadata(name='drive', type='ackermann_msgs/msg/AckermannDriveStamped', serialization_format='cdr')
            
        ]
        
        for topic in topics:
            self.writer.create_topic(topic)
        self.get_logger().info('Recording STARTED! Saving data to sim_Dataset/testrun')

    #def sensor_callback(self, odom_msg, odom_pf_msg, scan_msg, pose_msg, imu_msg):
    #def sensor_callback(self, odom_msg, scan_msg, pose_msg, imu_msg):
    def sensor_callback(self, odom_msg, odom_pf_msg, scan_msg, pose_msg, imu_raw_msg, imu_msg):
        if not self.recording_active:
            return #ignore once stopped

        try:
            timestamp = self.get_clock().now().nanoseconds
            
            self.writer.write('odom', serialize_message(odom_msg), timestamp)
            self.writer.write('pf_odom', serialize_message(odom_pf_msg), timestamp)
            self.writer.write('scan', serialize_message(scan_msg), timestamp)
            self.writer.write('pose', serialize_message(pose_msg), timestamp)
            self.writer.write('imu_raw', serialize_message(imu_raw_msg), timestamp)
            self.writer.write('imu', serialize_message(imu_msg), timestamp)
            
            # Update counter and print status every 10 messages
            self.msg_counter += 1
            if self.msg_counter % 10 == 0:
                self.get_logger().info(f'Writing data... ({self.msg_counter} messages recorded)')
                
        except Exception as e:
            self.get_logger().error(f'Recording error: {str(e)}')



##############################################################################
    def _check_time(self):
        elapsed_sec = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        if self.recording_active and elapsed_sec >= self.STOP_AFTER_SEC:
            self._stop_recording()

    def _stop_recording(self):
        self.recording_active = False
        self.writer.close()
        self.get_logger().info(f'Recording stopped after the time selected')
        #self.destroy_node()
        #rclpy.shutdown()

    def close(self):
        if self.recording_active:
            self._stop_recording()
###########################################################

def main(args=None):
    rclpy.init(args=args)
    collector = DataCollectionNode()
    
    try:
        collector.get_logger().info('Starting data collection...')
        rclpy.spin(collector)
    except KeyboardInterrupt:
        collector.get_logger().info('Keyboard interrupt received')
    finally:
        if collector.writer:
            collector.writer.close()
        collector.get_logger().info(f'Recording COMPLETE! Total messages: {collector.msg_counter}')
        collector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
