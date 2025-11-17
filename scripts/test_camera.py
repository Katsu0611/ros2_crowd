#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class TestCamera(Node):
    def __init__(self):
        super().__init__('test_camera')
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 10)
        self.timer = self.create_timer(1.0/30.0, self.publish)
        self.bridge = CvBridge()
        self.frame_count = 0
        self.get_logger().info('テストカメラ起動 - 30Hz')
        
    def publish(self):
        # カラフルなテスト画像を生成
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # グラデーション背景
        for i in range(480):
            img[i, :, 0] = int(255 * i / 480)  # Blue
            img[i, :, 1] = int(128)            # Green
            img[i, :, 2] = int(255 * (1 - i / 480))  # Red
        
        # 動くテキスト
        x = int(100 + 200 * np.sin(self.frame_count * 0.1))
        cv2.putText(img, "Test Camera", (x, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.putText(img, f"Frame: {self.frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 円を描画
        cv2.circle(img, (320, 240), 50, (0, 255, 255), 3)
        
        # ROS2メッセージに変換
        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_link'
        self.image_pub.publish(msg)
        
        # カメラ情報
        info = CameraInfo()
        info.header = msg.header
        info.width = 640
        info.height = 480
        info.k = [525.0, 0.0, 320.0, 0.0, 525.0, 240.0, 0.0, 0.0, 1.0]
        self.info_pub.publish(info)
        
        self.frame_count += 1

def main():
    rclpy.init()
    node = TestCamera()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

