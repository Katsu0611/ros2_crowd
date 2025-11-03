#!/usr/bin/env python3
"""
animove用 群流検知ノード（エゴモーション補償付き）

既存のNavigationシステムと連携して動作します。
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque
import time
from dataclasses import dataclass
import json

try:
    from ultralytics import YOLO
except ImportError:
    print("警告: 'ultralytics' ライブラリが見つかりません。")
    print("pip install ultralytics を実行してください。")


@dataclass
class CameraParameters:
    """カメラの内部パラメータ"""
    fx: float = 525.0
    fy: float = 525.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480
    camera_height: float = 0.5  # [m]
    camera_tilt: float = 0.0     # [rad]


class EgoMotionCompensator:
    """エゴモーション補償クラス"""
    
    def __init__(self, camera_params: CameraParameters):
        self.cam = camera_params
        x_coords = np.arange(self.cam.width)
        y_coords = np.arange(self.cam.height)
        self.X, self.Y = np.meshgrid(x_coords, y_coords)
        self.X_norm = (self.X - self.cam.cx) / self.cam.fx
        self.Y_norm = (self.Y - self.cam.cy) / self.cam.fy
        
    def compute_ego_flow(self, cmd_vel: Twist, dt: float = 0.033) -> np.ndarray:
        """cmd_velから画像平面上の期待フローを計算"""
        vx = cmd_vel.linear.x
        vy = cmd_vel.linear.y
        omega = cmd_vel.angular.z
        
        assumed_depth = self.cam.camera_height / np.cos(self.cam.camera_tilt)
        
        flow_translation = np.zeros((self.cam.height, self.cam.width, 2), dtype=np.float32)
        flow_translation[..., 0] = -self.cam.fx * vx * dt / assumed_depth
        
        if abs(vy) > 0.01:
            flow_translation[..., 1] = -self.cam.fy * vy * dt / assumed_depth
        
        flow_rotation = np.zeros_like(flow_translation)
        
        if abs(omega) > 0.01:
            center_x = self.cam.cx
            center_y = self.cam.cy
            dx = self.X - center_x
            dy = self.Y - center_y
            flow_rotation[..., 0] = -omega * dt * dy
            flow_rotation[..., 1] = omega * dt * dx
        
        ego_flow = flow_translation + flow_rotation
        return ego_flow
    
    def compensate_flow(self, observed_flow: np.ndarray, cmd_vel: Twist, dt: float = 0.033) -> np.ndarray:
        """観測フローからエゴモーションを差し引く"""
        ego_flow = self.compute_ego_flow(cmd_vel, dt)
        compensated_flow = observed_flow - ego_flow
        return compensated_flow


class CrowdFlowDetector:
    """群流パターン検知クラス"""
    
    def __init__(self, grid_size=40, flow_threshold=1.5):
        self.grid_size = grid_size
        self.flow_threshold = flow_threshold
        self.flow_history = deque(maxlen=10)
        
    def detect_crowd_patterns(self, frame1, frame2):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, pyr_scale=0.5, levels=3, winsize=20,
            iterations=3, poly_n=7, poly_sigma=1.5, flags=0
        )
        self.flow_history.append(flow)
        results = {
            'main_direction': self._detect_main_flow(flow),
            'congestion_areas': self._detect_congestion(flow),
            'counter_flows': self._detect_counter_flows(flow),
        }
        return flow, results
    
    def _detect_main_flow(self, flow):
        h, w = flow.shape[:2]
        regions = []
        for y in range(0, h - self.grid_size, self.grid_size):
            for x in range(0, w - self.grid_size, self.grid_size):
                region_flow = flow[y:y+self.grid_size, x:x+self.grid_size]
                avg_flow = np.mean(region_flow, axis=(0,1))
                magnitude = np.linalg.norm(avg_flow)
                if magnitude > self.flow_threshold:
                    angle = np.arctan2(avg_flow[1], avg_flow[0])
                    direction = self._classify_direction(angle)
                    regions.append({
                        'bbox': (x, y, self.grid_size, self.grid_size),
                        'magnitude': magnitude,
                        'direction': direction,
                        'angle': angle
                    })
        return regions
    
    def _detect_congestion(self, flow):
        magnitude = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
        h, w = magnitude.shape
        congestion_map = np.zeros((h, w), dtype=np.uint8)
        for y in range(0, h - self.grid_size, self.grid_size//2):
            for x in range(0, w - self.grid_size, self.grid_size//2):
                region = magnitude[y:y+self.grid_size, x:x+self.grid_size]
                avg_speed = np.mean(region)
                std_speed = np.std(region)
                if avg_speed < 0.5 and std_speed > 0.3:
                    congestion_map[y:y+self.grid_size, x:x+self.grid_size] = 255
        contours, _ = cv2.findContours(
            congestion_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        congestion_areas = []
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                congestion_areas.append(cv2.boundingRect(contour))
        return congestion_areas
    
    def _detect_counter_flows(self, flow):
        h, w = flow.shape[:2]
        counter_flows = []
        for y in range(0, h - self.grid_size, self.grid_size):
            for x in range(0, w - self.grid_size*2, self.grid_size):
                flow1 = flow[y:y+self.grid_size, x:x+self.grid_size]
                flow2 = flow[y:y+self.grid_size, x+self.grid_size:x+self.grid_size*2]
                avg_flow1 = np.mean(flow1, axis=(0,1))
                avg_flow2 = np.mean(flow2, axis=(0,1))
                dot_product = np.dot(avg_flow1, avg_flow2)
                if dot_product < -self.flow_threshold:
                    counter_flows.append({
                        'region1': (x, y, self.grid_size, self.grid_size),
                        'region2': (x+self.grid_size, y, self.grid_size, self.grid_size),
                        'strength': abs(dot_product)
                    })
        return counter_flows
    
    def _classify_direction(self, angle):
        directions = ['→', '↗', '↑', '↖', '←', '↙', '↓', '↘']
        index = int((angle + np.pi) / (2 * np.pi / 8) + 0.5) % 8
        return directions[index]
    
    def visualize_results(self, frame, results, scale_factor):
        vis = frame.copy()
        
        def scale_pt(pt):
            return int(pt * scale_factor)
        
        for region in results['main_direction']:
            x, y, w, h = region['bbox']
            x, y, w, h = scale_pt(x), scale_pt(y), scale_pt(w), scale_pt(h)
            cx, cy = x + w//2, y + h//2
            magnitude = region['magnitude'] * 10
            angle = region['angle']
            ex = int(cx + magnitude * np.cos(angle))
            ey = int(cy + magnitude * np.sin(angle))
            cv2.arrowedLine(vis, (cx, cy), (ex, ey), (0, 255, 0), 2)
            cv2.putText(vis, region['direction'], (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        for x, y, w, h in results['congestion_areas']:
            x, y, w, h = scale_pt(x), scale_pt(y), scale_pt(w), scale_pt(h)
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(vis, "CONGESTION", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        return vis


class AnimoveCrowdFlowNode(Node):
    """
    animove用 群流検知ノード
    """
    
    def __init__(self):
        super().__init__('animove_crowd_flow')
        
        # パラメータ宣言
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('target_width', 640)
        self.declare_parameter('grid_size', 30)
        self.declare_parameter('flow_threshold', 1.0)
        self.declare_parameter('person_threshold', 3)
        self.declare_parameter('yolo_model', 'yolov8n.pt')
        self.declare_parameter('enable_ego_compensation', True)
        self.declare_parameter('enable_visualization', True)
        
        # パラメータ取得
        camera_topic = self.get_parameter('camera_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.target_width = self.get_parameter('target_width').value
        grid_size = self.get_parameter('grid_size').value
        flow_threshold = self.get_parameter('flow_threshold').value
        self.person_threshold = self.get_parameter('person_threshold').value
        yolo_model_path = self.get_parameter('yolo_model').value
        self.enable_ego_compensation = self.get_parameter('enable_ego_compensation').value
        self.enable_visualization = self.get_parameter('enable_visualization').value
        
        # QoS設定
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # サブスクライバー
        self.image_sub = self.create_subscription(
            Image, camera_topic, self.image_callback, qos
        )
        self.cmd_vel_sub = self.create_subscription(
            Twist, cmd_vel_topic, self.cmd_vel_callback, qos
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, 10
        )
        
        # パブリッシャー
        self.crowd_info_pub = self.create_publisher(
            String, '/crowd_flow_info', 10
        )
        
        # 変数初期化
        self.bridge = CvBridge()
        self.prev_frame = None
        self.prev_time = None
        self.current_cmd_vel = Twist()
        self.camera_params = CameraParameters()
        self.camera_info_received = False
        
        # システム初期化
        try:
            self.get_logger().info(f'YOLOモデル {yolo_model_path} をロード中...')
            self.yolo_model = YOLO(yolo_model_path)
            self.get_logger().info('YOLOモデルのロード完了')
        except Exception as e:
            self.get_logger().error(f'YOLOモデルのロード失敗: {e}')
            self.yolo_model = None
        
        self.detector = CrowdFlowDetector(grid_size=grid_size, flow_threshold=flow_threshold)
        self.ego_compensator = EgoMotionCompensator(self.camera_params)
        
        # 統計用
        self.frame_count = 0
        self.start_time = time.time()
        
        self.get_logger().info('=== animove 群流検知ノード起動 ===')
        self.get_logger().info(f'カメラトピック: {camera_topic}')
        self.get_logger().info(f'cmd_velトピック: {cmd_vel_topic}')
        self.get_logger().info(f'エゴモーション補償: {"有効" if self.enable_ego_compensation else "無効"}')
        self.get_logger().info(f'可視化: {"有効" if self.enable_visualization else "無効"}')
    
    def camera_info_callback(self, msg: CameraInfo):
        """カメラ情報を取得"""
        if not self.camera_info_received:
            self.camera_params.fx = msg.k[0]
            self.camera_params.fy = msg.k[4]
            self.camera_params.cx = msg.k[2]
            self.camera_params.cy = msg.k[5]
            self.camera_params.width = msg.width
            self.camera_params.height = msg.height
            
            self.ego_compensator = EgoMotionCompensator(self.camera_params)
            
            self.camera_info_received = True
            self.get_logger().info(f'カメラパラメータ取得: fx={self.camera_params.fx:.1f}, fy={self.camera_params.fy:.1f}')
    
    def cmd_vel_callback(self, msg: Twist):
        """速度指令を保存"""
        self.current_cmd_vel = msg
    
    def image_callback(self, msg: Image):
        """メイン処理: 画像→YOLO→エゴモーション補償→群流検知"""
        
        try:
            current_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            current_time = self.get_clock().now()
            
            if self.prev_frame is None:
                self.prev_frame = current_frame
                self.prev_time = current_time
                return
            
            dt = (current_time - self.prev_time).nanoseconds / 1e9
            if dt < 0.001:
                dt = 0.033
            
            # --- YOLO人物検出 ---
            person_count = 0
            vis_frame = current_frame.copy()
            
            if self.yolo_model is not None:
                yolo_results = self.yolo_model(current_frame, classes=[0], verbose=False)
                person_count = len(yolo_results[0].boxes)
                if self.enable_visualization:
                    vis_frame = yolo_results[0].plot()
            
            # --- 群流検知 ---
            crowd_detected = False
            if person_count >= self.person_threshold:
                original_h, original_w = current_frame.shape[:2]
                scale_factor = original_w / self.target_width
                target_h = int(original_h / scale_factor)
                
                prev_small = cv2.resize(self.prev_frame, (self.target_width, target_h))
                curr_small = cv2.resize(current_frame, (self.target_width, target_h))
                
                flow, flow_results = self.detector.detect_crowd_patterns(prev_small, curr_small)
                
                # --- エゴモーション補償 ---
                if self.enable_ego_compensation:
                    flow = self.ego_compensator.compensate_flow(
                        flow, self.current_cmd_vel, dt
                    )
                    _, flow_results = self.detector.detect_crowd_patterns(prev_small, curr_small)
                
                # 可視化
                if self.enable_visualization:
                    vis_frame = self.detector.visualize_results(vis_frame, flow_results, scale_factor)
                
                # 群流情報をパブリッシュ
                if len(flow_results['main_direction']) > 0 or len(flow_results['congestion_areas']) > 0:
                    crowd_detected = True
                    crowd_info = {
                        'timestamp': current_time.nanoseconds,
                        'person_count': person_count,
                        'flow_regions': len(flow_results['main_direction']),
                        'congestion_areas': len(flow_results['congestion_areas']),
                        'main_flow_direction': flow_results['main_direction'][0]['direction'] if flow_results['main_direction'] else 'None',
                        'congestion_level': len(flow_results['congestion_areas']) / 10.0
                    }
                    
                    msg = String()
                    msg.data = json.dumps(crowd_info)
                    self.crowd_info_pub.publish(msg)
                    
                    self.get_logger().info(
                        f'群流検知: {person_count}人, 流れ={len(flow_results["main_direction"])}, '
                        f'混雑={len(flow_results["congestion_areas"])}',
                        throttle_duration_sec=2.0
                    )
            
            # --- 画面表示 ---
            if self.enable_visualization:
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                if elapsed > 1.0:
                    fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.start_time = time.time()
                    
                    cv2.putText(vis_frame, f'FPS: {fps:.1f}', (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.putText(vis_frame, f'Persons: {person_count}', (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                v = self.current_cmd_vel.linear.x
                w = self.current_cmd_vel.angular.z
                cv2.putText(vis_frame, f'v={v:.2f}m/s w={w:.2f}rad/s', (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                status = "EGO-COMP: ON" if self.enable_ego_compensation else "EGO-COMP: OFF"
                color = (0, 255, 255) if self.enable_ego_compensation else (128, 128, 128)
                cv2.putText(vis_frame, status, (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                if crowd_detected:
                    cv2.putText(vis_frame, "CROWD DETECTED", (10, 190),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                cv2.imshow('animove Crowd Flow', vis_frame)
                cv2.waitKey(1)
            
            self.prev_frame = current_frame
            self.prev_time = current_time
            
        except Exception as e:
            self.get_logger().error(f'処理エラー: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = AnimoveCrowdFlowNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('キーボード割り込みで終了')
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
