#!/usr/bin/env python3
"""
ROS2 群衆フロー検知ノード（完全版・修正版v5）

修正点:
1. 矢印(ベクトル)の描画を完全に削除しました（左側にも右側にも表示しません）。
2. HSVフローの明度ブースト(Boosted HSV)を削除し、元の自然な明度計算に戻しました。
   (支配的な方向への色合わせ機能は維持しています)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from dataclasses import dataclass, replace
from collections import deque

# YOLOv8のインポート（オプション）
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("警告: ultralytics がインストールされていません。")
    print("インストール: pip install ultralytics --break-system-packages")


@dataclass
class CameraParameters:
    """カメラの内部パラメータ"""
    fx: float = 525.0
    fy: float = 525.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480
    camera_height: float = 0.185
    camera_tilt: float = 0.1745


class EgoMotionCompensator:
    """エゴモーション補償クラス"""
    
    def __init__(self, camera_params: CameraParameters):
        self.cam = camera_params
        self.X = None
        self.Y = None
        
        # キャリブレーション済みパラメータ
        self.compensation_factor = 1.0
        self.velocity_scale = 1.1      # cmd_velとロボット実速度の比率
        self.focal_scale = 1.0
        self.height_scale = 1.0
        self.tilt_offset = 0.0
        
        self._update_grid(self.cam.width, self.cam.height)

    def update_camera_parameters(self, cam_params: CameraParameters):
        """カメラパラメータを更新"""
        is_resized = (self.cam.width != cam_params.width or 
                      self.cam.height != cam_params.height)
        self.cam = cam_params
        if is_resized or self.X is None:
            self._update_grid(self.cam.width, self.cam.height)

    def _update_grid(self, w: int, h: int):
        """画像座標のメッシュグリッドを作成"""
        self.X, self.Y = np.meshgrid(np.arange(w), np.arange(h))
        self.X = self.X.astype(np.float32)
        self.Y = self.Y.astype(np.float32)

    def _calculate_expected_radial_flow(self, linear_x: float, dt: float) -> np.ndarray:
        """前進による放射状フローを計算"""
        Vz = linear_x * self.velocity_scale
        if abs(Vz) < 0.001 or dt <= 0:
            return np.zeros((self.cam.height, self.cam.width, 2), dtype=np.float32)

        fx = self.cam.fx * self.focal_scale
        fy = self.cam.fy * self.focal_scale
        
        Z_avg = 1.0 * self.height_scale
        if Z_avg < 0.1:
            Z_avg = 0.1
        
        delta_Z = Vz * dt
        scale_ratio = delta_Z / (Z_avg - self.tilt_offset)

        X_norm = (self.X - self.cam.cx) / fx
        flow_x = fx * X_norm * scale_ratio
        Y_norm = (self.Y - self.cam.cy) / fy
        flow_y = fy * Y_norm * scale_ratio
        
        return np.stack([flow_x, flow_y], axis=-1).astype(np.float32)

    def _calculate_expected_rotational_flow(self, angular_z: float, dt: float) -> np.ndarray:
        """回転によるフローを計算"""
        if abs(angular_z) < 0.001 or dt <= 0:
            return np.zeros((self.cam.height, self.cam.width, 2), dtype=np.float32)

        theta = self.cam.camera_tilt + self.tilt_offset 
        Wz = angular_z
        Wy_cam = -Wz * np.sin(theta)
        Wz_cam = Wz * np.cos(theta)
        fx = self.cam.fx * self.focal_scale
        fy = self.cam.fy * self.focal_scale
        u_cx = self.X - self.cam.cx
        v_cy = self.Y - self.cam.cy
        u_cx_fx = u_cx / fx
        v_cy_fy = v_cy / fy
        
        flow_x_per_sec = fx * (Wz_cam * v_cy_fy - Wy_cam - Wy_cam * u_cx_fx**2)
        flow_y_per_sec = fy * (-Wz_cam * u_cx_fx - Wy_cam * u_cx_fx * v_cy_fy)
        
        flow_x = flow_x_per_sec * dt
        flow_y = flow_y_per_sec * dt

        return np.stack([flow_x, flow_y], axis=-1).astype(np.float32)

    def compensate_flow(self, flow: np.ndarray, linear_x: float, angular_z: float, dt: float) -> np.ndarray:
        """エゴモーション補償を実行"""
        expected_radial_flow = self._calculate_expected_radial_flow(linear_x, dt)
        expected_rotational_flow = self._calculate_expected_rotational_flow(angular_z, dt)
        
        expected_flow = expected_radial_flow - expected_rotational_flow
        residual_flow = flow - (expected_flow * self.compensation_factor)
        return residual_flow


class CrowdFlowDetector:
    """群衆フロー検知クラス"""
    
    def __init__(self, grid_size=40, flow_threshold=1.5):
        self.grid_size = grid_size
        self.flow_threshold = flow_threshold
        self.flow_history = deque(maxlen=10)
        
    def detect_crowd_patterns(self, flow, person_mask=None):
        """群衆パターンを検知"""
        # マスク適用（人物領域のみに限定）
        if person_mask is not None:
            masked_flow = flow.copy()
            masked_flow[~person_mask] = [0, 0]
        else:
            masked_flow = flow
        
        self.flow_history.append(masked_flow)
        
        results = {
            'main_direction': self._detect_main_flow(masked_flow, person_mask),
            'congestion_areas': self._detect_congestion(masked_flow),
            'counter_flows': self._detect_counter_flows(masked_flow),
        }
        
        return results
    
    def _detect_main_flow(self, flow, person_mask=None):
        """主要な流れを検知"""
        h, w = flow.shape[:2]
        regions = []
        
        for y in range(0, h - self.grid_size, self.grid_size):
            for x in range(0, w - self.grid_size, self.grid_size):
                region_flow = flow[y:y+self.grid_size, x:x+self.grid_size]
                
                # 人物マスクがある場合、グリッド内に人がいるか確認
                if person_mask is not None:
                    region_mask = person_mask[y:y+self.grid_size, x:x+self.grid_size]
                    if np.sum(region_mask) < (self.grid_size * self.grid_size * 0.2):
                        continue  # 人が少ないグリッドはスキップ
                
                avg_flow = np.mean(region_flow, axis=(0, 1))
                magnitude = np.linalg.norm(avg_flow)
                angle = np.arctan2(avg_flow[1], avg_flow[0])
                
                # 方向を分類（magnitudeも渡す）
                direction = self._classify_direction(angle, magnitude)
                
                # 有意な動きがある領域のみ記録（停止は除外）
                if direction != 'STATIC/NO_FLOW':
                    regions.append({
                        'bbox': (x, y, self.grid_size, self.grid_size),
                        'magnitude': magnitude,
                        'direction': direction,
                        'angle': angle
                    })
        
        return regions
    
    def _detect_congestion(self, flow):
        """混雑エリアを検知（動きが少なく、ばらつきが大きい領域）"""
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        h, w = magnitude.shape
        congestion_map = np.zeros((h, w), dtype=np.uint8)
        
        for y in range(0, h - self.grid_size, self.grid_size//2):
            for x in range(0, w - self.grid_size, self.grid_size//2):
                region = magnitude[y:y+self.grid_size, x:x+self.grid_size]
                avg_speed = np.mean(region)
                std_speed = np.std(region)
                
                # 平均速度が低く、標準偏差が高い = 混雑
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
        """対向流を検知（隣接グリッドで逆方向の流れ）"""
        h, w = flow.shape[:2]
        counter_flows = []
        
        for y in range(0, h - self.grid_size, self.grid_size):
            for x in range(0, w - self.grid_size*2, self.grid_size):
                flow1 = flow[y:y+self.grid_size, x:x+self.grid_size]
                flow2 = flow[y:y+self.grid_size, x+self.grid_size:x+self.grid_size*2]
                
                avg_flow1 = np.mean(flow1, axis=(0, 1))
                avg_flow2 = np.mean(flow2, axis=(0, 1))
                
                dot_product = np.dot(avg_flow1, avg_flow2)
                
                # 内積が負 = 逆方向
                if dot_product < -self.flow_threshold:
                    counter_flows.append({
                        'region1': (x, y, self.grid_size, self.grid_size),
                        'region2': (x+self.grid_size, y, self.grid_size, self.grid_size),
                        'strength': abs(dot_product)
                    })
        
        return counter_flows
    
    def _classify_direction(self, angle, magnitude):
        """
        角度と速度をロボット視点の5方向に分類
        """
        # 停止判定（flow_thresholdの半分以下は停止とみなす）
        STOP_THRESHOLD = self.flow_threshold * 0.5
        if magnitude < STOP_THRESHOLD:
            return 'STATIC/NO_FLOW'
        
        # ラジアンから度に変換
        angle_deg = np.degrees(angle)
        
        # 4方向に分類（ロボット視点）
        # 画像座標系: 下=ロボットに接近, 上=ロボットから離脱
        if -135 <= angle_deg < -45:
            return 'APPROACH'  # 接近
        elif -45 <= angle_deg < 45:
            return 'RIGHT'  # 右横切
        elif 45 <= angle_deg < 135:
            return 'RECEDE'  # 離脱
        else:
            return 'LEFT'  # 左横切


class CrowdFlowNode(Node):
    """ROS2 群衆フロー検知ノード（完全版）"""
    
    WINDOW_NAME = 'Crowd Flow Detection'
    
    def __init__(self):
        super().__init__('crowd_flow_detector')
        
        # === パラメータ宣言 ===
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        
        # 処理パラメータ
        self.declare_parameter('target_width', 320)
        self.declare_parameter('angular_filter_alpha', 0.05)
        self.declare_parameter('motion_threshold', 1.0)
        
        # 群衆検知パラメータ
        self.declare_parameter('grid_size', 30)
        self.declare_parameter('flow_threshold', 1.5)
        
        # YOLO関連
        self.declare_parameter('use_yolo', True)
        self.declare_parameter('yolo_model', 'yolov8n.pt')
        self.declare_parameter('yolo_confidence', 0.5)
        
        # 機能フラグ
        self.declare_parameter('enable_ego_compensation', True)
        self.declare_parameter('enable_visualization', True)
        self.declare_parameter('person_threshold', 1)
        
        # オプティカルフローパラメータ
        self.declare_parameter('flow_pyr_scale', 0.5)
        self.declare_parameter('flow_levels', 3)
        self.declare_parameter('flow_winsize', 10)
        self.declare_parameter('flow_iterations', 2)

        # === パラメータ取得 ===
        self.camera_topic = self.get_parameter('camera_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        
        self.target_width = self.get_parameter('target_width').value
        self.angular_filter_alpha = self.get_parameter('angular_filter_alpha').value
        self.motion_threshold = self.get_parameter('motion_threshold').value
        
        self.grid_size = self.get_parameter('grid_size').value
        self.flow_threshold = self.get_parameter('flow_threshold').value
        
        self.use_yolo = self.get_parameter('use_yolo').value
        self.enable_ego_compensation = self.get_parameter('enable_ego_compensation').value
        self.enable_visualization = self.get_parameter('enable_visualization').value
        self.person_threshold = self.get_parameter('person_threshold').value
        
        self.flow_pyr_scale = self.get_parameter('flow_pyr_scale').value
        self.flow_levels = self.get_parameter('flow_levels').value
        self.flow_winsize = self.get_parameter('flow_winsize').value
        self.flow_iterations = self.get_parameter('flow_iterations').value
        
        # === 初期化 ===
        self.camera_params = CameraParameters()
        self.compensator = EgoMotionCompensator(self.camera_params)
        
        self.crowd_detector = CrowdFlowDetector(
            grid_size=self.grid_size,
            flow_threshold=self.flow_threshold
        )
        
        # YOLOv8初期化
        self.yolo_model = None
        if self.use_yolo and YOLO_AVAILABLE:
            try:
                yolo_model_path = self.get_parameter('yolo_model').value
                self.yolo_model = YOLO(yolo_model_path)
                self.yolo_confidence = self.get_parameter('yolo_confidence').value
                self.get_logger().info('✓ YOLOv8ロード成功')
            except Exception as e:
                self.get_logger().error(f'❌ YOLOv8のロード失敗: {e}')
                self.yolo_model = None
        
        # QoS設定
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.bridge = CvBridge()

        # Subscriptions
        self.info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic,
            self.camera_info_callback, qos_reliable
        )
        self.cmd_vel_sub = self.create_subscription(
            Twist, self.cmd_vel_topic,
            self.cmd_vel_callback, qos_best_effort
        )
        self.image_sub = self.create_subscription(
            Image, self.camera_topic,
            self.image_callback, qos_best_effort
        )

        # 状態変数
        self.prev_gray = None
        self.prev_time = None
        self.linear_x = 0.0
        self.angular_z = 0.0
        self.compensated_angular_z = 0.0
        
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        
        cv2.namedWindow(self.WINDOW_NAME)
        
        # 起動ログ
        self.get_logger().info('=' * 50)
        self.get_logger().info('群衆フロー検知ノード起動')
        self.get_logger().info('=' * 50)

    def camera_info_callback(self, msg: CameraInfo):
        """カメラ情報コールバック"""
        if self.camera_params.width == msg.width and self.camera_params.fx == msg.k[0]:
            return
        
        scale = self.target_width / msg.width
        self.camera_params = replace(
            self.camera_params,
            fx=msg.k[0] * scale, fy=msg.k[4] * scale,
            cx=msg.k[2] * scale, cy=msg.k[5] * scale,
            width=int(msg.width * scale), height=int(msg.height * scale)
        )
        self.compensator.update_camera_parameters(self.camera_params)
        self.get_logger().info(
            f'カメラパラメータ更新: {self.camera_params.width}x{self.camera_params.height}'
        )

    def cmd_vel_callback(self, msg: Twist):
        """速度コマンドコールバック"""
        self.linear_x = msg.linear.x
        self.angular_z = msg.angular.z

    def image_callback(self, msg: Image):
        """画像コールバック（メイン処理）"""
        current_time = self.get_clock().now()
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # リサイズ
            h, w = cv_image.shape[:2]
            size_changed = False
            if w != self.camera_params.width:
                scale = self.camera_params.width / w
                target_h = int(h * scale)
                if abs(target_h - self.camera_params.height) > 1:
                    self.camera_params.height = target_h
                    self.compensator.update_camera_parameters(self.camera_params)
                    size_changed = True
                cv_image = cv2.resize(
                    cv_image,
                    (self.camera_params.width, self.camera_params.height)
                )
            
            curr_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            if self.prev_gray is None or self.prev_time is None or size_changed or self.prev_gray.shape != curr_gray.shape:
                self.prev_gray = curr_gray
                self.prev_time = current_time
                return

            try:
                dt = (current_time.nanoseconds - self.prev_time.nanoseconds) / 1e9
                if dt <= 0.001: return
            except Exception:
                dt = 0.03

            # === オプティカルフロー計算 ===
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, curr_gray, None,
                pyr_scale=self.flow_pyr_scale,
                levels=self.flow_levels,
                winsize=self.flow_winsize,
                iterations=self.flow_iterations,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
            
            # === エゴモーション補償 ===
            if self.enable_ego_compensation:
                alpha = self.angular_filter_alpha
                self.compensated_angular_z = (
                    self.compensated_angular_z * (1.0 - alpha) +
                    self.angular_z * alpha
                )
                residual_flow = self.compensator.compensate_flow(
                    flow, self.linear_x, self.compensated_angular_z, dt
                )
            else:
                residual_flow = flow
                self.compensated_angular_z = 0.0
            
            # === 閾値処理（ノイズ除去）===
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mask = mag < self.motion_threshold
            residual_flow[mask] = [0, 0]
            
            # === YOLO人物検出 ===
            person_mask = None
            yolo_boxes = []
            person_count = 0
            
            if self.use_yolo and self.yolo_model is not None:
                try:
                    results = self.yolo_model(cv_image, verbose=False, conf=self.yolo_confidence)
                    person_mask = np.zeros((cv_image.shape[0], cv_image.shape[1]), dtype=bool)
                    
                    for box in results[0].boxes:
                        if int(box.cls) == 0:  # 'person'クラス
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            person_mask[y1:y2, x1:x2] = True
                            yolo_boxes.append((x1, y1, x2, y2, float(box.conf)))
                            person_count += 1
                except Exception as e:
                    if self.frame_count % 150 == 0:
                        self.get_logger().error(f'YOLO検出エラー: {e}')
            
            # === 群衆パターン検知 ===
            if person_count >= self.person_threshold or not self.use_yolo:
                results = self.crowd_detector.detect_crowd_patterns(
                    residual_flow, person_mask
                )
            else:
                results = {
                    'main_direction': [],
                    'congestion_areas': [],
                    'counter_flows': []
                }

            # === 支配的なフロー方向の計算 ===
            dominant_angle_hsv = None
            dominant_direction_label = 'STATIC/NO_FLOW'
            
            if person_mask is not None and person_count > 0:
                person_flow_x = residual_flow[person_mask, 0]
                person_flow_y = residual_flow[person_mask, 1]
                
                if person_flow_x.size > 0:
                    avg_flow_x = np.mean(person_flow_x)
                    avg_flow_y = np.mean(person_flow_y)
                    
                    magnitude = np.linalg.norm([avg_flow_x, avg_flow_y])
                    if magnitude > self.flow_threshold * 0.5:
                        dominant_angle = np.arctan2(avg_flow_y, avg_flow_x)
                        
                        # 支配的な方向ラベルの決定
                        dominant_direction_label = self.crowd_detector._classify_direction(
                            dominant_angle, magnitude
                        )
                        
                        dominant_angle_hsv = (dominant_angle * 180 / np.pi / 2) % 180
            
            # === ターミナルログ出力（追加機能）===
            # 約1秒に1回ログ出力
            if self.frame_count % 30 == 0: 
                log_msg = f'👥 人数: {person_count} | ➡️ 支配的な流れ: {dominant_direction_label}'
                self.get_logger().info(log_msg)
            # ==================================

            # === 可視化 ===
            if self.enable_visualization:
                # 左側：生の画像（YOLOボックス付き）
                left_frame = cv_image.copy()
                for x1, y1, x2, y2, conf in yolo_boxes:
                    cv2.rectangle(left_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 矢印表示の削除（visualize_resultsの呼び出しを削除）
                
                # 右側：HSVフロー可視化（支配的な方向に色を合わせる、ブーストなし）
                right_frame = self._create_hsv_flow_visualization(
                    residual_flow, 
                    person_mask,
                    cv_image.shape[:2],
                    dominant_angle_hsv
                )
                
                # FPS表示
                self.fps = self.update_fps()
                fps_text = f'FPS: {self.fps:.1f} | Persons: {person_count}'
                cv2.putText(left_frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(right_frame, 'Direction Flow (HSV)', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                combined_frame = np.hstack([left_frame, right_frame])
                cv2.imshow(self.WINDOW_NAME, combined_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                raise KeyboardInterrupt
            
            self.prev_gray = curr_gray
            self.prev_time = current_time
            
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.get_logger().error(f'image_callback エラー: {e}')
            import traceback
            traceback.print_exc()
    
    def _create_hsv_flow_visualization(self, flow, person_mask, image_shape, dominant_angle_hsv=None):
        """HSV形式で光学フローを可視化（人物領域のみ、支配的な方向に色を合わせる）"""
        h, w = image_shape
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 1] = 255
        
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        base_hue = ang * 180 / np.pi / 2
        
        # 支配的な方向に色を合わせる（色相シフト）
        if dominant_angle_hsv is not None:
            # 支配的な角度を中央（90度）に合わせる
            hue_offset = dominant_angle_hsv - 90 
            hsv[..., 0] = (base_hue - hue_offset) % 180
        else:
            hsv[..., 0] = base_hue
            
        # Value（明度）：元の計算に戻す（ブースト削除）
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 人物マスクがある場合、人物領域のみ表示
        if person_mask is not None:
            masked_bgr = np.zeros_like(bgr)
            masked_bgr[person_mask] = bgr[person_mask]
            return masked_bgr
        else:
            return bgr
    
    def update_fps(self):
        """FPS計算"""
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()
            return fps
        return self.fps


def main(args=None):
    rclpy.init(args=args)
    node = CrowdFlowNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Ctrl+Cで終了します...')
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
