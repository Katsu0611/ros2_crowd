#!/usr/bin/env python3
"""
オプティカルフロー エゴモーション補償ノード（完成版）

キャリブレーション済みパラメータで動作
- compensation_factor: 1.0
- velocity_scale: 1.1
- height_scale: 1.0
- focal_scale: 1.0
- tilt_offset: 0.0
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
    """
    エゴモーション（前進・回転）によるオプティカルフローを
    計算し、補償（減算）するためのクラス
    """
    
    def __init__(self, camera_params: CameraParameters):
        self.cam = camera_params
        self.X = None  # 画像座標u (meshgrid)
        self.Y = None  # 画像座標v (meshgrid)
        
        # キャリブレーション済みパラメータ（固定値）
        self.compensation_factor = 1.0
        self.velocity_scale = 1.1      # Vz Scale = 110 相当
        self.focal_scale = 1.0
        self.height_scale = 1.0
        self.tilt_offset = 0.0
        
        self._update_grid(self.cam.width, self.cam.height)

    def update_camera_parameters(self, cam_params: CameraParameters):
        is_resized = (self.cam.width != cam_params.width or 
                      self.cam.height != cam_params.height)
        self.cam = cam_params
        if is_resized or self.X is None:
            self._update_grid(self.cam.width, self.cam.height)

    def _update_grid(self, w: int, h: int):
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
        
        # 単純な放射状モデル
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


class OpticalFlowCompensatedNode(Node):
    WINDOW_NAME = 'Optical Flow Compensated'
    
    def __init__(self):
        super().__init__('optical_flow_compensated')
        
        # パラメータ宣言
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel_joy')
        self.declare_parameter('target_width', 640)
        self.declare_parameter('visualization_mode', 'vectors')  # 'vectors' or 'color'
        self.declare_parameter('angular_filter_alpha', 0.05)
        self.declare_parameter('motion_threshold', 1.5)
        self.declare_parameter('show_original', True)  # Original表示の有無

        # パラメータ取得
        self.camera_topic = self.get_parameter('camera_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.target_width = self.get_parameter('target_width').value
        self.vis_mode = self.get_parameter('visualization_mode').value
        self.angular_filter_alpha = self.get_parameter('angular_filter_alpha').value
        self.motion_threshold = self.get_parameter('motion_threshold').value
        self.show_original = self.get_parameter('show_original').value
        
        self.camera_params = CameraParameters()
        self.compensator = EgoMotionCompensator(self.camera_params)

        # QoS
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
        
        # ウィンドウ作成
        cv2.namedWindow(self.WINDOW_NAME)
        
        self.get_logger().info('オプティカルフロー補償ノード起動（完成版）')
        self.get_logger().info(f'補償パラメータ: Comp={self.compensator.compensation_factor}, '
                             f'Vz_scale={self.compensator.velocity_scale}, '
                             f'Height={self.compensator.height_scale}')

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_params.width == msg.width and self.camera_params.fx == msg.k[0]:
            return
        
        self.get_logger().info('カメラ情報を受信しました。パラメータを更新します。')
        scale = self.target_width / msg.width
        self.camera_params = replace(
            self.camera_params,
            fx=msg.k[0] * scale, fy=msg.k[4] * scale,
            cx=msg.k[2] * scale, cy=msg.k[5] * scale,
            width=int(msg.width * scale), height=int(msg.height * scale)
        )
        self.compensator.update_camera_parameters(self.camera_params)
        self.get_logger().info(
            f'カメラパラメータ更新 (W: {self.camera_params.width}): '
            f'fx={self.camera_params.fx:.1f}, cx={self.camera_params.cx:.1f}'
        )

    def cmd_vel_callback(self, msg: Twist):
        self.linear_x = msg.linear.x
        self.angular_z = msg.angular.z

    def visualize_flow(self, flow, img_gray):
        """オプティカルフローを可視化"""
        if self.vis_mode == 'color':
            # HSVカラーマップ表示
            hsv = np.zeros((img_gray.shape[0], img_gray.shape[1], 3), dtype=np.uint8)
            hsv[..., 1] = 255
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            vis_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            # ベクトル表示
            vis_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            step = 16
            h, w = img_gray.shape
            y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
            fx, fy = flow[y, x].T
            lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
            lines = np.int32(lines + 0.5)
            cv2.polylines(vis_img, lines, 0, (0, 255, 0))
            for (x1, y1), (x2, y2) in lines:
                cv2.circle(vis_img, (x2, y2), 1, (0, 0, 255), -1)
        return vis_img

    def image_callback(self, msg: Image):
        current_time = self.get_clock().now()
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # リサイズ処理
            h, w = cv_image.shape[:2]
            if w != self.camera_params.width:
                scale = self.camera_params.width / w
                target_h = int(h * scale)
                if abs(target_h - self.camera_params.height) > 1:
                    self.camera_params.height = target_h
                    self.compensator.update_camera_parameters(self.camera_params)
                cv_image = cv2.resize(
                    cv_image, 
                    (self.camera_params.width, self.camera_params.height)
                )
            
            curr_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # 初期化
            if self.prev_gray is None or self.prev_time is None:
                self.prev_gray = curr_gray
                self.prev_time = current_time
                return

            # 時間差分計算
            try:
                dt = (current_time.nanoseconds - self.prev_time.nanoseconds) / 1e9
                if dt <= 0.001:
                    return
            except Exception as e:
                self.get_logger().error(f'dt 計算エラー: {e}')
                dt = 0.03  # フォールバック

            # オプティカルフロー計算
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # エゴモーション補償（角速度のスムージング処理含む）
            alpha = self.angular_filter_alpha
            self.compensated_angular_z = (
                self.compensated_angular_z * (1.0 - alpha) + 
                self.angular_z * alpha
            )
            
            residual_flow = self.compensator.compensate_flow(
                flow, 
                self.linear_x, 
                self.compensated_angular_z,
                dt
            )
            
            # 閾値処理（ノイズ除去）- Originalのフローに対して適用
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mask = mag < self.motion_threshold
            residual_flow[mask] = [0, 0]

            # FPS更新
            self.fps = self.update_fps()

            # 可視化
            vis_residual = self.visualize_flow(residual_flow, curr_gray)
            
            # テキスト追加
            text_residual = (
                f'Compensated Flow (Vz={self.linear_x:.2f} '
                f'Wz={self.compensated_angular_z:.2f}) FPS {self.fps:.1f}'
            )
            cv2.putText(
                vis_residual, text_residual, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )
            
            # Original表示オプション
            if self.show_original:
                vis_original = self.visualize_flow(flow, self.prev_gray)
                text_original = (
                    f'Original Flow (Vz={self.linear_x:.2f} '
                    f'Wz_cmd={self.angular_z:.2f})'
                )
                cv2.putText(
                    vis_original, text_original, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                )
                combined = cv2.vconcat([vis_original, vis_residual])
                cv2.imshow(self.WINDOW_NAME, combined)
            else:
                cv2.imshow(self.WINDOW_NAME, vis_residual)
            
            # キー入力処理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                raise KeyboardInterrupt
            
            self.prev_gray = curr_gray
            self.prev_time = current_time
            
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.get_logger().error(f'image_callback でエラー: {e}')
            import traceback
            traceback.print_exc()
    
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
    node = OpticalFlowCompensatedNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()