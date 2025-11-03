#!/usr/bin/env python3
"""
animove用 群流検知launchファイル
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    
    # Launch引数
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera/image_raw',
        description='カメラ画像トピック（RealSense: /camera/color/image_raw）'
    )
    
    camera_info_topic_arg = DeclareLaunchArgument(
        'camera_info_topic',
        default_value='/camera/camera_info',
        description='カメラ情報トピック'
    )
    
    cmd_vel_topic_arg = DeclareLaunchArgument(
        'cmd_vel_topic',
        default_value='/cmd_vel',
        description='速度指令トピック'
    )
    
    enable_ego_arg = DeclareLaunchArgument(
        'enable_ego_compensation',
        default_value='true',
        description='エゴモーション補償を有効化'
    )
    
    enable_viz_arg = DeclareLaunchArgument(
        'enable_visualization',
        default_value='true',
        description='可視化を有効化'
    )
    
    yolo_model_arg = DeclareLaunchArgument(
        'yolo_model',
        default_value='yolov8n.pt',
        description='YOLOモデル (yolov8n/s/m.pt)'
    )
    
    person_threshold_arg = DeclareLaunchArgument(
        'person_threshold',
        default_value='3',
        description='群流検知を開始する人数の閾値'
    )
    
    target_width_arg = DeclareLaunchArgument(
        'target_width',
        default_value='640',
        description='処理する画像の幅（小さいほど高速）'
    )
    
    grid_size_arg = DeclareLaunchArgument(
        'grid_size',
        default_value='30',
        description='グリッドサイズ（大きいほど粗い検知）'
    )
    
    flow_threshold_arg = DeclareLaunchArgument(
        'flow_threshold',
        default_value='1.0',
        description='フロー検出の閾値'
    )
    
    # 群流検知ノード
    crowd_flow_node = Node(
        package='animove',
        executable='crowd_flow_node.py',
        name='crowd_flow_detector',
        output='screen',
        parameters=[{
            'camera_topic': LaunchConfiguration('camera_topic'),
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
            'cmd_vel_topic': LaunchConfiguration('cmd_vel_topic'),
            'enable_ego_compensation': LaunchConfiguration('enable_ego_compensation'),
            'enable_visualization': LaunchConfiguration('enable_visualization'),
            'yolo_model': LaunchConfiguration('yolo_model'),
            'person_threshold': LaunchConfiguration('person_threshold'),
            'target_width': LaunchConfiguration('target_width'),
            'grid_size': LaunchConfiguration('grid_size'),
            'flow_threshold': LaunchConfiguration('flow_threshold'),
        }]
    )
    
    return LaunchDescription([
        camera_topic_arg,
        camera_info_topic_arg,
        cmd_vel_topic_arg,
        enable_ego_arg,
        enable_viz_arg,
        yolo_model_arg,
        person_threshold_arg,
        target_width_arg,
        grid_size_arg,
        flow_threshold_arg,
        crowd_flow_node,
    ])
