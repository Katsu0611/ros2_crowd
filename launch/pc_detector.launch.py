#!/usr/bin/env python3
"""
PC側 Launchファイル
群衆フロー検知ノード（YOLO + Optical Flow）を実行する
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from pathlib import Path

# ※注意：PC側にも crowd_flow_node.py が必要です
def find_script():
    """crowd_flow_node.pyを複数の場所から探す"""
    possible_paths = [
        Path.home() / 'animove_ws' / 'install' / 'animove' / 'lib' / 'animove' / 'crowd_flow_node.py',
        Path.home() / 'animove_ws' / 'src' / 'animove' / 'scripts' / 'crowd_flow_node.py',
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"[INFO] Found script at: {path}")
            return str(path)
    
    print(f"[ERROR] Script not found!")
    return str(possible_paths[0])

def generate_launch_description():
    
    script_path = find_script()
    
    # === Launch引数 (カメラデバイス以外) ===
    
    cmd_vel_topic_arg = DeclareLaunchArgument(
        'cmd_vel_topic',
        default_value='/cmd_vel', # このトピック名でロボット側の/cmd_velに配信される
        description='速度コマンドトピック'
    )
    
    target_width_arg = DeclareLaunchArgument(
        'target_width',
        default_value='320', # PCの性能が高ければ 640 にしてもよい
        description='処理する画像の幅'
    )
    
    enable_ego_compensation_arg = DeclareLaunchArgument(
        'enable_ego_compensation',
        default_value='true',
        description='エゴモーション補償'
    )
    
    use_yolo_arg = DeclareLaunchArgument(
        'use_yolo',
        default_value='true',
        description='YOLOv8の有効化'
    )
    
    yolo_model_arg = DeclareLaunchArgument(
        'yolo_model',
        default_value='yolov8n.pt',
        description='YOLOモデル'
    )
    
    yolo_confidence_arg = DeclareLaunchArgument(
        'yolo_confidence',
        default_value='0.5',
        description='YOLO信頼度閾値'
    )
    
    person_threshold_arg = DeclareLaunchArgument(
        'person_threshold',
        default_value='3',
        description='検知開始人数'
    )
    
    grid_size_arg = DeclareLaunchArgument(
        'grid_size',
        default_value='30',
        description='グリッドサイズ'
    )
    
    flow_threshold_arg = DeclareLaunchArgument(
        'flow_threshold',
        default_value='1.5',
        description='フロー閾値'
    )
    
    enable_visualization_arg = DeclareLaunchArgument(
        'enable_visualization',
        default_value='true', # PC側でデバッグウィンドウが開く
        description='可視化'
    )
    
    flow_winsize_arg = DeclareLaunchArgument(
        'flow_winsize',
        default_value='10',
        description='フローウィンドウサイズ'
    )
    
    flow_iterations_arg = DeclareLaunchArgument(
        'flow_iterations',
        default_value='2',
        description='フロー反復回数'
    )
    
    # === 群流検知ノード ===
    crowd_flow_node = Node(
        package='animove',
        executable='crowd_flow_node.py',
        name='crowd_flow_detector',
        output='screen',
        parameters=[{
            # ※ここで受け取るトピック名を指定
            'camera_topic': '/camera/image_raw',
            'camera_info_topic': '/camera/camera_info',
            
            'cmd_vel_topic': LaunchConfiguration('cmd_vel_topic'),
            'target_width': LaunchConfiguration('target_width'),
            'use_yolo': LaunchConfiguration('use_yolo'),
            'yolo_model': LaunchConfiguration('yolo_model'),
            'yolo_confidence': LaunchConfiguration('yolo_confidence'),
            'person_threshold': LaunchConfiguration('person_threshold'),
            'grid_size': LaunchConfiguration('grid_size'),
            'flow_threshold': LaunchConfiguration('flow_threshold'),
            'enable_ego_compensation': LaunchConfiguration('enable_ego_compensation'),
            'enable_visualization': LaunchConfiguration('enable_visualization'),
            'flow_winsize': LaunchConfiguration('flow_winsize'),
            'flow_iterations': LaunchConfiguration('flow_iterations'),
        }]
    )
    
    return LaunchDescription([
        # 引数
        cmd_vel_topic_arg,
        target_width_arg,
        use_yolo_arg,
        yolo_model_arg,
        yolo_confidence_arg,
        person_threshold_arg,
        grid_size_arg,
        flow_threshold_arg,
        enable_ego_compensation_arg,
        enable_visualization_arg,
        flow_winsize_arg,
        flow_iterations_arg,
        
        # ノード
        crowd_flow_node,
    ])