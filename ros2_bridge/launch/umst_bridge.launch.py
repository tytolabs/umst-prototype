"""
UMST ROS2 Bridge — full launch file.

Starts:
  1. gate_bridge      — REST bridge to Rust gate server
  2. telemetry_bridge — WebSocket bridge for real-time state
  3. robot_controller — demonstration controller (dry_run by default)

Prerequisites:
  cargo run --release --bin gate_server   (from the prototype directory)
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    gate_url_arg = DeclareLaunchArgument(
        "gate_url",
        default_value="http://localhost:8765",
        description="URL of the Rust UMST gate server",
    )

    ws_url_arg = DeclareLaunchArgument(
        "ws_url",
        default_value="ws://localhost:8766/telemetry",
        description="WebSocket URL for telemetry stream",
    )

    dry_run_arg = DeclareLaunchArgument(
        "dry_run",
        default_value="true",
        description="If true, robot commands are logged but not published",
    )

    gate_bridge = Node(
        package="umst_ros2_bridge",
        executable="gate_bridge",
        name="umst_gate_bridge",
        parameters=[{"gate_url": LaunchConfiguration("gate_url")}],
        output="screen",
    )

    telemetry_bridge = Node(
        package="umst_ros2_bridge",
        executable="telemetry_bridge",
        name="umst_telemetry_bridge",
        parameters=[{"ws_url": LaunchConfiguration("ws_url")}],
        output="screen",
    )

    robot_controller = Node(
        package="umst_ros2_bridge",
        executable="robot_controller",
        name="umst_robot_controller",
        parameters=[{"dry_run": LaunchConfiguration("dry_run")}],
        output="screen",
    )

    return LaunchDescription([
        gate_url_arg,
        ws_url_arg,
        dry_run_arg,
        gate_bridge,
        telemetry_bridge,
        robot_controller,
    ])
