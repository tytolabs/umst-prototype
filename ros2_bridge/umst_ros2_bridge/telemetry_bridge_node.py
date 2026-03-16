"""
UMST Telemetry Bridge Node
===========================
Subscribes to the Rust gate server's WebSocket telemetry stream (port 8766)
and republishes tensor state updates as ROS2 topics for visualization in
RViz2 or integration with other ROS2 nodes.

Architecture:
    Rust Gate Server (ws://localhost:8766/telemetry)
        → WebSocket JSON frames
        → TelemetryBridgeNode
        → /umst/telemetry/state (String, JSON)
        → /umst/telemetry/temperature (Float32)
        → /umst/telemetry/hydration (Float32)

Usage:
    ros2 run umst_ros2_bridge telemetry_bridge
"""

import json
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32

try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False


class TelemetryBridgeNode(Node):
    """Bridges WebSocket telemetry from the Rust gate server to ROS2 topics."""

    def __init__(self):
        super().__init__("umst_telemetry_bridge")

        self.declare_parameter("ws_url", "ws://localhost:8766/telemetry")
        self.declare_parameter("reconnect_interval_s", 5.0)

        self.ws_url = self.get_parameter("ws_url").get_parameter_value().string_value
        self.reconnect_s = (
            self.get_parameter("reconnect_interval_s")
            .get_parameter_value()
            .double_value
        )

        self.pub_state = self.create_publisher(String, "/umst/telemetry/state", 10)
        self.pub_temp = self.create_publisher(Float32, "/umst/telemetry/temperature", 10)
        self.pub_hydration = self.create_publisher(Float32, "/umst/telemetry/hydration", 10)

        self.frames_received = 0
        self.ws_thread = None
        self.running = True

        if not HAS_WEBSOCKET:
            self.get_logger().error(
                "websocket-client not installed — pip install websocket-client"
            )
            return

        self.start_ws_thread()
        self.get_logger().info(f"Telemetry bridge started — ws_url={self.ws_url}")

    def start_ws_thread(self):
        self.ws_thread = threading.Thread(target=self._ws_loop, daemon=True)
        self.ws_thread.start()

    def _ws_loop(self):
        while self.running:
            try:
                ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_message=self._on_ws_message,
                    on_error=self._on_ws_error,
                    on_close=self._on_ws_close,
                )
                ws.run_forever()
            except Exception as e:
                self.get_logger().warn(f"WebSocket error: {e}")

            if self.running:
                import time
                time.sleep(self.reconnect_s)

    def _on_ws_message(self, ws, message):
        self.frames_received += 1

        state_msg = String()
        state_msg.data = message
        self.pub_state.publish(state_msg)

        try:
            data = json.loads(message)
            if "temperature" in data:
                t = Float32()
                t.data = float(data["temperature"])
                self.pub_temp.publish(t)
            if "hydration_degree" in data:
                h = Float32()
                h.data = float(data["hydration_degree"])
                self.pub_hydration.publish(h)
        except (json.JSONDecodeError, KeyError):
            pass

    def _on_ws_error(self, ws, error):
        self.get_logger().warn(f"WebSocket error: {error}")

    def _on_ws_close(self, ws, close_status_code, close_msg):
        self.get_logger().info("WebSocket closed — will reconnect")

    def destroy_node(self):
        self.running = False
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TelemetryBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"Telemetry bridge: {node.frames_received} frames received")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
