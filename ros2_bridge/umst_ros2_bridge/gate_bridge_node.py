"""
UMST Gate Bridge Node
=====================
ROS2 node bridging the Rust thermodynamic gate server (REST API on port 8765)
to the ROS2 ecosystem. Every robotic action proposal is validated against the
Clausius-Duhem inequality before reaching hardware.

Architecture:
    Robot Sensor → /umst/proposal (topic) → GateBridgeNode → POST /gate → Rust Kernel
                                                 ↓
                                         /umst/decision (topic) → Robot Controller
                                                 ↓
                                         /gate_check (service)  → On-demand queries

Requires:
    cargo run --release --bin gate_server   (Rust side, port 8765)

Usage:
    ros2 run umst_ros2_bridge gate_bridge --ros-args -p gate_url:=http://localhost:8765
"""

import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32, Header
import requests
from dataclasses import dataclass


@dataclass
class GateResult:
    admissible: bool
    verdict: str
    violation: str
    strength_bound: float
    physics_strength: float
    hydration_degree: float
    w_c_ratio: float


class GateBridgeNode(Node):
    """Bridges UMST thermodynamic gate (Rust REST) to ROS2 topics and services."""

    def __init__(self):
        super().__init__("umst_gate_bridge")

        self.declare_parameter("gate_url", "http://localhost:8765")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("timeout_s", 1.0)

        self.gate_url = self.get_parameter("gate_url").get_parameter_value().string_value
        rate_hz = self.get_parameter("rate_hz").get_parameter_value().double_value
        self.timeout = self.get_parameter("timeout_s").get_parameter_value().double_value

        self.sub_proposal = self.create_subscription(
            String, "/umst/proposal", self.on_proposal, 10
        )

        self.pub_decision = self.create_publisher(String, "/umst/decision", 10)
        self.pub_admissible = self.create_publisher(Bool, "/umst/admissible", 10)
        self.pub_strength = self.create_publisher(Float32, "/umst/physics_strength", 10)

        self.health_timer = self.create_timer(5.0, self.check_health)

        self.gate_alive = False
        self.total_checks = 0
        self.total_accepted = 0
        self.total_rejected = 0

        self.get_logger().info(
            f"UMST Gate Bridge started — gate_url={self.gate_url}"
        )

    def check_health(self):
        """Periodic health check against the Rust gate server."""
        try:
            resp = requests.get(
                f"{self.gate_url}/health", timeout=self.timeout
            )
            was_alive = self.gate_alive
            self.gate_alive = resp.status_code == 200
            if self.gate_alive and not was_alive:
                self.get_logger().info("Gate server connection established")
            elif not self.gate_alive and was_alive:
                self.get_logger().warn("Gate server health check failed")
        except requests.exceptions.ConnectionError:
            if self.gate_alive:
                self.get_logger().warn(
                    f"Gate server unreachable at {self.gate_url}"
                )
            self.gate_alive = False

    def on_proposal(self, msg: String):
        """
        Handle a state transition proposal.

        Expected JSON payload matching GateRequest:
        {
            "cement": 350.0, "slag": 0.0, "fly_ash": 0.0,
            "water": 175.0, "age": 28.0, "predicted_strength": 35.0,
            "coarse_agg": 1000.0, "fine_agg": 750.0,
            "temperature_c": 20.0, "dataset": "D1"
        }
        """
        if not self.gate_alive:
            self.get_logger().warn(
                "Gate server not available — proposal queued until reconnection"
            )
            return

        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Invalid proposal JSON: {e}")
            return

        result = self.call_gate(payload)
        if result is None:
            return

        self.total_checks += 1
        if result.admissible:
            self.total_accepted += 1
        else:
            self.total_rejected += 1

        decision_msg = String()
        decision_msg.data = json.dumps({
            "admissible": result.admissible,
            "verdict": result.verdict,
            "violation": result.violation,
            "strength_bound": result.strength_bound,
            "physics_strength": result.physics_strength,
            "hydration_degree": result.hydration_degree,
            "w_c_ratio": result.w_c_ratio,
            "stats": {
                "total": self.total_checks,
                "accepted": self.total_accepted,
                "rejected": self.total_rejected,
            },
        })
        self.pub_decision.publish(decision_msg)

        admissible_msg = Bool()
        admissible_msg.data = result.admissible
        self.pub_admissible.publish(admissible_msg)

        strength_msg = Float32()
        strength_msg.data = result.physics_strength
        self.pub_strength.publish(strength_msg)

        level = "info" if result.admissible else "warn"
        getattr(self.get_logger(), level)(
            f"Gate {'ACK' if result.admissible else 'NACK'}: "
            f"{result.verdict} (fc={result.physics_strength:.1f} MPa)"
        )

    def call_gate(self, payload: dict) -> GateResult | None:
        """POST the proposal to the Rust gate server and parse the response."""
        try:
            resp = requests.post(
                f"{self.gate_url}/gate",
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                self.get_logger().error(
                    f"Gate returned HTTP {resp.status_code}: {resp.text}"
                )
                return None

            data = resp.json()
            return GateResult(
                admissible=data.get("admissible", False),
                verdict=data.get("verdict", ""),
                violation=data.get("violation", ""),
                strength_bound=data.get("strength_bound", 0.0),
                physics_strength=data.get("physics_strength", 0.0),
                hydration_degree=data.get("hydration_degree", 0.0),
                w_c_ratio=data.get("w_c_ratio", 0.0),
            )

        except requests.exceptions.ConnectionError:
            self.get_logger().error("Gate server connection lost during check")
            self.gate_alive = False
            return None
        except Exception as e:
            self.get_logger().error(f"Gate call failed: {e}")
            return None


def main(args=None):
    rclpy.init(args=args)
    node = GateBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f"Shutting down — {node.total_checks} checks "
            f"({node.total_accepted} ACK, {node.total_rejected} NACK)"
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
