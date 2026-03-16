# UMST ROS2 Bridge

ROS2 package bridging the UMST thermodynamic gate (Rust physics kernel) to the
ROS2 ecosystem for physics-gated robotic fabrication.

## Architecture

```
┌─────────────────────┐     REST POST /gate     ┌──────────────────┐
│  gate_bridge_node   │ ──────────────────────→  │  Rust Gate Server│
│  (ROS2 Python)      │ ←────────────────────── │  (port 8765)     │
└──────┬──────────────┘     JSON response        └──────────────────┘
       │
       ├── /umst/decision    (String, JSON)
       ├── /umst/admissible  (Bool)
       └── /umst/physics_strength (Float32)

┌─────────────────────┐     WebSocket            ┌──────────────────┐
│ telemetry_bridge    │ ←─────────────────────── │  Rust Telemetry  │
│  (ROS2 Python)      │                          │  (port 8766)     │
└──────┬──────────────┘                          └──────────────────┘
       ├── /umst/telemetry/state       (String)
       ├── /umst/telemetry/temperature (Float32)
       └── /umst/telemetry/hydration   (Float32)

┌─────────────────────┐
│ robot_controller    │ ←── /umst/decision
│  (ROS2 Python)      │
└──────┬──────────────┘
       ├── /robot/cmd           (String, JSON) — only if admissible
       └── /umst/replan_request (String, JSON) — on rejection
```

## Quick Start

```bash
# Terminal 1: Start the Rust gate server
cd prototype && cargo run --release --bin gate_server

# Terminal 2: Launch the ROS2 bridge (requires ROS2 Humble/Iron/Jazzy)
source /opt/ros/$ROS_DISTRO/setup.bash
cd ros2_bridge && colcon build && source install/setup.bash
ros2 launch umst_ros2_bridge umst_bridge.launch.py

# Terminal 3: Test with a proposal
ros2 topic pub /umst/proposal std_msgs/String "data: '{
  \"cement\": 350, \"water\": 175, \"age\": 28,
  \"predicted_strength\": 35, \"coarse_agg\": 1000,
  \"fine_agg\": 750, \"temperature_c\": 20, \"dataset\": \"D1\"
}'" --once
```

## Nodes

| Node | Purpose | Subscribes | Publishes |
|------|---------|-----------|-----------|
| `gate_bridge` | REST bridge to Rust gate | `/umst/proposal` | `/umst/decision`, `/umst/admissible` |
| `telemetry_bridge` | WebSocket telemetry relay | (WebSocket) | `/umst/telemetry/*` |
| `robot_controller` | Demo controller (dry-run) | `/umst/decision` | `/robot/cmd`, `/umst/replan_request` |

## Custom Messages

- `UMSTState.msg` — Material state tensor fields
- `GateDecision.msg` — Thermodynamic gate ACK/NACK
- `GateCheck.srv` — On-demand gate validation service

## Parameters

See `config/bridge_params.yaml` for all configurable parameters.

## Safety Invariant

No robot command is issued unless the thermodynamic gate returns `admissible: true`.
This is the ROS2-side enforcement of the Clausius-Duhem inequality.
