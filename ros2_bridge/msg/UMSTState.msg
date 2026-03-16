# UMST Material State — ROS2 message for the Unified Material-State Tensor
# Maps to the Rust MixTensor / GateRequest on the physics kernel side.

std_msgs/Header header

# Material composition (kg/m³)
float32 cement
float32 slag
float32 fly_ash
float32 water
float32 coarse_agg
float32 fine_agg

# Process state
float32 age                    # days
float32 temperature            # °C
string  dataset                # dataset identifier (D1, UHPC, LUNAR, etc.)

# Physics kernel outputs (populated after gate check)
float32 predicted_strength     # MPa
float32 hydration_degree       # [0, 1]
float32 w_c_ratio
