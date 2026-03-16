# Thermodynamic gate decision — ACK/NACK for a proposed state transition.
# Enforces the Clausius-Duhem inequality: D_int >= 0.

std_msgs/Header header

bool    admissible              # true = ACK (transition is safe)
string  verdict                 # human-readable summary
string  violation               # empty if admissible; otherwise the violated invariant
float32 strength_bound          # physics-predicted strength (MPa)
float32 physics_strength        # constitutive model output (MPa)
float32 hydration_degree        # computed alpha
float32 w_c_ratio               # water-cement ratio
