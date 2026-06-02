# Thin prototypes — path dependency notes

See **`THIN_PROTOTYPE_STATUS.md`** for current `cargo check` matrix, ROS2 defaults, and blockers.  
See **`GATE_SERVER.md`** for manifold vs legacy HTTP gate and dual-run parity.

Optional in `umst-core`: `umst-manifold = { path = "../../../../umst-manifold", … }` behind feature **`manifold-gate`** (shim module + `manifold_gate_parity` bin). Default builds stay **`umst-core`**-only.

Canonical HTTP gate: **`cargo run -p umst-manifold --features gate-server --bin gate_server`** (`UMST_GATE_ADDR`, default port **8787**).
