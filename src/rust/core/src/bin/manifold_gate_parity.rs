// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy — Studio TYTO
//!
//! HTTP mix-gate parity CLI (canonical + legacy JSON). See `umst-prototype/docs/GATE_SERVER.md`.
//!
//! ```text
//! cargo run -p umst-core --features manifold-gate --bin manifold_gate_parity <<<'{"cement":400,"water":200,"age":28,"predicted_strength":25}'
//! ```

use std::io::{self, Read};

fn main() {
    let mut body = String::new();
    io::stdin()
        .read_to_string(&mut body)
        .expect("read stdin");
    let canonical = umst_core::manifold_gate_shim::evaluate_canonical_json(&body);
    let legacy = umst_core::manifold_gate_shim::evaluate_legacy_json(&body);
    println!("{{\"canonical\":{canonical},\"legacy\":{legacy}}}");
}
