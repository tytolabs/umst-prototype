// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
// SPDX-License-Identifier: MIT
//
// Subprocess helper for `umst-manifold/tests/gate_dual_run_parity.rs` (100% parity lane).
// Reads one fixture bundle JSON on stdin; writes `{ "results": [...] }` on stdout.
// See `umst-prototype/docs/GATE_SERVER.md`.

use std::io::{self, Read};

use serde::{Deserialize, Serialize};
use umst_core::science::thermodynamic_filter::{ThermodynamicFilter, ThermodynamicState};

#[derive(Debug, Deserialize)]
struct MixInput {
    w_c: f64,
    alpha: f64,
    temp_k: f64,
    #[serde(default)]
    s_intrinsic_mpa: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct SnapshotInput {
    density: f64,
    temperature: f64,
    free_energy: f64,
    entropy: f64,
    hydration_degree: f64,
    strength: f64,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
enum FixtureCase {
    FromMix {
        old: MixInput,
        new: MixInput,
        dt_seconds: f64,
    },
    ExplicitSnapshot {
        old: SnapshotInput,
        new: SnapshotInput,
        dt_seconds: f64,
    },
}

#[derive(Debug, Deserialize)]
struct FixtureBundle {
    cases: Vec<FixtureCase>,
}

#[derive(Debug, Serialize, PartialEq)]
struct TransitionResult {
    accepted: bool,
    dissipation: f64,
    mass_conserved: bool,
    energy_positive: bool,
}

fn mix_to_state(m: &MixInput) -> ThermodynamicState {
    match m.s_intrinsic_mpa {
        Some(s) => ThermodynamicState::from_mix_calibrated(m.w_c, m.alpha, m.temp_k, s),
        None => ThermodynamicState::from_mix(m.w_c, m.alpha, m.temp_k),
    }
}

fn snapshot_to_state(s: &SnapshotInput) -> ThermodynamicState {
    ThermodynamicState {
        density: s.density,
        temperature: s.temperature,
        free_energy: s.free_energy,
        entropy: s.entropy,
        hydration_degree: s.hydration_degree,
        strength: s.strength,
    }
}

fn main() {
    let mut buf = String::new();
    io::stdin()
        .read_to_string(&mut buf)
        .expect("read stdin fixture bundle");
    let bundle: FixtureBundle = serde_json::from_str(&buf).expect("parse fixture JSON");

    let mut filter = ThermodynamicFilter::new();
    let mut results = Vec::with_capacity(bundle.cases.len());

    for case in &bundle.cases {
        let (old, new, dt) = match case {
            FixtureCase::FromMix { old, new, dt_seconds } => {
                (mix_to_state(old), mix_to_state(new), *dt_seconds)
            }
            FixtureCase::ExplicitSnapshot { old, new, dt_seconds } => {
                (
                    snapshot_to_state(old),
                    snapshot_to_state(new),
                    *dt_seconds,
                )
            }
        };
        let r = filter.check_transition(&old, &new, dt);
        results.push(TransitionResult {
            accepted: r.accepted,
            dissipation: r.dissipation,
            mass_conserved: r.mass_conserved,
            energy_positive: r.energy_positive,
        });
    }

    let out = serde_json::json!({ "results": results });
    println!("{}", serde_json::to_string(&out).expect("serialize results"));
}
