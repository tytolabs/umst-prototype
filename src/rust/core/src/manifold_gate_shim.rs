// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy — Studio TYTO
//!
//! Thin delegate to canonical `umst_manifold::gate::http_manifest` (stdlib-only gate math).

use serde::{Deserialize, Serialize};
use umst_manifold::gate::{
    evaluate_http_mix_manifest, gate_json_parse_response, hydration_degree,
    physics_compressive_strength_mpa, HttpGateManifest, HttpGateResponse, HttpMixProposal,
};

/// Canonical manifold `/gate` JSON (`admissible`, `codes`, `catalog_hash_hex`).
pub fn evaluate_canonical_json(body: &str) -> String {
    match serde_json::from_str::<HttpMixProposal>(body) {
        Ok(proposal) => {
            serde_json::to_string(&evaluate_http_mix_manifest(&proposal, &HttpGateManifest::default()))
                .unwrap_or_else(|_| serde_json::to_string(&gate_json_parse_response()).unwrap())
        }
        Err(_) => serde_json::to_string(&gate_json_parse_response()).unwrap(),
    }
}

/// Prototype / ROS2-facing gate JSON (legacy `gate_server` field names).
#[derive(Debug, Serialize)]
pub struct LegacyGateResponse {
    pub admissible: bool,
    pub verdict: String,
    pub violation: Option<String>,
    pub strength_bound: f64,
    pub physics_strength: f64,
    pub hydration_degree: f64,
    pub w_c_ratio: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub catalog_hash_hex: Option<String>,
}

/// Map manifold evaluation into legacy fields for `umst_ros2_bridge` without running prototype physics.
pub fn evaluate_legacy_json(body: &str) -> String {
    let proposal: HttpMixProposal = match serde_json::from_str(body) {
        Ok(p) => p,
        Err(e) => {
            return serde_json::to_string(&LegacyGateResponse {
                admissible: false,
                verdict: format!("Parse error: {e}"),
                violation: Some("invalid_request".into()),
                strength_bound: 0.0,
                physics_strength: 0.0,
                hydration_degree: 0.0,
                w_c_ratio: 0.0,
                catalog_hash_hex: None,
            })
            .unwrap();
        }
    };

    let manifest = HttpGateManifest::default();
    let canonical: HttpGateResponse = evaluate_http_mix_manifest(&proposal, &manifest);

    let total = proposal.cement + proposal.slag + proposal.fly_ash;
    let w_c = if total > 1.0e-9 {
        proposal.water / total
    } else {
        1.0
    };
    let scm_ratio = if total > 1.0e-9 {
        (proposal.slag + proposal.fly_ash) / total
    } else {
        0.0
    };
    let alpha = hydration_degree(proposal.age_days, proposal.temperature_c, scm_ratio);
    let physics_fc = physics_compressive_strength_mpa(
        w_c,
        alpha,
        manifest.air_void_fraction,
        manifest.strength_intrinsic_mpa,
    );
    let bound = physics_fc * (1.0 + manifest.admissibility_rel_margin);

    let (verdict, violation) = if canonical.admissible {
        (
            format!(
                "ACK: fc_physics={physics_fc:.1} MPa >= predicted={:.1} MPa",
                proposal.predicted_strength_mpa
            ),
            None,
        )
    } else {
        let code = canonical
            .codes
            .first()
            .cloned()
            .unwrap_or_else(|| "CLAUSIUS_GATE_STRENGTH_EXCESS".to_string());
        (
            format!(
                "NACK: predicted={:.1} exceeds bound={bound:.1} MPa ({code})",
                proposal.predicted_strength_mpa
            ),
            Some(code),
        )
    };

    serde_json::to_string(&LegacyGateResponse {
        admissible: canonical.admissible,
        verdict,
        violation,
        strength_bound: bound,
        physics_strength: physics_fc,
        hydration_degree: alpha,
        w_c_ratio: w_c,
        catalog_hash_hex: Some(canonical.catalog_hash_hex),
    })
    .unwrap()
}

/// Parse legacy prototype field names into [`HttpMixProposal`].
#[derive(Debug, Deserialize)]
struct LegacyGateRequest {
    cement: f64,
    #[serde(default)]
    slag: f64,
    #[serde(default)]
    fly_ash: f64,
    water: f64,
    #[serde(alias = "age")]
    age_days: f64,
    #[serde(alias = "predicted_strength")]
    predicted_strength_mpa: f64,
    #[serde(default = "default_temperature_c")]
    temperature_c: f64,
}

fn default_temperature_c() -> f64 {
    20.0
}

impl From<LegacyGateRequest> for HttpMixProposal {
    fn from(r: LegacyGateRequest) -> Self {
        HttpMixProposal {
            cement: r.cement,
            slag: r.slag,
            fly_ash: r.fly_ash,
            water: r.water,
            age_days: r.age_days,
            predicted_strength_mpa: r.predicted_strength_mpa,
            temperature_c: r.temperature_c,
        }
    }
}

/// Accept prototype `GateRequest` JSON keys and emit canonical manifold response.
pub fn evaluate_legacy_request_as_canonical(body: &str) -> String {
    match serde_json::from_str::<LegacyGateRequest>(body) {
        Ok(legacy) => {
            let proposal: HttpMixProposal = legacy.into();
            serde_json::to_string(&evaluate_http_mix_manifest(
                &proposal,
                &HttpGateManifest::default(),
            ))
            .unwrap_or_else(|_| serde_json::to_string(&gate_json_parse_response()).unwrap())
        }
        Err(_) => serde_json::to_string(&gate_json_parse_response()).unwrap(),
    }
}
