// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
// SPDX-License-Identifier: MIT
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize)]
pub struct SafetyReport {
    pub allowed: bool,
    pub reason: Option<String>,
    pub risk_score: f32,
}

#[wasm_bindgen]
pub struct SafetyGuard {
    // Compiled regexes could be cached here if we modify struct to hold state
}

#[wasm_bindgen]
impl SafetyGuard {
    #[wasm_bindgen(constructor)]
    pub fn new() -> SafetyGuard {
        SafetyGuard {}
    }

    pub fn validate_tool_call(&self, tool_name: &str, args_json: &str) -> Result<JsValue, JsValue> {
        let report = self.check(tool_name, args_json);
        Ok(serde_wasm_bindgen::to_value(&report)?)
    }

    fn check(&self, tool_name: &str, args: &str) -> SafetyReport {
        // 1. Precise Whitelist using Match
        match tool_name {
            "query_knowledge_base" => self.validate_query_safety(args),
            "spawn_panel" => self.validate_panel_config(args),
            // Safe read-only tools
            "list_ui_actions" | "list_datasets" | "get_system_status" => SafetyReport::safe(),
            // Safe Action tools (with no dangerous params)
            "execute_ui_action" => SafetyReport::safe(),
            _ => SafetyReport::deny("Tool not in Allowlist"),
        }
    }

    fn validate_query_safety(&self, args: &str) -> SafetyReport {
        // "Mutation Defense"
        // Reject any SQL-like mutation keywords, even if Dexie is used.
        // It enforces a "Read-Only" operational mode for the AI unless explicitly upgraded.
        let forbidden = ["delete", "drop", "clear", "put", "update"];
        let lower = args.to_lowercase();

        for term in forbidden {
            if lower.contains(term) {
                return SafetyReport::deny(&format!("Mutation disallowed: found '{}'", term));
            }
        }
        SafetyReport::safe()
    }

    fn validate_panel_config(&self, args: &str) -> SafetyReport {
        // Ensure it's valid JSON
        if serde_json::from_str::<serde_json::Value>(args).is_err() {
            return SafetyReport::deny("Invalid JSON");
        }
        SafetyReport::safe()
    }

    pub fn check_robotics_safety(&self, velocity: f64, load_kg: f64) -> Result<JsValue, JsValue> {
        let report = if velocity > 250.0 {
            SafetyReport::deny(&format!(
                "Velocity Limit Exceeded: {:.1}mm/s > 250.0mm/s (ISO 10218)",
                velocity
            ))
        } else if load_kg > 10.0 {
            SafetyReport::deny(&format!(
                "Payload Limit Exceeded: {:.1}kg > 10.0kg",
                load_kg
            ))
        } else {
            SafetyReport::safe()
        };

        Ok(serde_wasm_bindgen::to_value(&report)?)
    }
}

impl SafetyReport {
    fn safe() -> Self {
        SafetyReport {
            allowed: true,
            reason: None,
            risk_score: 0.0,
        }
    }
    fn deny(reason: &str) -> Self {
        SafetyReport {
            allowed: false,
            reason: Some(reason.to_string()),
            risk_score: 1.0,
        }
    }
}

#[wasm_bindgen]
pub struct Watchdog {
    last_tick: f64, // JS timestamp (performance.now())
    timeout_ms: f64,
}

#[wasm_bindgen]
impl Watchdog {
    #[wasm_bindgen(constructor)]
    pub fn new(timeout_ms: f64) -> Watchdog {
        Watchdog {
            last_tick: 0.0,
            timeout_ms,
        }
    }

    pub fn heartbeat(&mut self, now: f64) {
        self.last_tick = now;
    }

    pub fn is_safe(&self, now: f64) -> bool {
        if self.last_tick == 0.0 {
            return false;
        } // Not started
        (now - self.last_tick) < self.timeout_ms
    }
}
