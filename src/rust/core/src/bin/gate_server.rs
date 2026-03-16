// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
//!
//! Minimal UMST Gate Server (Original Prototype)
//! ==============================================
//! HTTP server exposing the physics kernel's thermodynamic gate via REST.
//!
//! Usage:
//!   cargo run --bin gate_server
//!   # Listens on http://0.0.0.0:8765
//!
//! Endpoints:
//!   POST /gate     — thermodynamic check
//!   GET  /health   — health check

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use umst_core::physics_kernel::{PhysicsConfig, PhysicsKernel};

#[derive(Deserialize, Debug)]
struct GateRequest {
    cement: f32,
    #[serde(default)]
    slag: f32,
    #[serde(default)]
    fly_ash: f32,
    water: f32,
    age: f32,
    predicted_strength: f32,
    #[serde(default = "default_temp")]
    temperature_c: f32,
}

fn default_temp() -> f32 { 20.0 }

#[derive(Serialize)]
struct GateResponse {
    admissible: bool,
    verdict: String,
    violation: Option<String>,
    strength_bound: f32,
    physics_strength: f32,
    hydration_degree: f32,
    w_c_ratio: f32,
}

fn handle_gate(body: &str) -> String {
    let req: GateRequest = match serde_json::from_str(body) {
        Ok(r) => r,
        Err(e) => {
            return serde_json::to_string(&GateResponse {
                admissible: false,
                verdict: format!("Parse error: {e}"),
                violation: Some("invalid_request".into()),
                strength_bound: 0.0,
                physics_strength: 0.0,
                hydration_degree: 0.0,
                w_c_ratio: 0.0,
            }).unwrap();
        }
    };

    let total_cement = req.cement + req.slag + req.fly_ash;
    let w_c = if total_cement > 0.0 { req.water / total_cement } else { 1.0 };
    let scm_ratio = if total_cement > 0.0 { (req.slag + req.fly_ash) / total_cement } else { 0.0 };

    let alpha = PhysicsKernel::compute_hydration_degree(req.age, req.temperature_c, scm_ratio);
    let air = 0.02;
    let config = PhysicsConfig::default();
    let strength = umst_core::science::strength::StrengthEngine::compute_powers(
        w_c, alpha, air, config.s_intrinsic,
    );

    let physics_fc = strength.compressive_strength;
    let admissible = req.predicted_strength <= physics_fc * 1.15;

    let (verdict, violation) = if admissible {
        (format!("ACK: fc_physics={physics_fc:.1} MPa >= predicted={:.1} MPa", req.predicted_strength), None)
    } else {
        (
            format!("NACK: predicted={:.1} exceeds fc_physics={physics_fc:.1} MPa", req.predicted_strength),
            Some("clausius_duhem_violation".into()),
        )
    };

    serde_json::to_string(&GateResponse {
        admissible,
        verdict,
        violation,
        strength_bound: physics_fc,
        physics_strength: physics_fc,
        hydration_degree: alpha,
        w_c_ratio: w_c,
    }).unwrap()
}

fn handle_request(stream: &mut std::net::TcpStream) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() { return; }

    let mut headers = Vec::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() { break; }
        if line.trim().is_empty() { break; }
        headers.push(line);
    }

    let content_length: usize = headers.iter()
        .find(|h| h.to_lowercase().starts_with("content-length:"))
        .and_then(|h| h.split(':').nth(1))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0);

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        let _ = reader.read_exact(&mut body);
    }
    let body_str = String::from_utf8_lossy(&body);

    let (status, response_body) = if request_line.starts_with("POST /gate") {
        ("200 OK", handle_gate(&body_str))
    } else if request_line.starts_with("GET /health") {
        ("200 OK", r#"{"status":"ok"}"#.to_string())
    } else {
        ("404 Not Found", r#"{"error":"not found"}"#.to_string())
    };

    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
        response_body.len(),
        response_body
    );
    let _ = stream.write_all(response.as_bytes());
}

fn main() {
    let addr = "0.0.0.0:8765";
    let listener = TcpListener::bind(addr).expect("Failed to bind to port 8765");
    eprintln!("UMST Gate Server (original) listening on {addr}");

    for stream in listener.incoming() {
        match stream {
            Ok(mut s) => handle_request(&mut s),
            Err(e) => eprintln!("Connection error: {e}"),
        }
    }
}
