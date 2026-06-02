# Thin prototype integration status

Last updated: 2026-05-21 (MaOS workspace sibling layout).

## Sibling path dependencies

| Crate | `umst-manifold` path dep | `umst-concrete-cartridge` path dep | Notes |
|-------|--------------------------|-------------------------------------|-------|
| `umst-prototype` → `umst-core` | Yes — **required** path dep (`thermodynamic_filter` shim) | No | Cartridge is a separate workspace; use `[patch]` in `.cargo/config.toml` (see example below). |
| `umst-prototype-2a` → `umst-core` | Yes — optional, feature `manifold-gate` | No | Same as above. |
| `umst-concrete-cartridge` workspace | Git `main` (CI) | N/A | Upstream documents **gitignored** `[patch]` for local `../umst-manifold` — do not commit workspace-root patch in that repo. |

Relative paths (from `*/src/rust/core/Cargo.toml`):

- `umst-prototype`: `../../../../umst-manifold`
- `umst-prototype-2a`: `../../../../../umst-manifold`

## `cargo check` results (2026-05-21)

| Target | Command | Status |
|--------|---------|--------|
| `umst-prototype` default | `cd umst-prototype/src/rust/core && cargo check` | **OK** (pulls `burn` 0.13 via required manifold dep) |
| `umst-prototype` + shim | `cargo check --features manifold-gate` | **OK** (default feature; `manifold_gate_shim` module) |
| `umst-prototype-2a` default | `cd umst-prototype-2a/prototype/src/rust && cargo check` | **OK** |
| `umst-prototype-2a` + shim | `cargo check -p umst-core --features manifold-gate` | **OK** (dual `burn` 0.16 + 0.13 in same graph — parity-only) |
| `umst-manifold` gate bin | `cd umst-manifold && cargo check --features gate-server --bin gate_server` | **OK** |

## Manifold gate shim (Rust)

Gate server choice and **100% dual-run** commands: **`GATE_SERVER.md`**.

With `manifold-gate` enabled on `umst-core`:

- Module: `manifold_gate_shim` → `umst_manifold::gate::http_manifest`
- Binary: `manifold_gate_parity` (stdin JSON → canonical + legacy JSON)

**Canonical HTTP server** (ROS2 / lab default going forward):

```bash
cd umst-manifold
UMST_GATE_ADDR=0.0.0.0:8787 cargo run --features gate-server --bin gate_server
```

Default listen: `8787` (not legacy prototype `8765`).

## ROS2 bridge

- README and `config/bridge_params.yaml` point at **manifold** `gate_server`.
- `gate_bridge_node.py` accepts both response shapes:
  - Legacy: `verdict`, `physics_strength`, …
  - Manifold: `codes`, `catalog_hash_hex` (mapped to legacy fields for topics)

Set `gate_url:=http://localhost:8787` when using manifold.

## Thermodynamic filter (2026-05-21)

| Crate | `thermodynamic_filter.rs` | Dual-run lane |
|-------|---------------------------|---------------|
| `umst-prototype` | **Shim** (~226 lines): WASM types + `ThermodynamicFilter` delegate to `umst_manifold::gate::mix_proposal` | `gate_dual_fixture` + golden fixtures (**8/8**) |
| `umst-prototype-2a` | **Hybrid** (~480 lines): Algorithm 1 delegates when `manifold-gate`; 2a layers stay local | Not in `gate_dual_run_parity` subprocess (uses v1 `gate_dual_fixture`) |

Parity command (SSOT exit for v1 shim + manifold `mix_proposal`):

```bash
cd umst-manifold && cargo test --test gate_dual_run_parity -- --nocapture
# → manifold vs prototype_golden 8/8 (100%)
# → manifold vs live prototype subprocess 8/8 (100%)
```

2a unit tests (default + optional delegation):

```bash
cd umst-prototype-2a/prototype/src/rust
cargo test -p umst-core thermodynamic_filter
cargo test -p umst-core --features manifold-gate thermodynamic_filter
```

### `umst-prototype-2a` hybrid layout (post 8/8)

| Layer | Owner | `manifold-gate` |
|-------|--------|-----------------|
| Mass bound, `D_int = −ρ ψ̇`, Powers strength monotonicity | `ThermodynamicMixFilter::check_transition` (SSOT) | Delegated |
| Explicit α irreversibility (`hydration_irreversible`) | 2a `second_law_extensions` | Local |
| `max_strength` topology cap (LLM strength hallucination guard) | 2a `ThermodynamicState` + extensions | Local |
| `Constitution::verify_transition` + DCS/CGS | 2a `science/constitution.rs` | Local |
| `evaluate_joint_functor` (MARL superposition veto) | 2a filter | Local |

Default `cargo build` / PPO / Burn **0.16** graph: **no** `umst-manifold` (inline Algorithm 1). Enable `manifold-gate` only for parity or gate-shim binaries.

### Constitution / CGS / MARL gaps vs manifold (do not thin yet)

| Capability | `umst-manifold` (`mix_proposal` / HTTP gate) | `umst-prototype-2a` |
|------------|-----------------------------------------------|------------------------|
| **Constitution** | No `PhysicalAxiom` witnesses; verdict is flat `AdmissibilityVerdict` | `Constitution::standard()` — Mass, Hydration, Clausius-Duhem, Strength axioms with formal refs (`umst-formal/*`) |
| **CGS / DCS** | Not on transition outcome | `AdmissibilityResult.cgs` (`9.5` / `3.0`) + `Constitution::score_transition` → `compute_dcs` |
| **`hydration_irreversible` flag** | Implicit via negative `D_int` on reverse α; not a separate field | Explicit invariant + rejection reason `HYDRATION_IRREVERSIBILITY_VIOLATION` |
| **`max_strength`** | `ThermodynamicStateSnapshot` has no cap field | Categorical bound; rejects `strength > max_strength` even when Clausius scalar passes |
| **MARL** | No joint superposition API | `evaluate_joint_functor` — density/ψ flux superposition + weakest-voxel strength, then `check_transition` |
| **Mass tolerance** | Fixed `|Δρ| < 100` kg/m³ | Constitution axiom: `|Δρ| < 0.01·ρ_old` (stricter on light transitions) |
| **Clausius check in Constitution** | N/A | Instantaneous `ψ_new − ψ_old` (no `dt`); inline gate uses `ψ̇` with `dt` — both retained intentionally |

**When to thin 2a further:** manifold exports Constitution/DCS witnesses (or JSON catalog), `max_strength` on snapshots, `evaluate_joint_transition`, and a **2a** `gate_dual_fixture` lane is green against the same `gate_dual_run_fixtures.json` plus MARL/topology cases.

### Why v1 stays a full shim (unchanged)

`umst-prototype` requires `umst-manifold` on the default graph (WASM/PPO). `umst-prototype-2a` keeps manifold **optional** so Burn **0.16** training does not pull Burn **0.13** unless parity is requested.

## Blockers (not yet wired)

1. **Crate rename / façade** — Prototypes remain `umst-core`; manifold library is `umst_manifold`. Full in-tree migration needs import rewrites or a thin façade crate (see `umst-manifold/docs/PROTOTYPE_GATE_MAP.md`).
2. **`umst-prototype-2a` Burn 0.16** — Default tensor stack pins `burn = 0.16`; manifold pins `0.13`. Optional `manifold-gate` is for parity only, not unified training.
3. **`umst-concrete-cartridge` in prototype Cargo.toml** — Would drag full workspace + conflicting Burn pins; use HTTP gate parity or patch manifold when developing cartridge locally.
4. **Telemetry WebSocket** — Still prototype `gate_server` on `8766` in 2a; manifold gate binary is HTTP-only today.
5. **OCR endpoints** — `umst-prototype-2a` `gate_server` extras are not in manifold `gate_server`.

## Local cartridge + manifold patch (optional)

Copy to `umst-prototype/src/rust/.cargo/config.toml` (gitignored):

```toml
[patch."https://github.com/tytolabs/umst-manifold.git"]
umst-manifold = { path = "../../../umst-manifold" }
```

Then add a dev-dependency on `umst-concrete-cartridge` only in a dedicated example crate — not in default `umst-core` build.

## Related docs

- **`GATE_SERVER.md`** — manifold vs legacy `gate_server`, dual-run 100% parity commands
- `THIN_PROTOTYPES_PATH_DEPS.md` (short pointer)
- `umst-manifold/docs/PROTOTYPE_2A_HOST_GAPS.md` — absorb vs stay (Constitution/CGS/MARL/`max_strength`), Mermaid composition
- `umst-manifold/docs/PROTOTYPE_GATE_MAP.md`
- `umst-manifold/docs/GateUnificationSpec.md`
