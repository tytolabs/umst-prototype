# Gate server — manifold vs legacy prototype

When to run which HTTP binary, how ROS2 maps responses, and how **100% dual-run parity** is enforced in CI. Prototype **`ThermodynamicFilter`** is a deprecation shim to `umst_manifold::gate::mix_proposal` (8/8 parity) — see [THIN_PROTOTYPE_STATUS.md](./THIN_PROTOTYPE_STATUS.md).

## Which server to use

| Goal | Run | Default port |
|------|-----|--------------|
| **SSOT / ROS2 / new integrations** | `umst-manifold` `gate_server` | **8787** (`UMST_GATE_ADDR`) |
| **Reproduce old papers, scripts, or 8765-only clients** | `umst-core` `gate_server` (this repo) | **8765** |
| **Compare one mix JSON without HTTP** | `manifold_gate_parity` (feature `manifold-gate`) | stdin → stdout |

### Canonical (prefer)

```bash
cd umst-manifold
UMST_GATE_ADDR=0.0.0.0:8787 cargo run --features gate-server --bin gate_server
```

- Response: `admissible`, `codes[]`, `catalog_hash_hex` (`gate::http_manifest`).
- Catalog digest pinned at manifold build time.

### Legacy prototype HTTP

```bash
cd umst-prototype/src/rust/core
cargo run -p umst-core --bin gate_server
```

- Response: `admissible`, `verdict`, `physics_strength`, `hydration_degree`, `w_c_ratio`, …
- Inline `PhysicsKernel` / strength engine — **not** catalog-pinned SSOT.
- Startup prints a **deprecation notice** to stderr; behavior unchanged.

Use legacy only when a client hard-codes **8765** or legacy field names and cannot be updated yet. For ROS2, set `gate_url:=http://localhost:8787` (see `ros2_bridge/` and `umst-prototype-2a/ros2_bridge/README.md`).

### Do not run both on the same host/port

| Server | Port |
|--------|------|
| manifold | 8787 |
| prototype | 8765 |

`umst-prototype-2a` may also bind **8766** (WebSocket telemetry); manifold gate is **HTTP-only** today.

## HTTP contract (mix proposal)

Same request body keys work on both stacks (prototype aliases accepted by manifold):

```json
{
  "cement": 400,
  "water": 200,
  "age": 28,
  "predicted_strength": 25,
  "temperature_c": 20
}
```

| Field (prototype) | Manifold |
|-------------------|----------|
| `age` | `age_days` |
| `predicted_strength` | `predicted_strength_mpa` |

`gate_bridge_node` accepts canonical or legacy-shaped **responses**.

## Dual-run 100% parity (reference)

Two layers — prototype `science/thermodynamic_filter.rs` is a **shim** (not duplicate math); keep `gate_dual_fixture` until live lane migrates off prototype subprocess.

### 1. Transition filter (mix / snapshot fixtures)

**Owner test:** `umst-manifold/tests/gate_dual_run_parity.rs`  
**Fixtures:** `umst-manifold/tests/data/gate_dual_run_fixtures.json` (goldens from prototype unit tests)  
**Subprocess helper:** `umst-core` binary `gate_dual_fixture` (stdin bundle → `{ "results": [...] }`)

```bash
cd umst-manifold
cargo test gate_dual_run_parity -- --nocapture
```

- `mix_proposal_gate_matches_prototype_golden_vectors` — asserts **100%** agreement vs embedded golden vectors (`golden_agree == total`).
- `mix_proposal_gate_live_subprocess_matches_manifold_when_available` — when `gate_dual_fixture` is built, asserts **100%** live agreement vs `ThermodynamicFilter` in `umst-prototype`.

Manifold side: `ThermodynamicMixFilter` / `MixProposalScalars`. Prototype side: `ThermodynamicFilter` (unchanged source of truth for this parity lane).

### 2. HTTP mix gate JSON (canonical vs legacy mapping)

**CLI:** `manifold_gate_parity` (requires `--features manifold-gate`)

```bash
cd umst-prototype/src/rust/core
cargo run -p umst-core --features manifold-gate --bin manifold_gate_parity \
  <<<'{"cement":400,"water":200,"age":28,"predicted_strength":25}'
```

Prints one line: `{"canonical":…,"legacy":…}` where `legacy` is `manifold_gate_shim::evaluate_legacy_json` (ROS2 field names, same admissibility as manifold `http_manifest`).

### 3. Manifest dual-run (manifold internal)

`UmstManifest::dual_run` runs transition gate and CBF independently; reject if either fails. Documented in `umst-manifold/docs/GateUnificationSpec.md` — separate from prototype HTTP servers.

## Optional `manifold-gate` feature

In `umst-core` `Cargo.toml`, feature **`manifold-gate`** adds path dep `umst-manifold` and module `manifold_gate_shim`. Default builds stay prototype-only (no Burn 0.13 pull). Use for parity CLI and shim development only — not for production gate serving.

## Migration checklist

- [ ] Point clients at **8787** and manifold `gate_server`
- [ ] Keep `cargo test gate_dual_run_parity` at **100%** after any filter or manifest change
- [ ] Run `manifold_gate_parity` on representative mix JSON when touching `http_manifest`
- [ ] Remove prototype `gate_server` only after no 8765 consumers and parity tests absorb any remaining behavior (e.g. 2a OCR / `/gate/full`, telemetry WS)

## Related

- [THIN_PROTOTYPE_STATUS.md](./THIN_PROTOTYPE_STATUS.md) — `cargo check` matrix, blockers
- [THIN_PROTOTYPES_PATH_DEPS.md](./THIN_PROTOTYPES_PATH_DEPS.md) — path deps pointer
- `umst-manifold/docs/PROTOTYPE_GATE_MAP.md` — path → module map
- `umst-manifold/docs/GateUnificationSpec.md` — `catalog_id`, dual-run policy
