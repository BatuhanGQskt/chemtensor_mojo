# Tests

## Introduction

The `src/tests/` folder exists to validate the Mojo implementation against the C ground truth and to check internal consistency. Reference data is produced by the C project and lives in `test_data/` as JSON files (MPO, DMRG, and MPS product-state dumps). The C implementation is treated as the reference; Mojo tests compare against these dumps or against observable quantities derived from them.

## Setup (C repository)

Clone the C reference implementation so it sits **sibling** to the Mojo project. `run_main.sh` in the C repo expects this layout and copies generated JSON into the Mojo `test_data/` directory.

Recommended layout (the parent folder can be named anything; only the sibling structure matters):

```
<parent_folder>/
├── chemtensor/          # C repo (clone here)
│   ├── run_main.sh
│   └── ...
└── chemtensor_mojo/
    └── chemtensor_mojo/ # this Mojo project
        ├── test_data/   # receives JSON from C
        └── src/tests/
```

Clone the C repository into the same parent directory as `chemtensor_mojo`:

```bash
cd /path/to/parent_folder
git clone git@github.com:BatuhanGQskt/chemtensor_c_replica.git
```

After cloning, the C project path is `../chemtensor` relative to `chemtensor_mojo/chemtensor_mojo`. The script `run_main.sh` uses `MOJO_TEST_DATA="${SCRIPT_DIR}/../chemtensor_mojo/chemtensor_mojo/test_data"` to copy files.

## How to run tests

### Step 1: Generate reference data (C)

From the C repository root (e.g. `chemtensor/` in the layout above), run:

```bash
./run_main.sh
```

This script builds the C project, runs `./main`, and copies all generated JSON from `build/generated/` into this project’s `test_data/`:

- `generated/mpo/*.json` → `test_data/`
- `generated/dmrg/*.json` → `test_data/`
- `generated/mps/*.json` → `test_data/`

Run this whenever the C output or reference data schema changes.

### Step 2: Run Mojo tests

From the Mojo project root (`chemtensor_mojo/chemtensor_mojo`):

```bash
./tools/run_tests.sh
```

This discovers and runs all `test_*.mojo` files under `src/tests` (default). Options:

- `./tools/run_tests.sh [-dw|--disable-warnings] [-q|--quiet] [test_dir]`
- Example: `./tools/run_tests.sh -dw src/tests/algorithms`
- Use `--quiet` or `-q` to hide per-test output and only show PASSED/FAILED lines plus the final summary.

At the end, a **test summary** is printed: results grouped by directory (e.g. `src/tests/state/mps/`), with per-file pass/fail and, when available, per-function status (from Mojo `TestSuite` output).

To run a single test file:

```bash
mojo run -I . src/tests/<path>/test_<name>.mojo
```

Example:

```bash
mojo run -I . src/tests/m_tensor/test_c_replica_dot.mojo
```

## test_data/

This directory holds C-generated JSON used as reference by Mojo tests:

- **MPO:** e.g. `ising_1d_*.json`, `heisenberg_xxz_*.json`, `bose_hubbard_*.json`
- **DMRG:** `dmrg_results_c_singlesite.json`, `dmrg_results_c_twosite.json`
- **MPS product states:** `mps_product_*.json`

Regenerate these by running the C project’s `run_main.sh` (see Step 1). Do not edit the JSON by hand for test purposes.

---

## Summaries by folder

### algorithms/

**Algorithm tests.** DMRG tests are grouped under **algorithms/dmrg/** (see `algorithms/dmrg/README.md`).

- **dmrg/** — DMRG test suite:
  - **test_dmrg.mojo** — Manual runs (7-site XXX, 11-site XXZ), prints results, writes JSON. Asserts norm ≈ 1, energy in range. Mirrors `chemtensor/manual_tests/dmrg.c`.
  - **test_dmrg_gauge_safe.mojo** — Gauge-invariant checks (TFIM): energy = ⟨ψ|H|ψ⟩, variance small, observables in range (Tiers A/B/C).
  - **test_dmrg_c_comparison.mojo** — Loads C reference from `test_data/dmrg_results_c_*.json`, runs Mojo DMRG with same parameters, compares `energy_final`, `bond_dims` length, norm.
  - **test_dmrg_exact_small.mojo** — 4-site Heisenberg XXX exact ground state energy (E₀ = -3) without C reference.
  - **dmrg_json_loader.mojo** — Loads DMRG reference JSON (used by C comparison test).

**C reference refresh:** In `chemtensor`, run `run_main.sh` so that C builds, runs DMRG, and copies `generated/dmrg/dmrg_results_c_*.json` into this project’s `test_data/`.

**Energy conventions (C vs Mojo):** For strict comparison, generate the C reference with the manual Heisenberg XXZ MPO (same convention as Mojo). See `algorithms/dmrg/README.md`.

---

### state/mpo/

**MPO tests.** Validate the Mojo MPO implementation against C and against physical properties.

- **test_mpo_c_comparison.mojo** — Element-wise comparison with C reference (Ising 2- and 4-site, Heisenberg XXZ 3-site, Bose-Hubbard 3-site). Tolerances: rtol=1e-10, atol=1e-12.
- **test_mpo_observables.mojo** — Physical checks: Hermiticity, ground state energy (analytical where possible), eigenvalue spectrum, trace. Models: Ising (2- and 4-site), Heisenberg XXZ (3-site), Identity MPO.
- **test_mpo_operations.mojo** — Tensor operations: MPO merge (adjacent sites), full MPO contraction to dense matrix, transpose, reshape (Ising 2-, 3-, 4-site).
- **test_mpo.mojo** — Construction and basic operations for Ising, Heisenberg, Heisenberg XXZ, identity MPO, merge/transpose/reshape/scale and consistency.
- Helpers/loaders: `mpo_test_helpers.mojo`, `mpo_json_loader.mojo`.

**Reference data:** JSON files in `test_data/` (e.g. `ising_1d_*.json`, `heisenberg_xxz_*.json`, `bose_hubbard_*.json`). Regenerate from C: build and run C so it exports MPO JSON, then run `run_main.sh` to copy into `test_data/`.

**Hamiltonians:** Ising (d=2, bond dim 3), Heisenberg XXZ (d=2, bond dim 5), Bose-Hubbard (d=3, bond dim 4). Bond structure: left boundary 1, bulk as above, right boundary 1.

**MPO tensor layout:** All MPO sites use shape `[Wl, d_in, d_out, Wr]` in row-major (C-contiguous) order: Wl = left virtual bond, d_in = physical input (ket), d_out = physical output, Wr = right virtual bond. Data must be filled in this index order; do not build as `[Wl, Wr, d, d]` and transpose, or the memory layout will not match C.

---

### state/mps/

**MPS tests.**

- **test_mps_extensive.mojo** — Product states, norms, state vectors, overlaps. No C reference; exercises `create_product_mps`, `mps_norm`, `mps_to_statevector`, and related observable checks.
- **test_mps_c_comparison.mojo** — Loads C reference from `test_data/mps_product_*.json` and compares norms and overlaps with Mojo MPS (observable-based only, not dense tensor element-wise).

---

### m_tensor/

**DenseTensor unit tests.**

- **test_dense_tensor.mojo** — Minimal smoke tests: stride computation, basic creation (GPU if available).
- **test_dense_tensor_basic.mojo** — Strides, reshape, flatten, transpose, norm, scale.
- **test_dense_tensor_dot.mojo** — Dot/contract tests (matrix-matrix, matrix-vector, etc.).
- **test_dense_tensor_qr.mojo** — QR decomposition (including non-2D and wide matrices).
- **test_dense_tensor_svd.mojo** — SVD tests.
- **test_c_replica_dot.mojo** — Replicates C dense_tensor_dot cases: matrix-matrix and matrix-vector multiplication, inner product.

Some tests require a compatible GPU (`has_accelerator()`); they are skipped otherwise.

---

### Root-level tests (under src/tests/)

- **test_svd_lapack.mojo** — SVD/LAPACK: known singular values, reconstruction, U/Vt orthonormality, singular value ordering, chi_max capping, truncation (chi_max, eps).
- **test_utils.mojo** — Shared utilities: tensor comparison, `assert_close`, `assert_equal`, and helpers used by other tests.

---

## Test coverage (overview)

- **Unit tests:** DenseTensor (strides, creation, reshape, transpose, norm, scale, dot, QR, SVD); SVD/LAPACK; MPO construction and operations; MPS product states and observables.
- **Integrity / C comparison:** MPO element-wise vs C, DMRG results vs C, MPS norms/overlaps vs C.
- **Gauge / physical:** DMRG gauge-invariant checks (TFIM); MPO Hermiticity, spectrum, trace.
