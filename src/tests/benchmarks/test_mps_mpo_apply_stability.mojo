"""
Stability and plausibility tests for mps_mpo_apply result (norm of MPO|psi>).

Uses the same setup as bench_contractions (C-compatible RNG seeds 42, Ising MPO,
nsites=6, d=2, chi_max=16) and the same norm path: sqrt(mps_overlap(op_psi, op_psi)).

Relation to C vs Mojo result difference:
- C gives a stable norm ~0.051; Mojo had run-to-run variation (~0.026–0.032). Adding
  ctx.synchronize() after apply_mpo and before scalar readback in overlap removes
  async/read-before-done issues, so the Mojo result should become deterministic.
- This test checks determinism (stability over 50 runs). A remaining gap between
  Mojo and C (e.g. 0.03 vs 0.051) would point to precision (float32 vs double) or
  an apply_mpo/overlap math difference, not sync.

- Stability: run apply_mpo -> norm 50 times; assert max(norms) - min(norms) < 1e-6.
- Plausibility: assert norm in [0.01, 0.2] (C reference ~0.051); catches gross bugs.
"""

from collections.list import List
from math import sqrt
from gpu.host import DeviceContext
from src.state.mps_state import MatrixProductState
from src.tests.benchmarks.rng_c_compat import create_random_mps_c_compatible
from src.state.mpo_state import MatrixProductOperator
from src.state.hamiltonians import create_ising_1d_mpo
from src.state.environments import mps_overlap, apply_mpo
from testing import TestSuite


alias NSITES = 6
alias D = 2
alias CHI_MAX = 16
alias NUM_STABILITY_RUNS = 50
alias STABILITY_TOL = 1e-6
# C mps_mpo_apply result for same params is ~0.051; allow generous range for float32
alias NORM_MIN = 0.01
alias NORM_MAX = 0.2


fn _apply_mpo_norm(ctx: DeviceContext, mpo: MatrixProductOperator[DType.float32], psi: MatrixProductState[DType.float32]) raises -> Float64:
    """Norm of apply_mpo(mpo, psi) via sqrt(<op_psi|op_psi>). Same path as bench_contractions."""
    var op_psi = apply_mpo[DType.float32](mpo, psi, ctx)
    return sqrt(mps_overlap[DType.float32](op_psi, op_psi, ctx))


fn test_mps_mpo_apply_stability() raises:
    """Same seed and params, 50 runs: norm of apply_mpo(mpo, psi) must be deterministic (max - min < 1e-6)."""
    var ctx = DeviceContext()
    var mpo = create_ising_1d_mpo[DType.float32](ctx, num_sites=NSITES, J=1.0, h_longitudinal=0.0, g_transverse=0.0)
    var psi = create_random_mps_c_compatible[DType.float32](ctx, NSITES, D, CHI_MAX, 42)

    var norms = List[Float64](capacity=NUM_STABILITY_RUNS)
    for _ in range(NUM_STABILITY_RUNS):
        norms.append(_apply_mpo_norm(ctx, mpo, psi))

    var nmin = norms[0]
    var nmax = norms[0]
    for i in range(1, NUM_STABILITY_RUNS):
        if norms[i] < nmin:
            nmin = norms[i]
        if norms[i] > nmax:
            nmax = norms[i]

    var spread = nmax - nmin
    if spread >= STABILITY_TOL:
        raise Error(
            "mps_mpo_apply norm not stable: max - min = " + String(spread) +
            " (required < " + String(STABILITY_TOL) + "); min = " + String(nmin) + ", max = " + String(nmax)
        )
    print("  ✓ mps_mpo_apply norm stable over " + String(NUM_STABILITY_RUNS) + " runs (spread = " + String(spread) + ")")


fn test_mps_mpo_apply_norm_plausible() raises:
    """Norm of apply_mpo(mpo, psi) for benchmark params should be in [0.01, 0.2] (C reference ~0.051)."""
    var ctx = DeviceContext()
    var mpo = create_ising_1d_mpo[DType.float32](ctx, num_sites=NSITES, J=1.0, h_longitudinal=0.0, g_transverse=0.0)
    var psi = create_random_mps_c_compatible[DType.float32](ctx, NSITES, D, CHI_MAX, 42)
    var n = _apply_mpo_norm(ctx, mpo, psi)

    if n < NORM_MIN or n > NORM_MAX:
        raise Error(
            "mps_mpo_apply norm out of plausible range: " + String(n) +
            " (expected [" + String(NORM_MIN) + ", " + String(NORM_MAX) + "], C reference ~0.051)"
        )
    print("  ✓ mps_mpo_apply norm = " + String(n) + " (plausible)")


fn main() raises:
    try:
        TestSuite.discover_tests[__functions_in_module()]().run()
    except e:
        print("Tests failed: " + String(e))
        raise e
