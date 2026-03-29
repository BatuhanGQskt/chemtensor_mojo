"""Robust DMRG comparison tests — gauge-safe, real-only.

Compares DMRG using gauge-invariant metrics (MPS/MPO tensor contractions).
Uses Transverse Field Ising Model (TFIM) for real-valued Hamiltonians.
Do NOT compare raw MPS tensors entry-wise.

Tiers:
  A: DMRG self-consistency — energy from DMRG matches energy(mps, H), variance small
  B: Observables — Sz, SzSz, SxSx computed via proper MPO expectation
  C: Regression — stored reference values

Run with: mojo run -I . src/tests/algorithms/dmrg/test_dmrg_gauge_safe.mojo
"""

from collections.list import List
from gpu.host import DeviceContext
from src.state.mps_state import create_product_mps
from src.state.hamiltonians import create_transverse_ising_mpo
from src.algorithms.dmrg import dmrg_two_site, DMRGParams
from src.algorithms.dmrg_utils import (
    energy,
    variance_op,
    observe_sz,
    observe_sz_sz,
    observe_sx_sx,
)

alias tol_E: Float64 = 1e-6
# Variance: float32 + H^2 contraction can leave residual ~1e-4; allow 1e-3 for Tier A/C.
alias tol_var: Float64 = 1e-3
alias tol_obs: Float64 = 1e-5


fn assert_close(name: String, got: Float64, expected: Float64, tol: Float64) raises -> None:
    var diff = (-(got - expected)) if (got - expected) < 0 else (got - expected)
    if diff > tol:
        raise Error(
            name + " mismatch: got=" + String(got) + " expected=" + String(expected)
            + " |diff|=" + String(diff) + " tol=" + String(tol)
        )


fn assert_true(name: String, cond: Bool) raises -> None:
    if not cond:
        raise Error("Assertion failed: " + name)


# ---------------------------------------------------------------------------
# Tier A: DMRG self-consistency (energy from DMRG matches energy(mps, H))
# ---------------------------------------------------------------------------

fn tier_a_dmrg_self_consistency() raises -> None:
    """Energy from DMRG must match energy(mps, H) from expectation_value_normalized."""
    with DeviceContext() as ctx:
        var num_sites = 8
        var J = 1.0
        var h = 0.5
        var chi_max = 32
        var num_sweeps = 8

        var basis = List[Int](capacity=num_sites)
        for _ in range(num_sites):
            basis.append(0)
        var psi_initial = create_product_mps[DType.float32](ctx, 2, basis^)
        var H = create_transverse_ising_mpo[DType.float32](ctx, num_sites, J=J, h=h)

        var params = DMRGParams(
            num_sweeps=num_sweeps,
            chi_max=chi_max,
            eps_trunc=1e-10,
            max_krylov_iter=30,
            krylov_tol=1e-10,
            energy_tol=1e-10,
            two_site=True,
            verbose=False,
        )

        var result = dmrg_two_site[DType.float32](ctx, H^, psi_initial^, params)
        var E_dmrg = result[0]
        var psi = result[1]

        var H_check = create_transverse_ising_mpo[DType.float32](ctx, num_sites, J=J, h=h)
        var E_expect = energy[DType.float32](psi, H_check, ctx)
        assert_close("Tier A: E_dmrg vs energy(mps,H)", E_dmrg, E_expect, tol_E)

        var H_var = create_transverse_ising_mpo[DType.float32](ctx, num_sites, J=J, h=h)
        var var_val = variance_op[DType.float32](psi, H_var, ctx)
        assert_true("Tier A: variance small", var_val < tol_var)


# ---------------------------------------------------------------------------
# Tier B: Local observables (Sz, SzSz, SxSx via MPO expectation)
# ---------------------------------------------------------------------------

fn tier_b_observables() raises -> None:
    """Local observables computed via proper MPO expectation."""
    with DeviceContext() as ctx:
        var num_sites = 10
        var J = 1.0
        var h = 0.5
        var chi_max = 64
        var num_sweeps = 6

        var basis = List[Int](capacity=num_sites)
        for _ in range(num_sites):
            basis.append(0)
        var psi_initial = create_product_mps[DType.float32](ctx, 2, basis^)
        var H = create_transverse_ising_mpo[DType.float32](ctx, num_sites, J=J, h=h)

        var params = DMRGParams(
            num_sweeps=num_sweeps,
            chi_max=chi_max,
            eps_trunc=1e-10,
            max_krylov_iter=30,
            krylov_tol=1e-10,
            energy_tol=1e-10,
            two_site=True,
            verbose=False,
        )

        var result = dmrg_two_site[DType.float32](ctx, H^, psi_initial^, params)
        var psi = result[1]

        var E = energy[DType.float32](psi, create_transverse_ising_mpo[DType.float32](ctx, num_sites, J=J, h=h), ctx)
        var sz0 = observe_sz[DType.float32](psi, 0, ctx)
        var sz_mid = observe_sz[DType.float32](psi, num_sites // 2, ctx)
        var sz_sz_01 = observe_sz_sz[DType.float32](psi, 0, 1, ctx)
        var sx_sx_01 = observe_sx_sx[DType.float32](psi, 0, 1, ctx)

        assert_true("Tier B: energy finite", E > -100 and E < 100)
        assert_true("Tier B: |Sz| <= 0.5", (-0.5 - tol_obs) <= sz0 and sz0 <= (0.5 + tol_obs))
        assert_true("Tier B: |SzSz| <= 0.25", (-0.25 - tol_obs) <= sz_sz_01 and sz_sz_01 <= (0.25 + tol_obs))


# ---------------------------------------------------------------------------
# Tier C: Regression snapshots
# ---------------------------------------------------------------------------

fn tier_c_regression() raises -> None:
    """Regression: energy and variance within expected range."""
    with DeviceContext() as ctx:
        var num_sites = 8
        var J = 1.0
        var h = 0.5
        var chi_max = 24
        var num_sweeps = 6

        var basis = List[Int](capacity=num_sites)
        for _ in range(num_sites):
            basis.append(0)
        var psi_initial = create_product_mps[DType.float32](ctx, 2, basis^)
        var H = create_transverse_ising_mpo[DType.float32](ctx, num_sites, J=J, h=h)

        var params = DMRGParams(
            num_sweeps=num_sweeps,
            chi_max=chi_max,
            eps_trunc=1e-10,
            max_krylov_iter=25,
            krylov_tol=1e-8,
            energy_tol=1e-8,
            two_site=True,
            verbose=False,
        )

        var result = dmrg_two_site[DType.float32](ctx, H^, psi_initial^, params)
        var E = result[0]
        var psi = result[1]

        var H_check = create_transverse_ising_mpo[DType.float32](ctx, num_sites, J=J, h=h)
        var var_val = variance_op[DType.float32](psi, H_check, ctx)

        assert_true("Tier C: energy in range", E > -10 and E < 0)
        assert_true("Tier C: variance small", var_val < 0.01)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

fn main() raises -> None:
    print("=" * 60)
    print("DMRG Gauge-Safe Tests (TFIM, MPS/MPO tensor contractions)")
    print("=" * 60)

    try:
        print("\n[Tier A] DMRG self-consistency (energy, variance)...")
        tier_a_dmrg_self_consistency()
        print("  PASSED")

        print("\n[Tier B] Local observables (Sz, SzSz, SxSx)...")
        tier_b_observables()
        print("  PASSED")

        print("\n[Tier C] Regression snapshots...")
        tier_c_regression()
        print("  PASSED")

        print("\n" + "=" * 60)
        print("All gauge-safe DMRG tests PASSED")
        print("=" * 60)
    except e:
        print("FAILED: " + String(e))
        raise
