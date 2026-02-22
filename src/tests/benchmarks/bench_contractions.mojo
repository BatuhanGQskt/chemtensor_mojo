"""
Timed benchmarks for MPO-MPO, MPS-MPO inner product, MPS-MPO apply, and MPS-MPS.
Writes one JSONL record per operation to results/perf/contraction_timings.jsonl
for comparison with C perf_contractions. Scalable via nsites, d, chi_max.
"""

from collections.list import List
from math import sqrt
from time import perf_counter_ns
from gpu.host import DeviceContext
from src.state.mps_state import (
    MatrixProductState,
    MPSSite,
    create_product_mps,
    create_uniform_mps,
    mps_norm,
)
from src.state.mpo_state import MatrixProductOperator, MPOSite
from src.state.hamiltonians import create_ising_1d_mpo
from src.state.environments import (
    expectation_value_two_mps,
    mps_overlap,
    apply_mpo,
)
from src.tests.state.mpo.mpo_test_helpers import mpo_to_full_matrix

# Default parameters (override via constants for scalability; CLI can be added later)
alias DEFAULT_NSITES = 6
alias DEFAULT_D = 2
alias DEFAULT_CHI_MAX = 16
alias NUM_RUNS = 3
alias RESULTS_DIR = "results/perf"
alias RESULTS_FILE = "results/perf/contraction_timings.jsonl"


fn _bond_dims_list(nsites: Int, chi_max: Int) -> List[Int]:
    """Return [1, chi_max, chi_max, ..., chi_max, 1] of length nsites+1."""
    var bonds = List[Int](capacity=nsites + 1)
    bonds.append(1)
    for _ in range(nsites - 1):
        bonds.append(chi_max)
    bonds.append(1)
    return bonds^


fn _append_jsonl(path: String, line: String) raises -> None:
    """Append one line to path (JSONL). Creates parent dir if needed."""
    var f = open(path, "a")
    f.write(line)
    f.write("\n")
    f.close()


fn _write_record(
    path: String,
    operation: String,
    nsites: Int,
    d: Int,
    chi_max: Int,
    time_seconds: Float64,
    result: Float64,
    runs: Int,
) raises -> None:
    var line = String("{\"backend\":\"mojo\",\"operation\":\"")
    line += operation
    line += "\",\"params\":{\"nsites\":"
    line += String(nsites)
    line += ",\"d\":"
    line += String(d)
    line += ",\"chi_max\":"
    line += String(chi_max)
    line += "},\"time_seconds\":"
    line += String(time_seconds)
    line += ",\"result\":"
    line += String(result)
    line += ",\"runs\":"
    line += String(runs)
    line += "}"
    _append_jsonl(path, line)


fn run_bench_contractions(
    nsites: Int = DEFAULT_NSITES,
    d: Int = DEFAULT_D,
    chi_max: Int = DEFAULT_CHI_MAX,
    out_path: String = RESULTS_FILE,
) raises -> None:
    """Run all four contraction benchmarks and append JSONL to out_path."""
    var ctx = DeviceContext()

    # Ising MPO
    var mpo = create_ising_1d_mpo[DType.float32](ctx, num_sites=nsites, J=1.0, h_longitudinal=0.0, g_transverse=0.0)

    # Two MPS with bond dims from chi_max (random for fair comparison with C)
    var bond_dims_psi = _bond_dims_list(nsites, chi_max)
    var bond_dims_chi = _bond_dims_list(nsites, chi_max)
    var psi = create_uniform_mps[DType.float32](ctx, nsites, d, bond_dims_psi^)
    var chi = create_uniform_mps[DType.float32](ctx, nsites, d, bond_dims_chi^)

    # 1) MPO-MPO: full contraction via mpo_to_full_matrix
    var t0 = perf_counter_ns()
    for _ in range(NUM_RUNS):
        var mat = mpo_to_full_matrix(mpo, ctx)
    var t1 = perf_counter_ns()
    var sec_mpo_mpo = Float64(t1 - t0) / 1e9 / NUM_RUNS
    var mat_once = mpo_to_full_matrix(mpo, ctx)
    var nrm_sq = mat_once.norm_sq(ctx)
    var result_mpo_mpo = Float64(0.0)
    if nrm_sq > 0.0:
        result_mpo_mpo = sqrt(nrm_sq)
    print("mpo_mpo: " + String(sec_mpo_mpo) + " s (result=" + String(result_mpo_mpo) + ")")
    _write_record(out_path, "mpo_mpo", nsites, d, chi_max, sec_mpo_mpo, result_mpo_mpo, NUM_RUNS)

    # 2) MPS-MPO inner product <chi|op|psi>
    t0 = perf_counter_ns()
    for _ in range(NUM_RUNS):
        var inner = expectation_value_two_mps[DType.float32](chi, mpo, psi, ctx)
    t1 = perf_counter_ns()
    var sec_inner = Float64(t1 - t0) / 1e9 / NUM_RUNS
    var result_inner = expectation_value_two_mps[DType.float32](chi, mpo, psi, ctx)
    print("mps_mpo_inner: " + String(sec_inner) + " s (result=" + String(result_inner) + ")")
    _write_record(out_path, "mps_mpo_inner", nsites, d, chi_max, sec_inner, result_inner, NUM_RUNS)

    # 3) MPS-MPO apply
    t0 = perf_counter_ns()
    for _ in range(NUM_RUNS):
        var op_psi = apply_mpo[DType.float32](mpo, psi, ctx)
    t1 = perf_counter_ns()
    var sec_apply = Float64(t1 - t0) / 1e9 / NUM_RUNS
    var op_psi = apply_mpo[DType.float32](mpo, psi, ctx)
    var result_apply = mps_norm[DType.float32](op_psi, ctx)
    print("mps_mpo_apply: " + String(sec_apply) + " s (result=" + String(result_apply) + ")")
    _write_record(out_path, "mps_mpo_apply", nsites, d, chi_max, sec_apply, result_apply, NUM_RUNS)

    # 4) MPS-MPS overlap <chi|psi>
    t0 = perf_counter_ns()
    for _ in range(NUM_RUNS):
        var ov = mps_overlap[DType.float32](chi, psi, ctx)
    t1 = perf_counter_ns()
    var sec_mps = Float64(t1 - t0) / 1e9 / NUM_RUNS
    var result_mps = mps_overlap[DType.float32](chi, psi, ctx)
    print("mps_mps: " + String(sec_mps) + " s (result=" + String(result_mps) + ")")
    _write_record(out_path, "mps_mps", nsites, d, chi_max, sec_mps, result_mps, NUM_RUNS)

    print("Appended 4 records to " + out_path)


fn main() raises:
    run_bench_contractions()
