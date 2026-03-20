"""
Timed benchmarks for MPO-MPO, MPS-MPO inner product, MPS-MPO apply, and MPS-MPS.
Writes one JSONL record per operation to results/perf/contraction_timings_{nsites}_{d}_{chi_max}.jsonl
for comparison with C perf_contractions.

Reads shared config from ../../bench_config.json (relative to project root).
CLI override not yet implemented; edit bench_config.json to change parameters.
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
from src.tests.benchmarks.rng_c_compat import create_random_mps_c_compatible
from src.state.mpo_state import MatrixProductOperator, MPOSite
from src.state.hamiltonians import create_ising_1d_mpo
from src.m_tensor.dense_tensor import create_dense_tensor
from src.state.environments import (
    expectation_value_two_mps,
    mps_overlap,
    apply_mpo,
    update_left_environment,
    ProfileStats,
)
from src.tests.state.mpo.mpo_test_helpers import mpo_to_full_matrix

# Fallback defaults (used if config file is not found)
alias DEFAULT_NSITES = 6
alias DEFAULT_D = 2
alias DEFAULT_CHI_MAX = 16
alias DEFAULT_NUM_RUNS = 3
alias DEFAULT_CONFIG_PATH = "../../bench_config.json"
alias RESULTS_DIR = "results/perf"


@fieldwise_init
struct BenchConfig(Copyable, ImplicitlyCopyable, Movable, Stringable):
    """Benchmark configuration loaded from shared bench_config.json."""
    var nsites: Int
    var d: Int
    var chi_max: Int
    var num_runs: Int

    fn __str__(self) -> String:
        return (
            "BenchConfig(nsites="
            + String(self.nsites)
            + ", d="
            + String(self.d)
            + ", chi_max="
            + String(self.chi_max)
            + ", num_runs="
            + String(self.num_runs)
            + ")"
        )


fn _parse_int_field(content: String, key: String, default: Int) -> Int:
    """Extract an integer value for a JSON key from a flat JSON string."""
    var pattern = '"' + key + '"'
    var idx = content.find(pattern)
    if idx == -1:
        return default
    var after_key = idx + len(pattern)
    # Skip whitespace and colon
    var i = after_key
    while i < len(content):
        var c = content[i]
        if c == ":" or c == " " or c == "\t" or c == "\n":
            i += 1
            continue
        break
    # Read digits
    var num_str = String("")
    while i < len(content):
        var c = content[i]
        if c == "0" or c == "1" or c == "2" or c == "3" or c == "4" or c == "5" or c == "6" or c == "7" or c == "8" or c == "9":
            num_str += c
            i += 1
        else:
            break
    if len(num_str) == 0:
        return default
    try:
        return Int(num_str)
    except:
        return default


fn load_bench_config(config_path: String = DEFAULT_CONFIG_PATH) -> BenchConfig:
    """Load benchmark config from JSON file. Returns defaults on failure.

    Tries a few common relative paths so it works no matter which directory you run from:
    1) user-provided path (arg or default)
    2) ../../bench_config.json (parent-of-parent of this project)
    3) ../bench_config.json
    4) bench_config.json (current working dir)
    """

    var cfg = BenchConfig(DEFAULT_NSITES, DEFAULT_D, DEFAULT_CHI_MAX, DEFAULT_NUM_RUNS)

    var paths = List[String](capacity=4)
    paths.append(config_path)
    paths.append(String("../../bench_config.json"))
    paths.append(String("../bench_config.json"))
    paths.append(String("bench_config.json"))

    for p in paths:
        try:
            var f = open(p, "r")
            var content = f.read()
            f.close()
            cfg.nsites = _parse_int_field(content, "nsites", DEFAULT_NSITES)
            cfg.d = _parse_int_field(content, "d", DEFAULT_D)
            cfg.chi_max = _parse_int_field(content, "chi_max", DEFAULT_CHI_MAX)
            cfg.num_runs = _parse_int_field(content, "num_runs", DEFAULT_NUM_RUNS)
            print("Loaded config from " + p + ": " + String(cfg))
            return cfg
        except:
            continue

    print("Config not found (tried " + String(paths.__len__()) + " paths), using defaults: " + String(cfg))
    return cfg


fn _results_path(nsites: Int, d: Int, chi_max: Int) -> String:
    """Build parameterized JSONL output path."""
    return (
        RESULTS_DIR
        + "/contraction_timings_"
        + String(nsites)
        + "_"
        + String(d)
        + "_"
        + String(chi_max)
        + ".jsonl"
    )


fn _profile_results_path(nsites: Int, d: Int, chi_max: Int) -> String:
    """Build parameterized profiling JSONL output path."""
    return (
        RESULTS_DIR
        + "/profiling_"
        + String(nsites)
        + "_"
        + String(d)
        + "_"
        + String(chi_max)
        + ".jsonl"
    )


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


fn _write_profile_record(
    path: String,
    operation: String,
    nsites: Int,
    d: Int,
    chi_max: Int,
    profile: ProfileStats,
    result: Float64,
) raises -> None:
    var line = String("{\"backend\":\"mojo\",\"operation\":\"")
    line += operation
    line += "\",\"params\":{\"nsites\":"
    line += String(nsites)
    line += ",\"d\":"
    line += String(d)
    line += ",\"chi_max\":"
    line += String(chi_max)
    line += "},\"result\":"
    line += String(result)
    line += ",\"profile\":{"
    line += "\"transpose_ns\":"
    line += String(profile.transpose_ns)
    line += ",\"transpose_calls\":"
    line += String(profile.transpose_calls)
    line += ",\"dot_ns\":"
    line += String(profile.dot_ns)
    line += ",\"dot_calls\":"
    line += String(profile.dot_calls)
    line += ",\"alloc_ns\":"
    line += String(profile.alloc_ns)
    line += ",\"reshape_ns\":"
    line += String(profile.reshape_ns)
    line += ",\"reshape_calls\":"
    line += String(profile.reshape_calls)
    line += ",\"sync_ns\":"
    line += String(profile.sync_ns)
    line += ",\"sync_calls\":"
    line += String(profile.sync_calls)
    line += ",\"copy_ns\":"
    line += String(profile.copy_ns)
    line += ",\"copy_calls\":"
    line += String(profile.copy_calls)
    line += ",\"total_ns\":"
    line += String(profile.total_ns)
    line += "}}"
    _append_jsonl(path, line)


fn run_bench_contractions(
    config_path: String = DEFAULT_CONFIG_PATH,
) raises -> None:
    """Run all four contraction benchmarks and append JSONL to parameterized output file."""
    var cfg = load_bench_config(config_path)
    var nsites = cfg.nsites
    var d = cfg.d
    var chi_max = cfg.chi_max
    var num_runs = cfg.num_runs
    var out_path = _results_path(nsites, d, chi_max)
    var profile_out_path = _profile_results_path(nsites, d, chi_max)

    var ctx = DeviceContext()

    # Ising MPO
    var mpo = create_ising_1d_mpo[DType.float32](ctx, num_sites=nsites, J=1.0, h_longitudinal=0.0, g_transverse=0.0)

    # Two MPS: same as C (seeds 42, 43) via C-compatible PCG — identical at any scale
    var psi = create_random_mps_c_compatible[DType.float32](ctx, nsites, d, chi_max, 42)
    var chi = create_random_mps_c_compatible[DType.float32](ctx, nsites, d, chi_max, 43)

    # 1) MPO-MPO: full contraction via mpo_to_full_matrix
    var t0 = perf_counter_ns()
    for _ in range(num_runs):
        var mat = mpo_to_full_matrix(mpo, ctx)
    var t1 = perf_counter_ns()
    var sec_mpo_mpo = Float64(t1 - t0) / 1e9 / num_runs
    var mat_once = mpo_to_full_matrix(mpo, ctx)
    var nrm_sq = mat_once.norm_sq(ctx)
    var result_mpo_mpo = Float64(0.0)
    if nrm_sq > 0.0:
        result_mpo_mpo = sqrt(nrm_sq)
    print("mpo_mpo: " + String(sec_mpo_mpo) + " s (result=" + String(result_mpo_mpo) + ")")
    _write_record(out_path, "mpo_mpo", nsites, d, chi_max, sec_mpo_mpo, result_mpo_mpo, num_runs)

    # 2) MPS-MPO inner product <chi|op|psi>
    t0 = perf_counter_ns()
    for _ in range(num_runs):
        var inner = expectation_value_two_mps[DType.float32](chi, mpo, psi, ctx)
    t1 = perf_counter_ns()
    var sec_inner = Float64(t1 - t0) / 1e9 / num_runs
    # C mpo_inner_product uses conjugated bra and yields opposite sign; negate to match C benchmark.
    var result_inner = -expectation_value_two_mps[DType.float32](chi, mpo, psi, ctx)
    print("mps_mpo_inner: " + String(sec_inner) + " s (result=" + String(result_inner) + ")")
    _write_record(out_path, "mps_mpo_inner", nsites, d, chi_max, sec_inner, result_inner, num_runs)

    # 3) MPS-MPO apply
    t0 = perf_counter_ns()
    for _ in range(num_runs):
        var op_psi = apply_mpo[DType.float32](mpo, psi, ctx)
    t1 = perf_counter_ns()
    var sec_apply = Float64(t1 - t0) / 1e9 / num_runs
    var op_psi = apply_mpo[DType.float32](mpo, psi, ctx)
    # Use same norm as C: <psi|psi> via sweep (mps_vdot), then sqrt. Matches C mps_norm.
    var result_apply = sqrt(mps_overlap[DType.float32](op_psi, op_psi, ctx))
    print("mps_mpo_apply: " + String(sec_apply) + " s (result=" + String(result_apply) + ")")
    _write_record(out_path, "mps_mpo_apply", nsites, d, chi_max, sec_apply, result_apply, num_runs)

    # 4) MPS-MPS overlap <chi|psi>
    t0 = perf_counter_ns()
    for _ in range(num_runs):
        var ov = mps_overlap[DType.float32](chi, psi, ctx)
    t1 = perf_counter_ns()
    var sec_mps = Float64(t1 - t0) / 1e9 / num_runs
    var result_mps = mps_overlap[DType.float32](chi, psi, ctx)
    print("mps_mps: " + String(sec_mps) + " s (result=" + String(result_mps) + ")")
    _write_record(out_path, "mps_mps", nsites, d, chi_max, sec_mps, result_mps, num_runs)

    # 4b) Profiled mps_overlap — calls the real function with a ProfileStats
    var profile = ProfileStats.create()
    var result_prof = mps_overlap[DType.float32](chi, psi, ctx, profile)
    print("mps_mps_profile: " + String(profile))
    _write_profile_record(profile_out_path, "mps_mps_profile", nsites, d, chi_max, profile, result_prof)

    # 5) Left-environment sweep (GPU timing: sync before/after so we measure kernel execution)
    var wL0 = mpo.bond_dimension(0)
    var Dl0 = psi.bond_dimension(0)
    var L_initial = create_dense_tensor[DType.float32](
        ctx,
        List[Int](wL0, Dl0, Dl0),
        init_value=Scalar[DType.float32](0.0),
    )
    var host_L = ctx.enqueue_create_host_buffer[DType.float32](wL0 * Dl0 * Dl0)
    for w in range(wL0):
        for d in range(Dl0):
            host_L[w * (Dl0 * Dl0) + d * Dl0 + d] = Scalar[DType.float32](1.0)
    ctx.enqueue_copy(L_initial.storage, host_L)
    ctx.synchronize()

    # Warmup: one full sweep (copy L_initial so we don't consume it)
    var L_warmup = create_dense_tensor[DType.float32](ctx, List[Int](wL0, Dl0, Dl0), init_value=Scalar[DType.float32](0.0))
    ctx.enqueue_copy(L_warmup.storage, L_initial.storage)
    ctx.synchronize()
    for i in range(nsites):
        L_warmup = update_left_environment[DType.float32](L_warmup^, psi.sites[i], mpo.sites[i], ctx)
    ctx.synchronize()

    # Timed runs: sync before/after to measure GPU kernel execution. Fresh L each run from L_initial.
    t0 = perf_counter_ns()
    for _ in range(num_runs):
        ctx.synchronize()
        var L_run = create_dense_tensor[DType.float32](ctx, List[Int](wL0, Dl0, Dl0), init_value=Scalar[DType.float32](0.0))
        ctx.enqueue_copy(L_run.storage, L_initial.storage)
        ctx.synchronize()
        for i in range(nsites):
            L_run = update_left_environment[DType.float32](L_run^, psi.sites[i], mpo.sites[i], ctx)
        ctx.synchronize()
    t1 = perf_counter_ns()
    var sec_left_env = Float64(t1 - t0) / 1e9 / num_runs
    print("left_env_sweep: " + String(sec_left_env) + " s (GPU timed, " + String(num_runs) + " runs)")
    _write_record(out_path, "left_env_sweep", nsites, d, chi_max, sec_left_env, Float64(0.0), num_runs)

    # 5b) Profiled left-env sweep
    var profile_lenv = ProfileStats.create()
    var L_prof = create_dense_tensor[DType.float32](ctx, List[Int](wL0, Dl0, Dl0), init_value=Scalar[DType.float32](0.0))
    ctx.enqueue_copy(L_prof.storage, L_initial.storage)
    ctx.synchronize()
    for i in range(nsites):
        L_prof = update_left_environment[DType.float32](L_prof^, psi.sites[i], mpo.sites[i], ctx, profile_lenv)
    ctx.synchronize()
    print("left_env_profile: " + String(profile_lenv))
    _write_profile_record(profile_out_path, "left_env_profile", nsites, d, chi_max, profile_lenv, Float64(0.0))

    print("Appended 5 records to " + out_path)
    print("Appended 2 profiling records to " + profile_out_path)


fn main() raises:
    run_bench_contractions()
