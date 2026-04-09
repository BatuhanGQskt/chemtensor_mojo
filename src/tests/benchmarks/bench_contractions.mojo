"""
Timed benchmarks for MPO-MPO, MPS-MPO inner product, MPS-MPO apply, MPS-MPS, and DMRG.
Writes one JSONL record per operation to results/perf/contraction_timings_{nsites}_{d}_{chi_max}.jsonl
for comparison with C perf_contractions.

Block-sparse tensors (ChemTensor layout) use the same generic ops as dense when you import from
``src.m_tensor.tensor_ops``: build ``C`` with ``allocate_block_sparse_for_tensor_dot``, then
``tensor_dot[DType.float32](mut C, A^, B^, ctx)`` (see ``src/m_tensor/block_sparse_tensor.mojo``).

Reads shared config from ../../bench_config.json (relative to project root).
CLI: pass `--mpo-mpo` to include the full MPO→dense matrix benchmark (very memory-heavy);
by default that step is skipped so large (nsites, d) runs stay tractable.
Pass `--dmrg-only` to run only DMRG benchmarks (reads dmrg_* keys from bench_config.json).

Optional: `mojo run -I . src/tests/benchmarks/bench_contractions.mojo --mpo-mpo`
"""

from collections.list import List
from math import sqrt
from sys import argv
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
from src.state.hamiltonians import create_ising_1d_mpo, create_heisenberg_xxz_mpo
from src.m_tensor.dense_tensor import create_dense_tensor
from src.state.environments import (
    expectation_value_two_mps,
    mps_overlap,
    apply_mpo,
    update_left_environment,
    ProfileStats,
)
from src.tests.state.mpo.mpo_test_helpers import mpo_to_full_matrix
from src.algorithms.dmrg import dmrg_two_site, DMRGParams

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


fn _cli_run_mpo_mpo() -> Bool:
    """True if argv contains --mpo-mpo (opt-in full MPO matrix contraction)."""
    var args = argv()
    var n = len(args)
    var i = 1
    while i < n:
        if args[i] == "--mpo-mpo":
            return True
        i += 1
    return False


fn _cli_dmrg_only() -> Bool:
    """True if argv contains --dmrg-only (skip MPO/MPS contraction timings)."""
    var args = argv()
    var n = len(args)
    var i = 1
    while i < n:
        if args[i] == "--dmrg-only":
            return True
        i += 1
    return False


fn _cli_wants_help() -> Bool:
    var args = argv()
    var n = len(args)
    var i = 1
    while i < n:
        if args[i] == "-h" or args[i] == "--help":
            return True
        i += 1
    return False


fn _print_bench_usage() raises -> None:
    print(
        "bench_contractions: --mpo-mpo enables full MPO→dense contraction"
        " (exact Hilbert space; huge memory). --dmrg-only runs only DMRG (bench_config dmrg_*)."
        " Default runs contractions then DMRG. Flags: -h, --help."
    )


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


fn _parse_float_field(content: String, key: String, default: Float64) -> Float64:
    """Extract a float value for a JSON key from a flat JSON string."""
    var pattern = '"' + key + '"'
    var idx = content.find(pattern)
    if idx == -1:
        return default
    var after_key = idx + len(pattern)
    var i = after_key
    while i < len(content):
        var c = content[i]
        if c == ":" or c == " " or c == "\t" or c == "\n":
            i += 1
            continue
        break
    var num_str = String("")
    var dot_seen = False
    var exp_seen = False
    while i < len(content):
        var c = content[i]
        if c == "+" or c == "-":
            if len(num_str) == 0 or (len(num_str) > 0 and (num_str[len(num_str) - 1] == "e" or num_str[len(num_str) - 1] == "E")):
                num_str += c
                i += 1
                continue
            break
        if c >= "0" and c <= "9":
            num_str += c
            i += 1
            continue
        if c == "." and not dot_seen:
            dot_seen = True
            num_str += c
            i += 1
            continue
        if (c == "e" or c == "E") and not exp_seen:
            exp_seen = True
            num_str += c
            i += 1
            continue
        break
    if len(num_str) == 0:
        return default
    try:
        return Float64(num_str)
    except:
        return default


fn run_bench_contractions(
    config_path: String = DEFAULT_CONFIG_PATH,
    run_mpo_mpo: Bool = False,
) raises -> None:
    """Run contraction benchmarks; MPO→full matrix is optional (see --mpo-mpo)."""
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

    var n_contraction_jsonl = 0

    # 1) MPO-MPO: full contraction via mpo_to_full_matrix (optional)
    if run_mpo_mpo:
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
        n_contraction_jsonl += 1
    else:
        print(
            "mpo_mpo: skipped (pass --mpo-mpo to benchmark full MPO→dense matrix; memory ~ d^(2·nsites))"
        )

    # 2) MPS-MPO inner product <chi|op|psi>
    var t0 = perf_counter_ns()
    for _ in range(num_runs):
        var inner = expectation_value_two_mps[DType.float32](chi, mpo, psi, ctx)
    t1 = perf_counter_ns()
    var sec_inner = Float64(t1 - t0) / 1e9 / num_runs
    # C mpo_inner_product uses conjugated bra and yields opposite sign; negate to match C benchmark.
    var result_inner = -expectation_value_two_mps[DType.float32](chi, mpo, psi, ctx)
    print("mps_mpo_inner: " + String(sec_inner) + " s (result=" + String(result_inner) + ")")
    _write_record(out_path, "mps_mpo_inner", nsites, d, chi_max, sec_inner, result_inner, num_runs)
    n_contraction_jsonl += 1

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
    n_contraction_jsonl += 1

    # 4) MPS-MPS overlap <chi|psi>
    t0 = perf_counter_ns()
    for _ in range(num_runs):
        var ov = mps_overlap[DType.float32](chi, psi, ctx)
    t1 = perf_counter_ns()
    var sec_mps = Float64(t1 - t0) / 1e9 / num_runs
    var result_mps = mps_overlap[DType.float32](chi, psi, ctx)
    print("mps_mps: " + String(sec_mps) + " s (result=" + String(result_mps) + ")")
    _write_record(out_path, "mps_mps", nsites, d, chi_max, sec_mps, result_mps, num_runs)
    n_contraction_jsonl += 1

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
    n_contraction_jsonl += 1

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

    print("Appended " + String(n_contraction_jsonl) + " records to " + out_path)
    print("Appended 2 profiling records to " + profile_out_path)


fn _dmrg_results_path(operation: String, nsites: Int, d: Int, chi_max: Int) -> String:
    """Build parameterized DMRG JSONL output path."""
    return (
        RESULTS_DIR
        + "/"
        + operation
        + "_timings_"
        + String(nsites)
        + "_"
        + String(d)
        + "_"
        + String(chi_max)
        + ".jsonl"
    )


fn _write_dmrg_record(
    path: String,
    operation: String,
    nsites: Int,
    d: Int,
    chi_max: Int,
    num_sweeps: Int,
    maxiter_lanczos: Int,
    J: Float64,
    D: Float64,
    h: Float64,
    tol_split: Float64,
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
    line += ",\"num_sweeps\":"
    line += String(num_sweeps)
    line += ",\"maxiter_lanczos\":"
    line += String(maxiter_lanczos)
    line += ",\"J\":"
    line += String(J)
    line += ",\"D\":"
    line += String(D)
    line += ",\"h\":"
    line += String(h)
    if tol_split >= 0.0:
        line += ",\"tol_split\":"
        line += String(tol_split)
    line += "},\"time_seconds\":"
    line += String(time_seconds)
    line += ",\"result\":"
    line += String(result)
    line += ",\"runs\":"
    line += String(runs)
    line += "}"
    _append_jsonl(path, line)


fn neel_basis(nsites: Int, d: Int) -> List[Int]:
    """Néel product state basis: [0,1,0,1,...]. Has overlap with AFM ground state."""
    var basis = List[Int](capacity=nsites)
    for i in range(nsites):
        basis.append(i % d)
    return basis^


fn run_bench_dmrg(config_path: String = DEFAULT_CONFIG_PATH) raises -> None:
    """Run DMRG benchmarks and append JSONL to parameterized output files.
    
    Reads optional DMRG-specific keys from bench_config.json:
    - dmrg_singlesite_nsites, dmrg_singlesite_d, dmrg_singlesite_chi_max
    - dmrg_singlesite_num_sweeps, dmrg_singlesite_maxiter_lanczos
    - dmrg_singlesite_J, dmrg_singlesite_D, dmrg_singlesite_h
    - dmrg_twosite_nsites, dmrg_twosite_d, dmrg_twosite_chi_max
    - dmrg_twosite_num_sweeps, dmrg_twosite_maxiter_lanczos
    - dmrg_twosite_J, dmrg_twosite_D, dmrg_twosite_h, dmrg_twosite_tol_split
    """
    var cfg = load_bench_config(config_path)
    var ctx = DeviceContext()
    var content = String("")
    try:
        var f = open(config_path, "r")
        content = f.read()
        f.close()
    except:
        content = String("")
    
    print("\n" + "=" * 60)
    print("DMRG Benchmarks")
    print("=" * 60)
    
    # ---- Single-site DMRG proxy (two-site with small chi_max) ----
    # Parameters matching C: nsites=7, d=2, J=1, D=1, h=0, chi_max=16, 6 sweeps
    var nsites_ss = _parse_int_field(content, "dmrg_singlesite_nsites", 7)
    var d_ss = _parse_int_field(content, "dmrg_singlesite_d", 2)
    var J_ss: Float64 = _parse_float_field(content, "dmrg_singlesite_J", 1.0)
    var D_ss: Float64 = _parse_float_field(content, "dmrg_singlesite_D", 1.0)
    var h_ss: Float64 = _parse_float_field(content, "dmrg_singlesite_h", 0.0)
    var chi_max_ss = _parse_int_field(content, "dmrg_singlesite_chi_max", 16)
    var num_sweeps_ss = _parse_int_field(content, "dmrg_singlesite_num_sweeps", 6)
    var maxiter_lanczos_ss = _parse_int_field(content, "dmrg_singlesite_maxiter_lanczos", 25)
    
    print("\n--- Single-site DMRG proxy (Heisenberg XXX, " + String(nsites_ss) + " sites) ---")
    print("Parameters: J=" + String(J_ss) + ", D=" + String(D_ss) + ", h=" + String(h_ss))
    print("chi_max=" + String(chi_max_ss) + ", num_sweeps=" + String(num_sweeps_ss))
    
    var H_ss = create_heisenberg_xxz_mpo[DType.float32](ctx, nsites_ss, J=J_ss, D=D_ss, h=h_ss)
    var basis_ss = neel_basis(nsites_ss, d_ss)
    var psi_ss = create_product_mps[DType.float32](ctx, d_ss, basis_ss^)
    
    var params_ss = DMRGParams(
        num_sweeps=num_sweeps_ss,
        chi_max=chi_max_ss,
        eps_trunc=1e-10,
        max_krylov_iter=maxiter_lanczos_ss,
        krylov_tol=1e-8,
        energy_tol=1e-8,
        two_site=True,
        verbose=False,
    )
    
    ctx.synchronize()
    var t0_ss = perf_counter_ns()
    var result_ss = dmrg_two_site[DType.float32](ctx, H_ss^, psi_ss^, params_ss)
    ctx.synchronize()
    var t1_ss = perf_counter_ns()
    
    var E_ss = result_ss[0]
    var sec_ss = Float64(t1_ss - t0_ss) / 1e9
    
    print("dmrg_singlesite: " + String(sec_ss) + " s (energy=" + String(E_ss) + ")")
    var out_path_ss = _dmrg_results_path("dmrg_singlesite", nsites_ss, d_ss, chi_max_ss)
    _write_dmrg_record(out_path_ss, "dmrg_singlesite", nsites_ss, d_ss, chi_max_ss, num_sweeps_ss,
                       maxiter_lanczos_ss, J_ss, D_ss, h_ss, Float64(-1.0), sec_ss, E_ss, 1)
    
    # ---- Two-site DMRG ----
    # Parameters matching C: nsites=11, d=2, J=1, D=0.5, h=0.2, chi_max=32, 4 sweeps
    var nsites_ts = _parse_int_field(content, "dmrg_twosite_nsites", 11)
    var d_ts = _parse_int_field(content, "dmrg_twosite_d", 2)
    var J_ts: Float64 = _parse_float_field(content, "dmrg_twosite_J", 1.0)
    var D_ts: Float64 = _parse_float_field(content, "dmrg_twosite_D", 0.5)
    var h_ts: Float64 = _parse_float_field(content, "dmrg_twosite_h", 0.2)
    var chi_max_ts = _parse_int_field(content, "dmrg_twosite_chi_max", 32)
    var num_sweeps_ts = _parse_int_field(content, "dmrg_twosite_num_sweeps", 4)
    var maxiter_lanczos_ts = _parse_int_field(content, "dmrg_twosite_maxiter_lanczos", 25)
    var tol_split_ts: Float64 = _parse_float_field(content, "dmrg_twosite_tol_split", 1e-10)
    
    print("\n--- Two-site DMRG (Heisenberg XXZ, " + String(nsites_ts) + " sites) ---")
    print("Parameters: J=" + String(J_ts) + ", D=" + String(D_ts) + ", h=" + String(h_ts))
    print("chi_max=" + String(chi_max_ts) + ", num_sweeps=" + String(num_sweeps_ts))
    
    var H_ts = create_heisenberg_xxz_mpo[DType.float32](ctx, nsites_ts, J=J_ts, D=D_ts, h=h_ts)
    var basis_ts = neel_basis(nsites_ts, d_ts)
    var psi_ts = create_product_mps[DType.float32](ctx, d_ts, basis_ts^)
    
    var params_ts = DMRGParams(
        num_sweeps=num_sweeps_ts,
        chi_max=chi_max_ts,
        eps_trunc=1e-10,
        max_krylov_iter=maxiter_lanczos_ts,
        krylov_tol=1e-8,
        energy_tol=1e-8,
        two_site=True,
        verbose=False,
    )
    
    ctx.synchronize()
    var t0_ts = perf_counter_ns()
    var result_ts = dmrg_two_site[DType.float32](ctx, H_ts^, psi_ts^, params_ts)
    ctx.synchronize()
    var t1_ts = perf_counter_ns()
    
    var E_ts = result_ts[0]
    var sec_ts = Float64(t1_ts - t0_ts) / 1e9
    
    print("dmrg_twosite: " + String(sec_ts) + " s (energy=" + String(E_ts) + ")")
    var out_path_ts = _dmrg_results_path("dmrg_twosite", nsites_ts, d_ts, chi_max_ts)
    _write_dmrg_record(out_path_ts, "dmrg_twosite", nsites_ts, d_ts, chi_max_ts, num_sweeps_ts,
                       maxiter_lanczos_ts, J_ts, D_ts, h_ts, tol_split_ts, sec_ts, E_ts, 1)
    
    print("\nAppended DMRG records to:")
    print("  " + out_path_ss)
    print("  " + out_path_ts)


fn main() raises:
    if _cli_wants_help():
        _print_bench_usage()
        return
    if _cli_dmrg_only():
        run_bench_dmrg()
        return
    var run_mpo = _cli_run_mpo_mpo()
    run_bench_contractions(run_mpo_mpo=run_mpo)
    run_bench_dmrg()
