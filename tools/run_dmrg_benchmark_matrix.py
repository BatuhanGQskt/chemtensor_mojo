#!/usr/bin/env python3
"""
Sweep DMRG benchmarks over a grid of (nsites, chi_max, num_sweeps).

For each job:
  1. Update Master_Thesis/dmrg_config.json (dmrg_*_nsites, dmrg_*_chi_max,
     dmrg_*_num_sweeps for the selected variant(s); all other keys unchanged).
  2. Clear existing C and Mojo JSONL files for the configuration.
  3. Run ./main N times (--repeat N) to collect C timing samples.
  4. Run mojo bench_contractions.mojo --dmrg-only N times for Mojo samples.
  5. Merge C source + Mojo results with merge_and_analyze_benchmarks.py --dmrg.

Original dmrg_config.json content is restored when the script exits (including on failure).

Variants:
  --variant singlesite   sweep dmrg_singlesite_* keys only
  --variant twosite      sweep dmrg_twosite_* keys only
  --variant both         sweep both sets of keys together (default)

Default sweep grid:
  nsites     in {4, 6, 8, 10, 12}
  chi_max    in {64, 128, 256, 512, 1024}
  num_sweeps in {5, 10, 15, 20}

Examples:
  python3 tools/run_dmrg_benchmark_matrix.py --dry-run
  python3 tools/run_dmrg_benchmark_matrix.py --variant singlesite
  python3 tools/run_dmrg_benchmark_matrix.py --nsites 4 6 8 --chi-max 128 256 --num-sweeps 10 20
  python3 tools/run_dmrg_benchmark_matrix.py --repeat 3
  python3 tools/run_dmrg_benchmark_matrix.py --jobs-file dmrg_jobs.json
  python3 tools/run_dmrg_benchmark_matrix.py --continue-on-error --build   # builds once, then reuses binary
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Default sweep grid - DMRG is expensive so keep ranges conservative.
DEFAULT_NSITES_VALUES    = (12, )
DEFAULT_CHI_VALUES       = (1024, ) #(64, 128, 256, 512, 1024, 2048) #(64, 128, 256, 512, 1024)
DEFAULT_NUM_SWEEPS_VALUES = (5, 10, 15, 20)

VARIANTS = ("singlesite", "twosite", "both")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _paths(script: Path) -> tuple[Path, Path, Path, Path]:
    """Return (mojo_repo, thesis, chemtensor, dmrg_config) paths."""
    mojo_repo  = script.resolve().parent.parent
    thesis     = mojo_repo.parent.parent
    chemtensor = thesis / "chemtensor"
    bench      = thesis / "dmrg_config.json"
    return mojo_repo, thesis, chemtensor, bench


def _default_jobs(
    nsites_values: tuple[int, ...],
    chi_values: tuple[int, ...],
    num_sweeps_values: tuple[int, ...],
) -> list[tuple[int, int, int]]:
    return [
        (ns, chi, sw)
        for ns  in nsites_values
        for chi in chi_values
        for sw  in num_sweeps_values
    ]


def _parse_jobs_file(path: Path) -> list[tuple[int, int, int]]:
    """Parse a JSON array of [nsites, chi_max, num_sweeps] triples."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("jobs file must be a JSON array of [nsites, chi_max, num_sweeps] triples")
    out: list[tuple[int, int, int]] = []
    for item in data:
        if (
            isinstance(item, (list, tuple))
            and len(item) == 3
            and all(isinstance(x, int) for x in item)
        ):
            out.append((int(item[0]), int(item[1]), int(item[2])))
            continue
        raise ValueError(
            f"expected [nsites, chi_max, num_sweeps] with integers, got {item!r}"
        )
    return out


def _apply_dmrg_params(
    bench_path: Path,
    nsites: int,
    chi_max: int,
    num_sweeps: int,
    variant: str,
) -> None:
    """Rewrite dmrg_config.json with the given DMRG parameters."""
    data = json.loads(bench_path.read_text(encoding="utf-8"))
    prefixes = []
    if variant in ("singlesite", "both"):
        prefixes.append("dmrg_singlesite_")
    if variant in ("twosite", "both"):
        prefixes.append("dmrg_twosite_")
    for prefix in prefixes:
        data[f"{prefix}nsites"]     = int(nsites)
        data[f"{prefix}chi_max"]    = int(chi_max)
        data[f"{prefix}num_sweeps"] = int(num_sweeps)
    bench_path.write_text(
        json.dumps(data, indent=4) + "\n",
        encoding="utf-8",
    )


def _run_steps(
    *,
    steps: list[tuple[str, list[str], Path]],
    continue_on_error: bool,
) -> int:
    exit_code = 0
    for name, cmd, cwd in steps:
        print(f"--> {name}: {' '.join(cmd)}", flush=True)
        try:
            r = subprocess.run(cmd, cwd=cwd, check=False)
        except OSError as e:
            print(f"error: failed to run {name}: {e}", file=sys.stderr)
            exit_code = 1
            if not continue_on_error:
                return exit_code
            break
        if r.returncode != 0:
            print(
                f"error: {name} exited with {r.returncode}",
                file=sys.stderr,
            )
            exit_code = 1
            if not continue_on_error:
                return r.returncode
            break
    return exit_code


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep DMRG benchmarks over a 3-D grid of "
            "(nsites, chi_max, num_sweeps)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- sweep dimensions ---
    parser.add_argument(
        "--nsites",
        type=int,
        nargs="+",
        metavar="N",
        help=(
            f"nsites values to sweep (default: {list(DEFAULT_NSITES_VALUES)}). "
            "Ignored when --jobs-file is given."
        ),
    )
    parser.add_argument(
        "--chi-max",
        type=int,
        nargs="+",
        metavar="CHI",
        help=(
            f"chi_max values to sweep (default: {list(DEFAULT_CHI_VALUES)}). "
            "Ignored when --jobs-file is given."
        ),
    )
    parser.add_argument(
        "--num-sweeps",
        type=int,
        nargs="+",
        metavar="SW",
        help=(
            f"num_sweeps values to sweep (default: {list(DEFAULT_NUM_SWEEPS_VALUES)}). "
            "Ignored when --jobs-file is given."
        ),
    )

    # --- job list ---
    parser.add_argument(
        "--jobs-file",
        type=Path,
        help=(
            "JSON file containing a list of [nsites, chi_max, num_sweeps] triples, "
            "e.g. [[4,128,10],[8,256,15]]. Overrides --nsites/--chi-max/--num-sweeps."
        ),
    )

    # --- DMRG variant ---
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        default="both",
        help=(
            "Which DMRG variant's config keys to update: "
            "'singlesite', 'twosite', or 'both' (default: both)."
        ),
    )

    # --- run-time flags ---
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print steps only; do not write dmrg_config or execute commands.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to the next job if any step fails.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Force rebuild of C binary (cmake/make) before first job.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help="Repeat each benchmark job N times (default: 1).",
    )
    parser.add_argument(
        "--mojo",
        default=os.environ.get("MOJO", "mojo"),
        help="Mojo executable (default: $MOJO or 'mojo').",
    )

    args = parser.parse_args()
    if args.repeat < 1:
        print("error: --repeat must be >= 1", file=sys.stderr)
        return 1

    # --- resolve paths ---
    script = Path(__file__).resolve()
    mojo_repo, thesis, chemtensor, bench_path = _paths(script)

    bench_mojo = mojo_repo / "src" / "tests" / "benchmarks" / "bench_contractions.mojo"
    merge_py  = mojo_repo / "tools" / "merge_and_analyze_benchmarks.py"

    for label, path in (
        ("dmrg_config.json",                    bench_path),
        ("bench_contractions.mojo",             bench_mojo),
        ("tools/merge_and_analyze_benchmarks.py", merge_py),
    ):
        if not path.is_file():
            print(f"error: missing {label}: {path}", file=sys.stderr)
            return 1

    # --- build job list ---
    if args.jobs_file:
        jobs = _parse_jobs_file(args.jobs_file)
    else:
        nsites_values     = tuple(args.nsites)     if args.nsites     else DEFAULT_NSITES_VALUES
        chi_values        = tuple(args.chi_max)    if args.chi_max    else DEFAULT_CHI_VALUES
        num_sweeps_values = tuple(args.num_sweeps) if args.num_sweeps else DEFAULT_NUM_SWEEPS_VALUES
        jobs = _default_jobs(nsites_values, chi_values, num_sweeps_values)

    if not jobs:
        print("error: no jobs", file=sys.stderr)
        return 1

    total = len(jobs)
    print(
        f"DMRG benchmark sweep: {total} job(s) | variant={args.variant} | repeat={args.repeat}",
        flush=True,
    )

    # --- fixed mojo command (--dmrg-only always set) ---
    mojo_bench_cmd = [
        args.mojo,
        "run",
        "-I",
        ".",
        str(bench_mojo.relative_to(mojo_repo)),
        "--dmrg-only",
    ]

    original = bench_path.read_bytes()
    exit_code = 0

    try:
        for i, (nsites, chi_max, num_sweeps) in enumerate(jobs, start=1):
            first_job = i == 1
            print(
                f"\n========== [{i}/{total}] nsites={nsites}  chi_max={chi_max}"
                f"  num_sweeps={num_sweeps}  variant={args.variant} ==========\n",
                flush=True,
            )

            if args.dry_run:
                print(
                    f"  would set dmrg_config "
                    f"dmrg_*_nsites={nsites}  dmrg_*_chi_max={chi_max}"
                    f"  dmrg_*_num_sweeps={num_sweeps}  (variant={args.variant})"
                )
                print("  would clear C source and Mojo JSONL files for this configuration")
                if first_job:
                    print("  would build C binary if needed (cmake + make)")
                for rep in range(1, args.repeat + 1):
                    print(f"  C backend repetition {rep}/{args.repeat}:")
                    print(f"    would run ./main directly in chemtensor/build/")
                for rep in range(1, args.repeat + 1):
                    print(f"  Mojo backend repetition {rep}/{args.repeat}:")
                    print(f"    would {' '.join(mojo_bench_cmd)}")
                print(f"  would merge: python3 {merge_py} --dmrg")
                continue

            _apply_dmrg_params(bench_path, nsites, chi_max, num_sweeps, args.variant)

            # Clear C JSONL files before running repetitions to avoid duplicates.
            # Note: We clear only the C source files; the merge will read from C source directly.
            c_perf_dir = chemtensor / "build" / "generated" / "perf"
            mojo_perf_dir = mojo_repo / "results" / "perf"
            jsonl_basenames: list[str] = []
            if args.variant in ("singlesite", "both"):
                jsonl_basenames.append(
                    f"dmrg_singlesite_timings_{nsites}_2_{chi_max}_{num_sweeps}.jsonl"
                )
            if args.variant in ("twosite", "both"):
                jsonl_basenames.append(
                    f"dmrg_twosite_timings_{nsites}_2_{chi_max}_{num_sweeps}.jsonl"
                )

            # Clear C source files (avoid accumulated entries)
            for basename in jsonl_basenames:
                c_jsonl = c_perf_dir / basename
                if c_jsonl.is_file():
                    print(f"  Clearing C source: {c_jsonl}", flush=True)
                    c_jsonl.unlink()

            # Clear Mojo results/perf files (avoid old accumulated C entries from run_main.sh)
            for basename in jsonl_basenames:
                mojo_jsonl = mojo_perf_dir / basename
                if mojo_jsonl.is_file():
                    print(f"  Clearing Mojo results: {mojo_jsonl}", flush=True)
                    mojo_jsonl.unlink()

            # Run all C repetitions first.
            # We run ./main directly (not run_main.sh) to avoid its append-to-Mojo behavior.
            # run_main.sh appends C results to Mojo results/perf which causes duplication.
            c_main_binary = chemtensor / "build" / "main"
            need_build = not c_main_binary.is_file() or (args.build and first_job)
            if need_build:
                print("  Building C binary (cmake + make)...", flush=True)
                build_step: list[tuple[str, list[str], Path]] = [
                    ("cmake + make", ["bash", "-c", "mkdir -p build && cd build && cmake .. && make"], chemtensor),
                ]
                step_rc = _run_steps(steps=build_step, continue_on_error=args.continue_on_error)
                if step_rc != 0:
                    exit_code = step_rc
                    if not args.continue_on_error:
                        return exit_code

            for rep in range(1, args.repeat + 1):
                print(
                    f"--- C backend repetition {rep}/{args.repeat} for job [{i}/{total}] ---",
                    flush=True,
                )
                # Run ./main directly from build dir (it reads dmrg_config.json and writes to generated/perf/)
                c_step: list[tuple[str, list[str], Path]] = [
                    (
                        "./main (C DMRG)",
                        ["./main"],
                        chemtensor / "build",
                    ),
                ]
                step_rc = _run_steps(steps=c_step, continue_on_error=args.continue_on_error)
                if step_rc != 0:
                    exit_code = step_rc
                    if not args.continue_on_error:
                        return exit_code

            # Run all Mojo repetitions
            for rep in range(1, args.repeat + 1):
                print(
                    f"--- Mojo backend repetition {rep}/{args.repeat} for job [{i}/{total}] ---",
                    flush=True,
                )
                mojo_step: list[tuple[str, list[str], Path]] = [
                    (
                        "mojo bench_contractions (--dmrg-only)",
                        mojo_bench_cmd,
                        mojo_repo,
                    ),
                ]
                step_rc = _run_steps(steps=mojo_step, continue_on_error=args.continue_on_error)
                if step_rc != 0:
                    exit_code = step_rc
                    if not args.continue_on_error:
                        return exit_code

            # Merge once at the end (after all repetitions)
            print(
                f"--- Merging results for job [{i}/{total}] ---",
                flush=True,
            )
            merge_step: list[tuple[str, list[str], Path]] = [
                (
                    "merge_and_analyze_benchmarks (--dmrg)",
                    [sys.executable, str(merge_py), "--dmrg"],
                    mojo_repo,
                ),
            ]
            step_rc = _run_steps(steps=merge_step, continue_on_error=args.continue_on_error)
            if step_rc != 0:
                exit_code = step_rc
                if not args.continue_on_error:
                    return exit_code

    finally:
        if not args.dry_run:
            bench_path.write_bytes(original)
            print("\nRestored original dmrg_config.json.", flush=True)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
