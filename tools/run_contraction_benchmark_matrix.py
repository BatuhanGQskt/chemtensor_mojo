#!/usr/bin/env python3
"""
Run the contraction and/or DMRG benchmark pipeline for each (nsites, chi_max) pair in a fixed grid.

For each job (contraction sweep only):
  1. Update Master_Thesis/bench_config.json ("nsites" and "chi_max"; "d" and dmrg_* keys unchanged).
  2. bash run_main.sh in chemtensor/ (build + main + perf_contractions, copies jsonl to Mojo).
  3. mojo run bench_contractions.mojo from the Mojo project root.
  4. python3 tools/merge_and_analyze_benchmarks.py (contraction) then --dmrg.

DMRG-only mode (--dmrg-only): one run; bench_config is not rewritten for contraction sweep.
  1. bash run_main.sh --dmrg-only (main + copy dmrg jsonl; skips perf_contractions).
  2. mojo ... bench_contractions.mojo --dmrg-only
  3. merge_and_analyze_benchmarks.py --dmrg

The default matrix matches the contraction sweep:
  nsites in {2, 4, 6, 8, 10} × chi_max in {16, 32, 64, 128, 256, 512, 1024}.

Original bench_config.json content is restored when the script exits (including on failure).

Examples:
  python3 tools/run_contraction_benchmark_matrix.py --dry-run
  python3 tools/run_contraction_benchmark_matrix.py --continue-on-error
  python3 tools/run_contraction_benchmark_matrix.py --build  # pass --build to run_main.sh
  python3 tools/run_contraction_benchmark_matrix.py --dmrg-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# Number of sites × bond dimension chi_max ("d" stays as in bench_config.json).
NSITES_VALUES = (25, ) #  (2, 4, 6, 8, 10)
CHI_VALUES = (2048, )


def _paths(script: Path) -> tuple[Path, Path, Path, Path]:
    """Mojo repo root, thesis root, chemtensor root, bench_config path."""
    mojo_repo = script.resolve().parent.parent
    thesis = mojo_repo.parent.parent
    chemtensor = thesis / "chemtensor"
    bench = thesis / "bench_config.json"
    return mojo_repo, thesis, chemtensor, bench


def _default_jobs() -> list[tuple[int, int]]:
    return [(ns, chi) for ns in NSITES_VALUES for chi in CHI_VALUES]


def _parse_jobs_file(path: Path) -> list[tuple[int, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("jobs file must be a JSON array of [nsites, chi_max] pairs")
    out: list[tuple[int, int]] = []
    for item in data:
        if (
            isinstance(item, (list, tuple))
            and len(item) == 2
            and all(isinstance(x, int) for x in item)
        ):
            out.append((int(item[0]), int(item[1])))
            continue
        raise ValueError(f"expected [nsites, chi_max] with integers, got {item!r}")
    return out


def _apply_nsites_chi(bench_path: Path, nsites: int, chi_max: int) -> None:
    data = json.loads(bench_path.read_text(encoding="utf-8"))
    data["nsites"] = int(nsites)
    data["chi_max"] = int(chi_max)
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep contraction benchmarks over (nsites, chi_max), and/or run DMRG perf "
            "(bench_config dmrg_* keys)."
        )
    )
    parser.add_argument(
        "--jobs-file",
        type=Path,
        help="JSON array of [nsites, chi_max] pairs, e.g. [[2,16],[4,32]] (ignored with --dmrg-only)",
    )
    parser.add_argument(
        "--dmrg-only",
        action="store_true",
        help=(
            "Single run: skip contraction sweep and perf_contractions; use dmrg_* from bench_config; "
            "merge with --dmrg only."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print steps only; do not write bench_config or run commands.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to the next (nsites, chi_max) if a step fails.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Pass --build to run_main.sh (cmake/make before ./main).",
    )
    parser.add_argument(
        "--mpo-mpo",
        action="store_true",
        help="Pass --mpo-mpo to run_main.sh (very heavy C MPO→dense benchmark; ignored with --dmrg-only).",
    )
    parser.add_argument(
        "--mojo",
        default=os.environ.get("MOJO", "mojo"),
        help="Mojo executable (default: $MOJO or 'mojo').",
    )
    args = parser.parse_args()

    script = Path(__file__).resolve()
    mojo_repo, thesis, chemtensor, bench_path = _paths(script)

    run_main = chemtensor / "run_main.sh"
    bench_mojo = mojo_repo / "src" / "tests" / "benchmarks" / "bench_contractions.mojo"
    merge_py = mojo_repo / "tools" / "merge_and_analyze_benchmarks.py"

    for label, path in (
        ("chemtensor/run_main.sh", run_main),
        ("bench_config.json", bench_path),
        ("bench_contractions.mojo", bench_mojo),
        ("tools/merge_and_analyze_benchmarks.py", merge_py),
    ):
        if not path.is_file():
            print(f"error: missing {label}: {path}", file=sys.stderr)
            return 1

    if args.dmrg_only:
        if args.jobs_file:
            print(
                "warning: --jobs-file ignored with --dmrg-only (single DMRG run)",
                file=sys.stderr,
            )
        jobs: list[tuple[int, int] | None] = [None]
    else:
        jobs = _parse_jobs_file(args.jobs_file) if args.jobs_file else _default_jobs()
    if not jobs:
        print("error: no jobs", file=sys.stderr)
        return 1

    original = bench_path.read_bytes()
    exit_code = 0

    def build_run_main_cmd(*, dmrg_only: bool) -> list[str]:
        cmd = ["bash", str(run_main)]
        if args.build:
            cmd.append("--build")
        if dmrg_only:
            cmd.append("--dmrg-only")
        elif args.mpo_mpo:
            cmd.append("--mpo-mpo")
        return cmd

    mojo_bench_cmd = [
        args.mojo,
        "run",
        "-I",
        ".",
        str(bench_mojo.relative_to(mojo_repo)),
    ]
    if args.dmrg_only:
        mojo_bench_cmd.append("--dmrg-only")
    elif args.mpo_mpo:
        mojo_bench_cmd.append("--mpo-mpo")

    try:
        for i, job in enumerate(jobs, start=1):
            if job is None:
                print(
                    f"\n========== [{i}/{len(jobs)}] DMRG-only (bench_config dmrg_* unchanged) ==========\n",
                    flush=True,
                )
                if args.dry_run:
                    print(
                        "  would not rewrite nsites/chi_max for contraction sweep"
                    )
                    print(f"  would bash {' '.join(build_run_main_cmd(dmrg_only=True))}")
                    print(f"  would {' '.join(mojo_bench_cmd)}")
                    print(
                        f"  would python3 {merge_py} --dmrg"
                    )
                    continue

                steps: list[tuple[str, list[str], Path]] = [
                    (
                        "run_main.sh",
                        build_run_main_cmd(dmrg_only=True),
                        chemtensor,
                    ),
                    ("mojo bench_contractions (DMRG)", mojo_bench_cmd, mojo_repo),
                    (
                        "merge_and_analyze_benchmarks (--dmrg)",
                        [sys.executable, str(merge_py), "--dmrg"],
                        mojo_repo,
                    ),
                ]
            else:
                nsites, chi_max = job
                print(
                    f"\n========== [{i}/{len(jobs)}] nsites={nsites} chi_max={chi_max} ==========\n",
                    flush=True,
                )
                if args.dry_run:
                    print(
                        f"  would set bench_config nsites={nsites} chi_max={chi_max} (d and dmrg_* unchanged)"
                    )
                    print(
                        f"  would bash {' '.join(build_run_main_cmd(dmrg_only=False))}"
                    )
                    print(f"  would {' '.join(mojo_bench_cmd)}")
                    print(f"  would python3 {merge_py}")
                    print(f"  would python3 {merge_py} --dmrg")
                    continue

                _apply_nsites_chi(bench_path, nsites, chi_max)

                steps = [
                    (
                        "run_main.sh",
                        build_run_main_cmd(dmrg_only=False),
                        chemtensor,
                    ),
                    ("mojo bench_contractions", mojo_bench_cmd, mojo_repo),
                    (
                        "merge_and_analyze_benchmarks (contractions)",
                        [sys.executable, str(merge_py)],
                        mojo_repo,
                    ),
                    (
                        "merge_and_analyze_benchmarks (--dmrg)",
                        [sys.executable, str(merge_py), "--dmrg"],
                        mojo_repo,
                    ),
                ]

            step_rc = _run_steps(
                steps=steps,
                continue_on_error=args.continue_on_error,
            )
            if step_rc != 0:
                exit_code = step_rc
                if not args.continue_on_error:
                    return exit_code
    finally:
        if not args.dry_run:
            bench_path.write_bytes(original)
            print("\nRestored original bench_config.json.", flush=True)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
