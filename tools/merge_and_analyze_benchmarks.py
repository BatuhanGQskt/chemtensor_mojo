#!/usr/bin/env python3
"""
Merge one or more contraction (or perf) JSONL files by concatenation, then run
tools/analyze_benchmarks.py on the combined file so C vs Mojo rows appear together.

Latest line wins per (backend, operation) when timestamps are absent — append C
first then Mojo (or the reverse) depending on which “latest run” you want to keep.

Examples:
  # Explicit paths (typical: C build output, then Mojo results):
  python3 tools/merge_and_analyze_benchmarks.py \\
      ../chemtensor/build/generated/perf/contraction_timings_2_2_256.jsonl \\
      results/perf/contraction_timings_2_2_256.jsonl

  # No args: read nsites/d/chi_max from Master_Thesis/bench_config.json (or BENCH_CONFIG)
  # and merge contraction_timings_{nsites}_{d}_{chi_max}.jsonl from C + Mojo dirs:
  python3 tools/merge_and_analyze_benchmarks.py

  # Ignore bench_config; use newest Mojo contraction_timings_*.jsonl + matching C file:
  python3 tools/merge_and_analyze_benchmarks.py --newest-mojo

  # Auto-pick default C + Mojo dirs (Master_Thesis layout) for a given filename:
  python3 tools/merge_and_analyze_benchmarks.py --auto contraction_timings_2_2_256.jsonl

  # Pass flags to the analyzer after -- :
  python3 tools/merge_and_analyze_benchmarks.py --auto contraction_timings_2_2_256.jsonl -- --show-runs

  # DMRG: use bench_config.json dmrg_* keys; merge C+Mojo singlesite/twosite JSONLs, then analyze:
  python3 tools/merge_and_analyze_benchmarks.py --dmrg

  # Override C perf directory (also supports env CHEMTENSOR_C_PERF_DIR):
  python3 tools/merge_and_analyze_benchmarks.py --auto timings.jsonl --c-dir /path/to/c/build/generated/perf
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _repo_root(script: Path) -> Path:
    """chemtensor_mojo project root (directory containing results/, tools/)."""
    return script.resolve().parent.parent


def _default_c_perf_dir(repo: Path) -> Path:
    env = os.environ.get("CHEMTENSOR_C_PERF_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    # Master_Thesis/chemtensor_mojo/chemtensor_mojo -> Master_Thesis/chemtensor/build/generated/perf
    thesis = repo.parent.parent
    return (thesis / "chemtensor" / "build" / "generated" / "perf").resolve()


def _default_mojo_perf_dir(repo: Path) -> Path:
    return (repo / "results" / "perf").resolve()


def _default_bench_config_path(repo: Path) -> Path:
    env = os.environ.get("BENCH_CONFIG", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    thesis = repo.parent.parent
    return (thesis / "bench_config.json").resolve()


def dmrg_basenames_from_bench_config(config_path: Path) -> tuple[str, str] | None:
    """Return (singlesite_basename, twosite_basename) using the same rules as bench_config-driven perf files."""
    if not config_path.is_file():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        ns = int(data.get("nsites", 6))
        d = int(data.get("d", 2))
        chi = int(data.get("chi_max", 16))
        ss_ns = int(data.get("dmrg_singlesite_nsites", ns))
        ss_d = int(data.get("dmrg_singlesite_d", d))
        ss_chi = int(data.get("dmrg_singlesite_chi_max", chi))
        ts_ns = int(data.get("dmrg_twosite_nsites", ns))
        ts_d = int(data.get("dmrg_twosite_d", d))
        ts_chi = int(data.get("dmrg_twosite_chi_max", chi))
    except (TypeError, ValueError):
        return None
    ss = f"dmrg_singlesite_timings_{ss_ns}_{ss_d}_{ss_chi}.jsonl"
    ts = f"dmrg_twosite_timings_{ts_ns}_{ts_d}_{ts_chi}.jsonl"
    return (ss, ts)


def contraction_basename_from_bench_config(config_path: Path) -> str | None:
    """Return contraction_timings_{nsites}_{d}_{chi_max}.jsonl if config has those keys."""
    if not config_path.is_file():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if "nsites" not in data or "d" not in data or "chi_max" not in data:
        return None
    try:
        ns = int(data["nsites"])
        d = int(data["d"])
        chi = int(data["chi_max"])
    except (TypeError, ValueError):
        return None
    return f"contraction_timings_{ns}_{d}_{chi}.jsonl"


def _newest_contraction_basename(mojo_perf_dir: Path) -> str | None:
    """Basename of the newest contraction_timings_*.jsonl under mojo perf (skip _merged_*)."""
    files = [
        p
        for p in mojo_perf_dir.glob("contraction_timings_*.jsonl")
        if p.is_file() and not p.name.startswith("_merged_")
    ]
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0].name


def merge_jsonl(inputs: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as out:
        for path in inputs:
            if not path.is_file():
                raise FileNotFoundError(f"not a file: {path}")
            text = path.read_text(encoding="utf-8")
            if text and not text.endswith("\n"):
                text += "\n"
            out.write(text)


def main() -> int:
    analyze_extra: list[str] = []
    argv = sys.argv[1:]
    if "--" in argv:
        sep = argv.index("--")
        analyze_extra = argv[sep + 1 :]
        argv = argv[:sep]

    parser = argparse.ArgumentParser(
        description="Merge perf JSONL files and run tools/analyze_benchmarks.py"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="JSONL files to concatenate in order (e.g. C jsonl then Mojo jsonl)",
    )
    parser.add_argument(
        "--auto",
        metavar="FILENAME",
        help="Use default C perf dir + results/perf for this basename (e.g. contraction_timings_2_2_256.jsonl)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Merged file path (default: results/perf/_merged_<stem>)",
    )
    parser.add_argument(
        "--c-dir",
        type=Path,
        help="Override C JSONL directory for --auto (default: CHEMTENSOR_C_PERF_DIR or ../chemtensor/build/generated/perf)",
    )
    parser.add_argument(
        "--mojo-dir",
        type=Path,
        help="Override Mojo JSONL directory for --auto (default: results/perf)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="bench_config.json (default: <thesis>/bench_config.json or BENCH_CONFIG env)",
    )
    parser.add_argument(
        "--newest-mojo",
        action="store_true",
        help="Do not read bench_config; pick newest results/perf/contraction_timings_*.jsonl",
    )
    parser.add_argument(
        "--dmrg",
        action="store_true",
        help=(
            "Merge DMRG JSONLs from bench_config (dmrg_singlesite_*, dmrg_twosite_*) "
            "and run analyze_benchmarks.py --dmrg"
        ),
    )
    parser.add_argument(
        "--no-analyze",
        action="store_true",
        help="Only merge; do not run the analyzer",
    )
    args = parser.parse_args(argv)

    script = Path(__file__).resolve()
    repo = _repo_root(script)
    analyzer = script.parent / "analyze_benchmarks.py"
    if not analyzer.is_file():
        print(f"error: analyzer not found at {analyzer}", file=sys.stderr)
        return 1

    inputs: list[Path] = [Path(p).expanduser().resolve() for p in args.inputs]

    auto_name = args.auto
    if args.dmrg and (auto_name or inputs):
        print(
            "error: --dmrg uses bench_config DMRG paths only; "
            "do not pass positional files or --auto",
            file=sys.stderr,
        )
        return 1
    if args.dmrg and args.newest_mojo:
        print("error: --dmrg and --newest-mojo cannot be combined", file=sys.stderr)
        return 1

    if auto_name and inputs:
        print("error: use either positional inputs or --auto, not both", file=sys.stderr)
        return 1

    c_dir = args.c_dir.resolve() if args.c_dir else _default_c_perf_dir(repo)
    m_dir = args.mojo_dir.resolve() if args.mojo_dir else _default_mojo_perf_dir(repo)

    if args.dmrg:
        cfg_path = args.config.resolve() if args.config else _default_bench_config_path(repo)
        pair = dmrg_basenames_from_bench_config(cfg_path)
        if not pair:
            print(
                f"error: could not read DMRG keys from bench config: {cfg_path}\n"
                "  Expected nsites, d, chi_max and/or dmrg_singlesite_* / dmrg_twosite_* integers.",
                file=sys.stderr,
            )
            return 1
        ss_name, ts_name = pair
        print(f"From bench config {cfg_path}:\n  singlesite: {ss_name}\n  twosite:   {ts_name}\n", flush=True)
        c_ss = (c_dir / ss_name).resolve()
        m_ss = (m_dir / ss_name).resolve()
        c_ts = (c_dir / ts_name).resolve()
        m_ts = (m_dir / ts_name).resolve()
        missing: list[str] = []
        for label, p in [
            ("C singlesite", c_ss),
            ("Mojo singlesite", m_ss),
            ("C twosite", c_ts),
            ("Mojo twosite", m_ts),
        ]:
            if not p.is_file():
                missing.append(f"{label}: {p}")
        if missing:
            print("error: missing DMRG JSONL file(s):", file=sys.stderr)
            for line in missing:
                print(f"  {line}", file=sys.stderr)
            return 1

        out_dir = repo / "results" / "perf" / "_dmrg_compare"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_ss = out_dir / ss_name
        out_ts = out_dir / ts_name
        print(
            f"Merging DMRG (C then Mojo lines per file):\n"
            f"  {ss_name}\n"
            f"  {ts_name}"
        )
        try:
            merge_jsonl([c_ss, m_ss], out_ss)
            merge_jsonl([c_ts, m_ts], out_ts)
        except OSError as e:
            print(f"error: merge failed: {e}", file=sys.stderr)
            return 1

        for path, label in [(out_ss, ss_name), (out_ts, ts_name)]:
            with path.open(encoding="utf-8") as mf:
                n = sum(1 for _ in mf)
            print(f"Wrote {label} ({n} lines): {path}", flush=True)

        if args.no_analyze:
            print(
                f"Skipping analysis (--no-analyze). Compare with:\n"
                f"  python3 {analyzer} --dmrg --c-dmrg {out_dir} --mojo-dmrg {out_dir}"
            )
            return 0

        cmd = [
            sys.executable,
            str(analyzer),
            "--dmrg",
            "--c-dmrg",
            str(out_dir),
            "--mojo-dmrg",
            str(out_dir),
            *analyze_extra,
        ]
        print("Running:", " ".join(cmd), flush=True)
        return subprocess.call(cmd)

    if not auto_name and not inputs:
        if not args.newest_mojo:
            cfg_path = args.config.resolve() if args.config else _default_bench_config_path(repo)
            from_cfg = contraction_basename_from_bench_config(cfg_path)
            if from_cfg:
                auto_name = from_cfg
                print(
                    f"From bench config {cfg_path}: nsites/d/chi_max → {auto_name}\n",
                    flush=True,
                )
            elif cfg_path.is_file():
                print(
                    f"warning: {cfg_path} missing nsites/d/chi_max; "
                    "use --newest-mojo or pass files / --auto",
                    file=sys.stderr,
                )
        if not auto_name:
            picked = _newest_contraction_basename(m_dir)
            if not picked:
                print(
                    f"error: no contraction_timings_*.jsonl under {m_dir}\n"
                    "  Run Mojo benchmarks first, or pass files / --auto FILENAME",
                    file=sys.stderr,
                )
                return 1
            auto_name = picked
            print(f"Default: using newest Mojo timings file → {auto_name}\n", flush=True)

    if auto_name:
        if auto_name.startswith("/") or auto_name.startswith(".."):
            print("error: --auto expects a basename only (e.g. contraction_timings_2_2_256.jsonl)", file=sys.stderr)
            return 1
        c_file = (c_dir / auto_name).resolve()
        m_file = (m_dir / auto_name).resolve()
        if not c_file.is_file():
            print(f"error: C file missing: {c_file}", file=sys.stderr)
            return 1
        if not m_file.is_file():
            print(f"error: Mojo file missing: {m_file}", file=sys.stderr)
            return 1
        inputs = [c_file, m_file]
        print(f"Merging:\n  C:    {c_file}\n  Mojo: {m_file}")

    if len(inputs) < 1:
        parser.print_help()
        print("\nerror: pass at least one input jsonl, or use --auto FILENAME", file=sys.stderr)
        return 1

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
    else:
        stem = inputs[0].stem
        out_path = repo / "results" / "perf" / f"_merged_{stem}.jsonl"

    try:
        merge_jsonl(inputs, out_path)
    except OSError as e:
        print(f"error: merge failed: {e}", file=sys.stderr)
        return 1

    with out_path.open(encoding="utf-8") as mf:
        line_count1 = sum(1 for _ in mf)
    print(f"Wrote merged JSONL ({line_count1} lines): {out_path}", flush=True)

    if args.no_analyze:
        print(f"Skipping analysis (--no-analyze). Compare with:\n  python3 {analyzer} --file {out_path}")
        return 0

    cmd = [sys.executable, str(analyzer), "--file", str(out_path), *analyze_extra]
    print("Running:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
