#!/usr/bin/env python3
"""
Benchmark Analysis: C vs Mojo Performance Comparison

This script analyzes and compares the performance of tensor contraction operations
and DMRG algorithms between the C (CPU) and Mojo (GPU) implementations of chemtensor.

Usage:
    python analyze_benchmarks.py [--file FILE] [--all] [--dmrg]
    
Options:
    --file FILE    Analyze a specific JSONL file
    --all          Analyze all contraction_timings_*.jsonl files
    --latest       Use only the latest run for each backend/operation (default)
    --dmrg         Analyze DMRG benchmark results
"""

import json
import glob
import os
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
import math
import re


def parse_contraction_timings_filename(filename: str) -> Optional[tuple[int, int, int]]:
    """Read nsites, d, chi_max from *contraction_timings_{nsites}_{d}_{chi_max}.jsonl (any prefix e.g. _merged_)."""
    m = re.search(r"contraction_timings_(\d+)_(\d+)_(\d+)\.jsonl$", filename)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


@dataclass
class BenchmarkResult:
    backend: str
    operation: str
    time_seconds: float
    result: float
    runs: int
    params: dict
    timestamp: Optional[int] = None
    
    @property
    def time_ms(self) -> float:
        return self.time_seconds * 1000
    
    @property
    def time_us(self) -> float:
        return self.time_seconds * 1e6


def load_jsonl(filepath: str) -> list[dict]:
    """Load all records from a JSONL file."""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def parse_results(records: list[dict]) -> dict[str, dict[str, list[BenchmarkResult]]]:
    """Parse records into structured results by backend and operation."""
    results = defaultdict(lambda: defaultdict(list))
    
    for rec in records:
        br = BenchmarkResult(
            backend=rec.get('backend', 'unknown'),
            operation=rec.get('operation', 'unknown'),
            time_seconds=rec.get('time_seconds', 0.0),
            result=rec.get('result', float('nan')),
            runs=rec.get('runs', 1),
            params=rec.get('params', {}),
            timestamp=rec.get('timestamp')
        )
        results[br.backend][br.operation].append(br)
    
    return results


def get_latest_results(results: dict) -> dict[str, dict[str, BenchmarkResult]]:
    """Get latest result per backend/operation.
    
    Prefer the highest timestamp if available; fall back to last-loaded record.
    """
    latest = {}
    for backend, ops in results.items():
        latest[backend] = {}
        for op, runs in ops.items():
            if runs:
                with_ts = [r for r in runs if r.timestamp is not None]
                if with_ts:
                    latest[backend][op] = max(with_ts, key=lambda r: r.timestamp)
                else:
                    latest[backend][op] = runs[-1]
    return latest


def format_time(seconds: float) -> str:
    """Format time in appropriate units."""
    if seconds >= 1.0:
        return f"{seconds:.3f} s"
    elif seconds >= 0.001:
        return f"{seconds * 1000:.3f} ms"
    else:
        return f"{seconds * 1e6:.1f} µs"


def format_result(value: float) -> str:
    """Format numerical result."""
    if math.isnan(value):
        return f"{Colors.RED}NaN{Colors.END}"
    elif abs(value) < 1e-6:
        return f"{value:.6e}"
    elif abs(value) < 1:
        return f"{value:.9f}"
    else:
        return f"{value:.6f}"


def compute_speedup(c_time: float, mojo_time: float) -> tuple[float, str]:
    """Compute speedup ratio and return colored string."""
    if c_time <= 0 or mojo_time <= 0:
        return 0.0, "N/A"
    
    ratio = c_time / mojo_time
    
    if ratio > 1.0:
        # Mojo is faster
        color = Colors.GREEN
        text = f"{ratio:.2f}x faster"
    elif ratio < 1.0:
        # C is faster
        color = Colors.RED
        inv_ratio = 1.0 / ratio
        text = f"{inv_ratio:.2f}x slower"
    else:
        color = Colors.YELLOW
        text = "same"
    
    return ratio, f"{color}{text}{Colors.END}"


def check_result_match(c_result: float, mojo_result: float, rtol: float = 1e-3) -> tuple[bool, str]:
    """Check if results match within tolerance."""
    if math.isnan(c_result) or math.isnan(mojo_result):
        return False, f"{Colors.RED}NaN detected{Colors.END}"
    
    if c_result == 0 and mojo_result == 0:
        return True, f"{Colors.GREEN}✓ Match{Colors.END}"
    
    if c_result == 0:
        rel_err = abs(mojo_result)
    else:
        rel_err = abs((mojo_result - c_result) / c_result)
    
    if rel_err < rtol:
        return True, f"{Colors.GREEN}✓ Match (err={rel_err:.2e}){Colors.END}"
    else:
        return False, f"{Colors.RED}✗ Mismatch (err={rel_err:.2e}){Colors.END}"


def print_header(text: str, char: str = "="):
    """Print a formatted header."""
    width = 80
    print(f"\n{Colors.BOLD}{Colors.CYAN}{char * width}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(width)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{char * width}{Colors.END}\n")


def print_subheader(text: str):
    """Print a formatted subheader."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'─' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'─' * 60}{Colors.END}")


def analyze_file(filepath: str, show_all_runs: bool = False):
    """Analyze a single benchmark file."""
    filename = os.path.basename(filepath)
    
    parsed_name = parse_contraction_timings_filename(filename)
    if parsed_name:
        nsites, d, chi_max = parsed_name
        param_str = f"nsites={nsites}, d={d}, chi_max={chi_max}"
    else:
        param_str = filename
    
    print_header(f"Benchmark Analysis: {param_str}")
    
    records = load_jsonl(filepath)
    if not records:
        print(f"{Colors.RED}No records found in {filepath}{Colors.END}")
        return
    
    results = parse_results(records)
    latest = get_latest_results(results)
    
    # Get all operations
    all_ops = set()
    for backend_ops in results.values():
        all_ops.update(backend_ops.keys())
    
    # Operations to compare (exclude mojo-only operations for comparison)
    compare_ops = ['mpo_mpo', 'mps_mpo_inner', 'mps_mpo_apply', 'mps_mps']
    
    # Print summary table
    print_subheader("Performance Summary (Latest Run)")
    
    print(f"{'Operation':<20} {'C Time':>12} {'Mojo Time':>12} {'Speedup':>18} {'Result Match':>20}")
    print("─" * 82)
    
    summary_data = []
    
    for op in compare_ops:
        c_res = latest.get('c', {}).get(op)
        mojo_res = latest.get('mojo', {}).get(op)
        
        if c_res and mojo_res:
            c_time_str = format_time(c_res.time_seconds)
            mojo_time_str = format_time(mojo_res.time_seconds)
            _, speedup_str = compute_speedup(c_res.time_seconds, mojo_res.time_seconds)
            _, match_str = check_result_match(c_res.result, mojo_res.result)
            
            print(f"{op:<20} {c_time_str:>12} {mojo_time_str:>12} {speedup_str:>28} {match_str:>30}")
            
            summary_data.append({
                'op': op,
                'c_time': c_res.time_seconds,
                'mojo_time': mojo_res.time_seconds,
                'c_result': c_res.result,
                'mojo_result': mojo_res.result
            })
        elif c_res:
            print(f"{op:<20} {format_time(c_res.time_seconds):>12} {'N/A':>12} {'N/A':>18} {'N/A':>20}")
        elif mojo_res:
            print(f"{op:<20} {'N/A':>12} {format_time(mojo_res.time_seconds):>12} {'N/A':>18} {'N/A':>20}")
    
    # Mojo-only operations
    mojo_only_ops = [op for op in all_ops if op not in compare_ops]
    if mojo_only_ops:
        print(f"\n{Colors.YELLOW}Mojo-only operations:{Colors.END}")
        for op in sorted(mojo_only_ops):
            mojo_res = latest.get('mojo', {}).get(op)
            if mojo_res:
                print(f"  {op:<20} {format_time(mojo_res.time_seconds):>12}")
    
    # Detailed analysis
    print_subheader("Detailed Analysis")
    
    for data in summary_data:
        op = data['op']
        c_time = data['c_time']
        mojo_time = data['mojo_time']
        ratio = c_time / mojo_time if mojo_time > 0 else 0
        
        print(f"\n{Colors.BOLD}{op}{Colors.END}")
        print(f"  C (CPU):     {format_time(c_time):>12}  |  Result: {format_result(data['c_result'])}")
        print(f"  Mojo (GPU):  {format_time(mojo_time):>12}  |  Result: {format_result(data['mojo_result'])}")
        
        if ratio > 1.0:
            print(f"  {Colors.GREEN}→ Mojo is {ratio:.2f}x faster{Colors.END}")
        elif ratio < 1.0 and ratio > 0:
            print(f"  {Colors.RED}→ Mojo is {1/ratio:.2f}x slower{Colors.END}")
            
            # Provide analysis for slow operations
            if op == 'mps_mps':
                print(f"  {Colors.YELLOW}  Analysis: MPS-MPS overlap may still have overhead from:{Colors.END}")
                print(f"  {Colors.YELLOW}    - GPU kernel launch latency (~20-50µs per launch){Colors.END}")
                print(f"  {Colors.YELLOW}    - Small tensor sizes where launch overhead > compute{Colors.END}")
                print(f"  {Colors.YELLOW}    - Memory allocation per operation{Colors.END}")
    
    # Run history (if multiple runs exist)
    if show_all_runs:
        print_subheader("Run History")
        
        for op in compare_ops:
            mojo_runs = results.get('mojo', {}).get(op, [])
            if len(mojo_runs) > 1:
                print(f"\n{Colors.BOLD}{op} (Mojo runs):{Colors.END}")
                for i, run in enumerate(mojo_runs):
                    print(f"  Run {i+1}: {format_time(run.time_seconds):>12}  |  Result: {format_result(run.result)}")
    
    # Overall assessment
    print_subheader("Overall Assessment")
    
    mojo_wins = sum(1 for d in summary_data if d['c_time'] > d['mojo_time'])
    c_wins = sum(1 for d in summary_data if d['c_time'] < d['mojo_time'])
    total = len(summary_data)
    
    has_c = 'c' in latest and bool(latest['c'])
    has_mojo = 'mojo' in latest and bool(latest['mojo'])
    
    if not has_c and has_mojo:
        print(
            f"  {Colors.YELLOW}Only Mojo records in this file — C times show N/A above.{Colors.END}"
        )
        print(
            f"  {Colors.YELLOW}Run C perf_contractions (or run_main.sh) and append the matching "
            f"contraction_timings_*.jsonl lines to compare.{Colors.END}"
        )
    elif has_c and not has_mojo:
        print(
            f"  {Colors.YELLOW}Only C records in this file — Mojo times show N/A above.{Colors.END}"
        )
    
    print(f"  Operations where Mojo (GPU) is faster: {Colors.GREEN}{mojo_wins}/{total}{Colors.END}")
    print(f"  Operations where C (CPU) is faster:    {Colors.RED}{c_wins}/{total}{Colors.END}")
    
    # Check for result mismatches
    mismatches = []
    for d in summary_data:
        match, _ = check_result_match(d['c_result'], d['mojo_result'])
        if not match:
            mismatches.append(d['op'])
    
    if total == 0:
        print(f"\n  {Colors.YELLOW}No C+Mojo pairs for compare_ops — nothing to score for speedup/match.{Colors.END}")
    elif mismatches:
        print(f"\n  {Colors.RED}⚠ Result mismatches detected in: {', '.join(mismatches)}{Colors.END}")
        print(f"  {Colors.YELLOW}  This indicates potential bugs in the Mojo implementation.{Colors.END}")
    else:
        print(f"\n  {Colors.GREEN}✓ All results match between C and Mojo implementations{Colors.END}")


def analyze_all_files(directory: str):
    """Analyze all benchmark files in a directory."""
    pattern = os.path.join(directory, "contraction_timings_*.jsonl")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"{Colors.RED}No benchmark files found in {directory}{Colors.END}")
        return
    
    print_header("Multi-Configuration Benchmark Analysis", "═")
    print(f"Found {len(files)} benchmark file(s)\n")
    
    for filepath in files:
        analyze_file(filepath)
        print("\n" + "═" * 80 + "\n")


def print_ascii_bar_chart(data: list[dict], title: str = "Performance Comparison"):
    """Print an ASCII bar chart comparing C and Mojo times."""
    if not data:
        return
    
    print_subheader(title)
    
    # Find max time for scaling
    max_time = max(max(d['c_time'], d['mojo_time']) for d in data)
    bar_width = 40
    
    for d in data:
        op = d['op']
        c_time = d['c_time']
        mojo_time = d['mojo_time']
        
        c_bar_len = int((c_time / max_time) * bar_width) if max_time > 0 else 0
        mojo_bar_len = int((mojo_time / max_time) * bar_width) if max_time > 0 else 0
        
        print(f"\n{Colors.BOLD}{op}{Colors.END}")
        print(f"  C:    {'█' * c_bar_len}{'░' * (bar_width - c_bar_len)} {format_time(c_time)}")
        
        if mojo_time < c_time:
            bar_color = Colors.GREEN
        else:
            bar_color = Colors.RED
        print(f"  Mojo: {bar_color}{'█' * mojo_bar_len}{'░' * (bar_width - mojo_bar_len)}{Colors.END} {format_time(mojo_time)}")


def print_scaling_analysis(all_results: dict):
    """Analyze how performance scales with chi_max."""
    print_header("Scaling Analysis: Performance vs chi_max", "─")
    
    # Group by chi_max
    chi_data = defaultdict(lambda: {'c': {}, 'mojo': {}})
    
    for chi_max, results in all_results.items():
        latest = get_latest_results(results)
        for backend in ['c', 'mojo']:
            if backend in latest:
                for op, res in latest[backend].items():
                    chi_data[chi_max][backend][op] = res.time_seconds
    
    if len(chi_data) < 2:
        print("  Need at least 2 different chi_max values for scaling analysis")
        return
    
    chi_values = sorted(chi_data.keys())
    ops = ['mpo_mpo', 'mps_mpo_inner', 'mps_mpo_apply', 'mps_mps']
    
    print(f"\n{'chi_max':>10}", end="")
    for op in ops:
        print(f"  {op:>15}", end="")
    print()
    print("─" * (10 + 17 * len(ops)))
    
    for chi in chi_values:
        print(f"\n{Colors.BOLD}chi={chi}{Colors.END}")
        for backend in ['c', 'mojo']:
            label = f"  {backend.upper():>6}:"
            print(label, end="")
            for op in ops:
                t = chi_data[chi][backend].get(op, 0)
                if t > 0:
                    print(f"  {format_time(t):>15}", end="")
                else:
                    print(f"  {'N/A':>15}", end="")
            print()
        
        # Print speedup row
        print(f"  {'Ratio':>6}:", end="")
        for op in ops:
            c_t = chi_data[chi]['c'].get(op, 0)
            m_t = chi_data[chi]['mojo'].get(op, 0)
            if c_t > 0 and m_t > 0:
                ratio = c_t / m_t
                if ratio > 1:
                    color = Colors.GREEN
                    text = f"{ratio:.1f}x▲"
                else:
                    color = Colors.RED
                    text = f"{1/ratio:.1f}x▼"
                print(f"  {color}{text:>15}{Colors.END}", end="")
            else:
                print(f"  {'N/A':>15}", end="")
        print()


def analyze_dmrg_benchmarks(c_path: str = None, mojo_path: str = None):
    """Analyze DMRG benchmark results from C and Mojo.
    
    Args:
        c_path: Path to C DMRG timing JSONL file or directory containing dmrg_*_timings_*.jsonl
        mojo_path: Path to Mojo DMRG timing JSONL file or directory containing dmrg_*_timings_*.jsonl
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Default directories
    if c_path is None:
        c_path = os.path.join(script_dir, '..', '..', '..', '..', 'chemtensor', 'build', 'generated', 'perf')
    if mojo_path is None:
        mojo_path = script_dir

    def load_dmrg_records(path_or_dir: str) -> tuple[list[dict], list[str]]:
        """Load DMRG records from a file or all matching files in a directory."""
        if os.path.isfile(path_or_dir):
            return load_jsonl(path_or_dir), [path_or_dir]
        if os.path.isdir(path_or_dir):
            pattern = os.path.join(path_or_dir, "dmrg_*_timings_*.jsonl")
            files = sorted(glob.glob(pattern))
            records: list[dict] = []
            for f in files:
                records.extend(load_jsonl(f))
            return records, files
        return [], []
    
    print_header("DMRG Performance Comparison: C vs Mojo")

    def _resolved(p: str | None) -> str | None:
        if p is None:
            return None
        try:
            return str(Path(p).resolve())
        except OSError:
            return os.path.abspath(p)

    rc, rm = _resolved(c_path), _resolved(mojo_path)
    same_path = rc is not None and rm is not None and rc == rm

    c_records: list[dict] = []
    c_files: list[str] = []
    mojo_records: list[dict] = []
    mojo_files: list[str] = []

    if same_path:
        c_records, c_files = load_dmrg_records(c_path)
        if c_records:
            print(
                f"Loaded {len(c_records)} DMRG records (C + Mojo) from {len(c_files)} file(s) under {c_path}"
            )
            for f in c_files:
                print(f"  - {os.path.basename(f)}")
        else:
            print(f"{Colors.YELLOW}DMRG results not found: {c_path}{Colors.END}")
            print(
                f"{Colors.YELLOW}Run C and Mojo DMRG benchmarks, then merge with "
                f"tools/merge_and_analyze_benchmarks.py --dmrg{Colors.END}"
            )
    else:
        # Load C results
        c_records, c_files = load_dmrg_records(c_path)
        if c_records:
            print(f"Loaded {len(c_records)} C DMRG records from {len(c_files)} file(s)")
            for f in c_files:
                print(f"  - {os.path.basename(f)}")
        else:
            print(f"{Colors.YELLOW}C DMRG results not found: {c_path}{Colors.END}")
            print(f"{Colors.YELLOW}Run C benchmarks first (cd chemtensor && ./run_main.sh --build){Colors.END}")

        # Load Mojo results
        mojo_records, mojo_files = load_dmrg_records(mojo_path)
        if mojo_records:
            print(f"Loaded {len(mojo_records)} Mojo DMRG records from {len(mojo_files)} file(s)")
            for f in mojo_files:
                print(f"  - {os.path.basename(f)}")
        else:
            print(f"{Colors.YELLOW}Mojo DMRG results not found: {mojo_path}{Colors.END}")
            print(f"{Colors.YELLOW}Run Mojo benchmarks first (mojo bench_contractions.mojo){Colors.END}")

    if not c_records and not mojo_records:
        print(f"\n{Colors.RED}No DMRG benchmark data found.{Colors.END}")
        return
    
    # DMRG operations to compare
    dmrg_ops = ['dmrg_singlesite', 'dmrg_twosite']

    # Parse full record lists for robust parameter-matched comparison.
    all_records = c_records + mojo_records
    results = parse_results(all_records)

    def param_signature(params: dict) -> tuple:
        """Signature to ensure C/Mojo are compared with identical benchmark params."""
        return (
            params.get('nsites'),
            params.get('d'),
            params.get('chi_max'),
            params.get('num_sweeps'),
            params.get('maxiter_lanczos'),
            params.get('J'),
            params.get('D'),
            params.get('h'),
            params.get('tol_split', None),
        )

    def latest_for_signature(runs: list[BenchmarkResult]) -> BenchmarkResult:
        """Pick latest run by timestamp when available, else by file/load order."""
        with_ts = [r for r in runs if r.timestamp is not None]
        if with_ts:
            return max(with_ts, key=lambda r: r.timestamp)
        return runs[-1]

    # For each operation, choose a param signature present in both backends.
    latest: dict[str, dict[str, BenchmarkResult]] = {'c': {}, 'mojo': {}}
    for op in dmrg_ops:
        c_runs = results.get('c', {}).get(op, [])
        m_runs = results.get('mojo', {}).get(op, [])
        if not c_runs or not m_runs:
            if c_runs:
                latest['c'][op] = latest_for_signature(c_runs)
            if m_runs:
                latest['mojo'][op] = latest_for_signature(m_runs)
            continue

        c_by_sig: dict[tuple, list[BenchmarkResult]] = defaultdict(list)
        m_by_sig: dict[tuple, list[BenchmarkResult]] = defaultdict(list)
        for r in c_runs:
            c_by_sig[param_signature(r.params)].append(r)
        for r in m_runs:
            m_by_sig[param_signature(r.params)].append(r)

        common_sigs = [sig for sig in c_by_sig.keys() if sig in m_by_sig]
        if common_sigs:
            # Pick the most recently updated signature.
            def sig_recency(sig: tuple) -> float:
                cr = latest_for_signature(c_by_sig[sig])
                mr = latest_for_signature(m_by_sig[sig])
                c_t = cr.timestamp if cr.timestamp is not None else -1
                m_t = mr.timestamp if mr.timestamp is not None else -1
                return max(c_t, m_t)

            chosen_sig = max(common_sigs, key=sig_recency)
            latest['c'][op] = latest_for_signature(c_by_sig[chosen_sig])
            latest['mojo'][op] = latest_for_signature(m_by_sig[chosen_sig])
        else:
            # Fallback: still show latest per backend, but this means params differ.
            latest['c'][op] = latest_for_signature(c_runs)
            latest['mojo'][op] = latest_for_signature(m_runs)
    
    # Print summary table
    print_subheader("DMRG Performance Summary")
    
    print(f"{'Operation':<20} {'C Time':>12} {'Mojo Time':>12} {'Speedup':>18} {'Energy Match':>20}")
    print("─" * 82)
    
    summary_data = []
    
    for op in dmrg_ops:
        c_res = latest.get('c', {}).get(op)
        mojo_res = latest.get('mojo', {}).get(op)
        
        if c_res and mojo_res:
            c_time_str = format_time(c_res.time_seconds)
            mojo_time_str = format_time(mojo_res.time_seconds)
            _, speedup_str = compute_speedup(c_res.time_seconds, mojo_res.time_seconds)
            _, match_str = check_result_match(c_res.result, mojo_res.result, rtol=2e-3)  # DMRG can have larger tolerance
            
            print(f"{op:<20} {c_time_str:>12} {mojo_time_str:>12} {speedup_str:>28} {match_str:>30}")
            
            summary_data.append({
                'op': op,
                'c_time': c_res.time_seconds,
                'mojo_time': mojo_res.time_seconds,
                'c_result': c_res.result,
                'mojo_result': mojo_res.result,
                'c_params': c_res.params,
                'mojo_params': mojo_res.params
            })
        elif c_res:
            print(f"{op:<20} {format_time(c_res.time_seconds):>12} {'N/A':>12} {'N/A':>18} {'N/A':>20}")
        elif mojo_res:
            print(f"{op:<20} {'N/A':>12} {format_time(mojo_res.time_seconds):>12} {'N/A':>18} {'N/A':>20}")
    
    # Detailed analysis
    print_subheader("Detailed DMRG Analysis")
    
    for data in summary_data:
        op = data['op']
        c_time = data['c_time']
        mojo_time = data['mojo_time']
        ratio = c_time / mojo_time if mojo_time > 0 else 0
        
        print(f"\n{Colors.BOLD}{op}{Colors.END}")
        
        # Print parameters
        c_params = data.get('c_params', {})
        mojo_params = data.get('mojo_params', {})
        
        if c_params:
            params_str = f"nsites={c_params.get('nsites', '?')}, d={c_params.get('d', '?')}, chi_max={c_params.get('chi_max', '?')}, sweeps={c_params.get('num_sweeps', '?')}"
            print(f"  Parameters: {params_str}")
        
        print(f"  C (CPU):     {format_time(c_time):>12}  |  Energy: {format_result(data['c_result'])}")
        print(f"  Mojo (GPU):  {format_time(mojo_time):>12}  |  Energy: {format_result(data['mojo_result'])}")
        
        if ratio > 1.0:
            print(f"  {Colors.GREEN}→ Mojo is {ratio:.2f}x faster{Colors.END}")
        elif ratio < 1.0 and ratio > 0:
            print(f"  {Colors.RED}→ Mojo is {1/ratio:.2f}x slower{Colors.END}")
            print(f"  {Colors.YELLOW}  Note: DMRG involves many sequential operations (sweeps, SVD, Lanczos).{Colors.END}")
            print(f"  {Colors.YELLOW}  GPU speedup depends on system size and chi_max.{Colors.END}")
    
    # Overall assessment
    print_subheader("DMRG Overall Assessment")
    
    mojo_wins = sum(1 for d in summary_data if d['c_time'] > d['mojo_time'])
    c_wins = sum(1 for d in summary_data if d['c_time'] < d['mojo_time'])
    total = len(summary_data)
    
    if total > 0:
        print(f"  Operations where Mojo (GPU) is faster: {Colors.GREEN}{mojo_wins}/{total}{Colors.END}")
        print(f"  Operations where C (CPU) is faster:    {Colors.RED}{c_wins}/{total}{Colors.END}")
        
        # Check for energy mismatches
        mismatches = []
        for d in summary_data:
            match, _ = check_result_match(d['c_result'], d['mojo_result'], rtol=2e-3)
            if not match:
                mismatches.append(d['op'])
        
        if mismatches:
            print(f"\n  {Colors.RED}⚠ Energy mismatches detected in: {', '.join(mismatches)}{Colors.END}")
            print(f"  {Colors.YELLOW}  This may indicate different convergence or initial state.{Colors.END}")
        else:
            print(f"\n  {Colors.GREEN}✓ All DMRG energies match between C and Mojo (within 0.2% tolerance){Colors.END}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze C vs Mojo benchmark results")
    parser.add_argument('--file', '-f', type=str, help='Specific JSONL file to analyze')
    parser.add_argument('--all', '-a', action='store_true', help='Analyze all benchmark files')
    parser.add_argument('--show-runs', '-r', action='store_true', help='Show all run history')
    parser.add_argument('--chart', '-c', action='store_true', help='Show ASCII bar charts')
    parser.add_argument('--scaling', '-s', action='store_true', help='Show scaling analysis')
    parser.add_argument('--dmrg', action='store_true', help='Analyze DMRG benchmark results')
    parser.add_argument('--c-dmrg', type=str, help='Path to C DMRG timing JSONL file')
    parser.add_argument('--mojo-dmrg', type=str, help='Path to Mojo DMRG timing JSONL file')
    parser.add_argument('--dir', '-d', type=str, default='.', help='Directory containing benchmark files')
    
    args = parser.parse_args()
    
    # Determine the directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.dir == '.':
        search_dir = script_dir
    else:
        search_dir = args.dir
    
    if args.dmrg:
        # Analyze DMRG benchmarks
        analyze_dmrg_benchmarks(c_path=args.c_dmrg, mojo_path=args.mojo_dmrg)
    elif args.scaling:
        # Load all files for scaling analysis
        pattern = os.path.join(search_dir, "contraction_timings_*.jsonl")
        files = glob.glob(pattern)
        all_results = {}
        for filepath in files:
            filename = os.path.basename(filepath)
            parsed = parse_contraction_timings_filename(filename)
            if parsed is None:
                continue
            chi_max = parsed[2]
            records = load_jsonl(filepath)
            all_results[chi_max] = parse_results(records)
        print_scaling_analysis(all_results)
    elif args.file:
        filepath = args.file if os.path.isabs(args.file) else os.path.join(search_dir, args.file)
        if os.path.exists(filepath):
            analyze_file(filepath, show_all_runs=args.show_runs)
        else:
            print(f"{Colors.RED}File not found: {filepath}{Colors.END}")
            sys.exit(1)
    elif args.all:
        analyze_all_files(search_dir)
    else:
        # Default: analyze the most recent file (by modification time)
        pattern = os.path.join(search_dir, "contraction_timings_*.jsonl")
        files = glob.glob(pattern)
        if files:
            # Sort by modification time, newest first
            files.sort(key=os.path.getmtime, reverse=True)
            print(f"Analyzing most recent file: {os.path.basename(files[0])}")
            analyze_file(files[0], show_all_runs=args.show_runs)
        else:
            print(f"{Colors.RED}No benchmark files found. Run benchmarks first.{Colors.END}")
            print(f"Usage: python {os.path.basename(__file__)} --file <file.jsonl>")
            print(f"       python {os.path.basename(__file__)} --all")
            print(f"       python {os.path.basename(__file__)} --dmrg")
            sys.exit(1)


if __name__ == "__main__":
    main()
