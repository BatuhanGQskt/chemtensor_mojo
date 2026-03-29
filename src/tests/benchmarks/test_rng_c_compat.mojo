"""
Compare Mojo rng_c_compat with C rng.c output.

Reads test_data/rng_reference.txt produced by the C project's rng_export
(see chemtensor/test/util/rng_export.c). Same seed must yield identical
rand_uint32 and randnf sequences so that benchmarks (e.g. random MPS) match.

Generate reference: from chemtensor repo run ./run_main.sh (builds and runs
rng_export 42 64, copies rng_reference.txt to chemtensor_mojo/test_data/).
"""

from collections.list import List
from python import Python, PythonObject
from testing import TestSuite
from src.tests.benchmarks.rng_c_compat import seed_rng_state


fn _split_line(s: String) -> List[String]:
    """Split a line on spaces; returns list of tokens."""
    var out = List[String]()
    var start: Int = 0
    while start < len(s):
        var pos = s.find(" ", start)
        if pos == -1:
            pos = len(s)
        if pos > start:
            out.append(s[start:pos])
        start = pos + 1
    return out^


fn _load_rng_reference(filepath: String) raises -> Tuple[Bool, UInt64, Int, List[UInt32], List[Float32]]:
    """Load C export file. Returns (ok, seed, n, u32_list, randnf_list)."""
    var py = Python.import_module("builtins")
    var os = Python.import_module("os")
    if not Bool(py.bool(os.path.exists(filepath))):
        return Tuple[Bool, UInt64, Int, List[UInt32], List[Float32]](False, 0, 0, List[UInt32](), List[Float32]())

    var f = py.open(filepath, "r")
    var content = f.read()
    f.close()

    var lines = content.split("\n")
    var line_count = Int(py.int(py.len(lines)))

    # Expect: "seed <seed>", "n <count>", "u32", <n> u32 lines, "randnf", <n> float lines
    if line_count < 5:
        return Tuple[Bool, UInt64, Int, List[UInt32], List[Float32]](False, 0, 0, List[UInt32](), List[Float32]())
    var seed_line = String(py.str(lines[0]))
    var n_line = String(py.str(lines[1]))
    var u32_header = String(py.str(lines[2]))

    # Parse "seed 42" and "n 64" (space-separated)
    var seed_parts = _split_line(seed_line)
    var n_parts = _split_line(n_line)
    if len(seed_parts) < 2 or len(n_parts) < 2 or u32_header != "u32":
        return Tuple[Bool, UInt64, Int, List[UInt32], List[Float32]](False, 0, 0, List[UInt32](), List[Float32]())

    var seed_str = seed_parts[1]
    var n_val = Int(atol(n_parts[1]))
    var seed = UInt64(atol(seed_str))

    var u32_list = List[UInt32](capacity=n_val)
    for i in range(n_val):
        var idx = 3 + i
        if idx >= line_count:
            return Tuple[Bool, UInt64, Int, List[UInt32], List[Float32]](False, 0, 0, List[UInt32](), List[Float32]())
        var s = String(py.str(lines[idx]))
        u32_list.append(UInt32(atol(s)))

    var randnf_header_idx = 3 + n_val
    if randnf_header_idx >= line_count or String(py.str(lines[randnf_header_idx])) != "randnf":
        return Tuple[Bool, UInt64, Int, List[UInt32], List[Float32]](False, 0, 0, List[UInt32](), List[Float32]())

    var randnf_list = List[Float32](capacity=n_val)
    for i in range(n_val):
        var idx = randnf_header_idx + 1 + i
        if idx >= line_count:
            return Tuple[Bool, UInt64, Int, List[UInt32], List[Float32]](False, 0, 0, List[UInt32](), List[Float32]())
        var s = String(py.str(lines[idx]))
        randnf_list.append(Float32(atof(s)))

    return Tuple[Bool, UInt64, Int, List[UInt32], List[Float32]](True, seed, n_val, u32_list^, randnf_list^)


fn test_rng_vs_c_reference() raises:
    """Compare Mojo rng_c_compat output with C rng_reference.txt."""
    var ref_path = String("test_data/rng_reference.txt")
    var t = _load_rng_reference(ref_path)
    var ok = t[0]
    var ref_seed = t[1]
    var n = t[2]
    var ref_u32 = t[3].copy()
    var ref_randnf = t[4].copy()

    if not ok:
        print("[SKIP] test_data/rng_reference.txt not found or invalid. Run chemtensor/run_main.sh to generate it.")
        return

    var rng = seed_rng_state(ref_seed)

    # Compare rand_uint32 sequence
    for i in range(n):
        var result = rng.rand_uint32()
        rng = result.next
        if result.value != ref_u32[i]:
            raise Error(
                "rand_uint32 mismatch at index " + String(i) + ": Mojo " + String(result.value) + " vs C " + String(ref_u32[i])
            )
    print("  rand_uint32: " + String(n) + " values match C")

    # Compare randnf sequence (allow small float tolerance)
    var atol: Float32 = 1e-6
    var rtol: Float32 = 1e-5
    for i in range(n):
        var result = rng.randnf()
        rng = result.next
        var mojo_val = result.value
        var c_val = ref_randnf[i]
        var diff = abs(mojo_val - c_val)
        var tol = atol + rtol * max(abs(mojo_val), abs(c_val))
        if diff > tol:
            raise Error(
                "randnf mismatch at index " + String(i) + ": Mojo " + String(mojo_val) + " vs C " + String(c_val)
            )
    print("  randnf: " + String(n) + " values match C (atol=" + String(atol) + ", rtol=" + String(rtol) + ")")
    print("  RNG C/Mojo compatibility: OK")


fn main() raises:
    try:
        TestSuite.discover_tests[__functions_in_module()]().run()
    except e:
        print("Tests failed: " + String(e))
        raise e
