"""
Compare ChemTensor-Mojo tensors / dense Hamiltonian against a Python reference dump.

This is meant to help debug large DMRG energy discrepancies by checking:
- MPO site tensor axis/layout and values
- Dense Hamiltonian (from contracting the MPO) matches Python-exported dense H

Expected input (produced by the Python side, e.g. `mpo_mps_dump.py`):
- out/tensors_chemtensor.npz containing:
    - H_dense (2**N x 2**N)
    - mpo_A_0, mpo_A_1, ... (raw export)
    - mpo_A_0_DlDrdd, ... (axis-normalized export matching [Dl, Dr, d_in, d_out])

Run:
  mojo run src/tests/test_compare_python_dump.mojo

Configure the dump location via env vars (preferred):
  CHEMTENSOR_DUMP_NPZ=/abs/path/to/out/tensors_chemtensor.npz
or:
  CHEMTENSOR_DUMP_DIR=/abs/path/to/out   (expects tensors_chemtensor.npz inside)

Also configure the Ising parameters to match the Python dump (defaults shown):
  CHEMTENSOR_ISING_J=1.0
  CHEMTENSOR_ISING_H=0.0   (longitudinal field, multiplies Z)
  CHEMTENSOR_ISING_G=1.0   (transverse field, multiplies X)
"""

from sys import has_accelerator
from collections.list import List
from collections.optional import Optional
from gpu.host import DeviceContext
from python import Python, PythonObject
from testing import TestSuite

from src.state.hamiltonians import create_ising_1d_mpo
from src.state.mpo_state import MatrixProductOperator

# Reuse dense contraction helpers (and a couple asserts) from the existing sanity test.
from src.tests.test_mpo_mps_sanity import (
    assert_close,
    assert_true,
    contract_mpo_to_dense,
    max_abs_diff,
    pack_rank4_sites_f64,
)


fn _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


fn _env_get_str(os: PythonObject, key: String) raises -> Optional[String]:
    var env = os.environ
    # os.environ behaves like a dict[str,str]
    if env.__contains__(key):
        return String(env.__getitem__(key))
    return None


fn _env_get_f64(os: PythonObject, key: String, default: Float64) raises -> Float64:
    var s = _env_get_str(os, key)
    if s is None:
        return default
    var builtins = Python.import_module("builtins")
    var float_fn = builtins.float
    return Float64(float_fn(s.value()))


fn _resolve_npz_path() raises -> String:
    var os = Python.import_module("os")

    var p = _env_get_str(os, "CHEMTENSOR_DUMP_NPZ")
    if p is not None:
        var path = p.value()
        if os.path.exists(path):
            return path
        raise Error("CHEMTENSOR_DUMP_NPZ points to a non-existent file: " + path)

    var d = _env_get_str(os, "CHEMTENSOR_DUMP_DIR")
    if d is not None:
        var dirp = d.value()
        var candidate = os.path.join(dirp, "tensors_chemtensor.npz")
        if os.path.exists(candidate):
            return String(candidate)
        raise Error("CHEMTENSOR_DUMP_DIR does not contain tensors_chemtensor.npz: " + String(candidate))

    # Default fallback: repo-local out/tensors_chemtensor.npz
    var candidate = os.path.join("out", "tensors_chemtensor.npz")
    if os.path.exists(candidate):
        return String(candidate)

    var p_str = String(p.value()) if p else "None"
    var d_str = String(d.value()) if d else "None"
    raise Error(
        "Could not find Python dump.\n"
        + "Checked env CHEMTENSOR_DUMP_NPZ=" + p_str + "\n"
        + "Checked env CHEMTENSOR_DUMP_DIR=" + d_str + "\n"
        + "Checked default: " + candidate + "\n"
        + "Please set CHEMTENSOR_DUMP_DIR to the folder containing tensors_chemtensor.npz"
    )


fn _npz_keys(npz: PythonObject) raises -> List[String]:
    var keys_py = npz.files  # list[str]
    var keys = List[String](capacity=Int(len(keys_py)))
    for i in range(Int(len(keys_py))):
        keys.append(String(keys_py.__getitem__(i)))
    return keys^

fn _infer_nsites_from_npz(npz: PythonObject) raises -> Int:
    # Infer N from keys like "mpo_A_0_DlDrdd", "mpo_A_1", ...
    var re = Python.import_module("re")
    var builtins = Python.import_module("builtins")
    var int_fn = builtins.int
    var keys_py = npz.files
    var max_idx: Int = -1
    for i in range(Int(len(keys_py))):
        var k = String(keys_py.__getitem__(i))
        var m = re.match("^mpo_A_(\\d+)", k)
        if m is None:
            continue
        var idx = Int(int_fn(m.group(1)))
        if idx > max_idx:
            max_idx = idx
    if max_idx < 0:
        raise Error("Could not infer nsites from npz keys (expected keys like 'mpo_A_0...').")
    return max_idx + 1


fn _np_get_f64(builtins: PythonObject, arr: PythonObject, i0: Int, i1: Int, i2: Int, i3: Int) raises -> Float64:
    # numpy.ndarray.item(i0,i1,i2,i3) -> scalar
    var float_fn = builtins.float
    return Float64(float_fn(arr.item(i0, i1, i2, i3)))


fn _load_dense_from_npz(builtins: PythonObject, np: PythonObject, npz: PythonObject) raises -> List[Float64]:
    if not npz.__contains__("H_dense"):
        raise Error("npz is missing key 'H_dense'")
    var H = npz.__getitem__("H_dense")
    # Ensure float64 and flatten.
    var flat = np.asarray(H, dtype=np.float64).reshape(-1)
    var n = Int(flat.size)
    var out = List[Float64](capacity=n)
    for i in range(n):
        out.append(Float64(builtins.float(flat.__getitem__(i))))
    return out^


fn _choose_mpo_key_for_site(npz: PythonObject, site: Int) raises -> String:
    # Prefer raw key matching Mojo's convention:
    # [Dl, d_in, d_out, Dr] (called mpo_A_i in the Python dump).
    var k_raw = "mpo_A_" + String(site)
    if npz.__contains__(k_raw):
        return k_raw
    # Fallback (old convention, though we expect raw now)
    var k_norm = "mpo_A_" + String(site) + "_DlDrdd"
    if npz.__contains__(k_norm):
        return k_norm
    raise Error("npz missing MPO site key for site " + String(site) + " (tried " + k_raw + " and " + k_norm + ")")


fn check_mpo_site_tensors_match_python_dump(ctx: DeviceContext, mpo: MatrixProductOperator[DType.float64], npz: PythonObject) raises -> None:
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")

    var Wpack = pack_rank4_sites_f64(mpo.sites, ctx)
    var tol = 1e-12

    for site in range(mpo.num_sites()):
        var key = _choose_mpo_key_for_site(npz, site)
        var A_py = np.asarray(npz.__getitem__(key), dtype=np.float64)

        # Shape check.
        var shp = A_py.shape
        if Int(len(shp)) != 4:
            raise Error("Python MPO site tensor is not rank-4 at site " + String(site) + " key=" + key)

        var s0 = Int(shp.__getitem__(0))
        var s1 = Int(shp.__getitem__(1))
        var s2 = Int(shp.__getitem__(2))
        var s3 = Int(shp.__getitem__(3))

        var mojo_shape = mpo.site_shape(site)
        assert_true("site " + String(site) + " shape[0]", mojo_shape[0] == s0)
        assert_true("site " + String(site) + " shape[1]", mojo_shape[1] == s1)
        assert_true("site " + String(site) + " shape[2]", mojo_shape[2] == s2)
        assert_true("site " + String(site) + " shape[3]", mojo_shape[3] == s3)

        # Value check (full element-wise; tensors are tiny for Ising MPO).
        for i0 in range(s0):
            for i1 in range(s1):
                for i2 in range(s2):
                    for i3 in range(s3):
                        var got = Wpack.get(site, i0, i1, i2, i3)
                        var expected = _np_get_f64(builtins, A_py, i0, i1, i2, i3)
                        assert_close(
                            "MPO tensor mismatch: site="
                            + String(site)
                            + " key="
                            + key
                            + " ["
                            + String(i0)
                            + ","
                            + String(i1)
                            + ","
                            + String(i2)
                            + ","
                            + String(i3)
                            + "]",
                            got,
                            expected,
                            tol,
                        )


fn check_dense_h_matches_python_dump(
    ctx: DeviceContext,
    nsites: Int,
    J: Float64,
    h: Float64,
    g: Float64,
    H_py_flat: List[Float64],
) raises -> None:
    var dim = 1 << nsites
    if len(H_py_flat) != dim * dim:
        raise Error(
            "Python H_dense size mismatch: expected "
            + String(dim * dim)
            + " but got "
            + String(len(H_py_flat))
            + " (nsites="
            + String(nsites)
            + ")"
        )

    # The only ambiguity should be basis-index ordering. Try both conventions and accept either.
    # Note: `contract_mpo_to_dense` consumes the MPO, so build two identical MPOs.
    var mpo_msb = create_ising_1d_mpo[DType.float64](ctx, nsites, J=J, h_longitudinal=h, g_transverse=g)
    var mpo_lsb = create_ising_1d_mpo[DType.float64](ctx, nsites, J=J, h_longitudinal=h, g_transverse=g)
    var H_msb = contract_mpo_to_dense(mpo_msb^, ctx, msb_first=True)
    var H_lsb = contract_mpo_to_dense(mpo_lsb^, ctx, msb_first=False)

    var err_msb = max_abs_diff(H_py_flat, H_msb)
    var err_lsb = max_abs_diff(H_py_flat, H_lsb)

    print("Python dump dense-H check: err(msb_first)=", err_msb, " err(lsb_first)=", err_lsb)

    var tol = 1e-10  # allow a tiny bit more slack if Python uses float32 internally
    if err_msb <= tol or err_lsb <= tol:
        return

    raise Error(
        "Dense H mismatch vs Python dump: err(msb_first)="
        + String(err_msb)
        + " err(lsb_first)="
        + String(err_lsb)
    )


fn test_python_dump_comparison() raises:
    print("\n" + "#" * 70)
    print("# COMPARE AGAINST PYTHON DUMP (MPO tensors + dense H)")
    print("#" * 70)

    @parameter
    if not has_accelerator():
        print("No compatible GPU found - skipping Python dump comparison (DeviceContext required).")
        return

    var os = Python.import_module("os")
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")

    var npz_path: String
    try:
        npz_path = _resolve_npz_path()
    except e:
        print("Skipping Python dump comparison: " + String(e))
        return

    print("Loading Python dump from: ", npz_path)
    var npz = np.load(npz_path)
    var nsites = _infer_nsites_from_npz(npz)

    var J = _env_get_f64(os, "CHEMTENSOR_ISING_J", 1.0)
    var h = _env_get_f64(os, "CHEMTENSOR_ISING_H", 0.0)
    var g = _env_get_f64(os, "CHEMTENSOR_ISING_G", 1.0)
    print("Using Ising params: nsites=", nsites, " J=", J, " h_longitudinal=", h, " g_transverse=", g)

    # Load dense H from Python dump (if present).
    var H_py = _load_dense_from_npz(builtins, np, npz)

    with DeviceContext() as ctx:
        # Compare local MPO tensors
        var mpo_local = create_ising_1d_mpo[DType.float64](ctx, nsites, J=J, h_longitudinal=h, g_transverse=g)
        check_mpo_site_tensors_match_python_dump(ctx, mpo_local^, npz)
        print("✓ MPO site tensors match Python dump (key: mpo_A_*)")

        # Compare dense Hamiltonian (try both basis-index conventions)
        check_dense_h_matches_python_dump(ctx, nsites, J, h, g, H_py)
        print("✓ Dense Hamiltonian matches Python dump (up to basis ordering)")

    print("#" * 70)
    print("# PYTHON DUMP COMPARISON PASSED")
    print("#" * 70 + "\n")


fn main():
    try:
        TestSuite.discover_tests[__functions_in_module()]().run()
    except e:
        print("Tests failed: " + String(e))
