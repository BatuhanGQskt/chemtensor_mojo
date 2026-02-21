"""
JSON loader for DMRG reference data from C implementation.

Loads energy_final, en_sweeps, bond_dims, norm, and params for
implementation-agnostic comparison (same schema as MPS/MPO reference loaders).
Used by test_dmrg_c_comparison.mojo to compare Mojo DMRG output against C.
"""

from collections.list import List
from python import Python, PythonObject


@fieldwise_init
struct DMRGReference(Copyable, Movable):
    """Reference DMRG results loaded from C implementation JSON export.

    Matches the schema from dmrg_results_to_json (C) / save_dmrg_results_to_json (Mojo).
    """
    var impl: String
    var model: String
    var nsites: Int
    var d: Int
    var J: Float64
    var D: Float64
    var h: Float64
    var num_sweeps: Int
    var maxiter_lanczos: Int
    var chi_max: Int
    var tol_split: Float64  # -1.0 if omitted (single-site)
    var energy_final: Float64
    var en_sweeps: List[Float64]
    var bond_dims: List[Int]
    var norm: Float64
    var entropy: List[Float64]  # optional; empty if not present


fn _py_float(py: PythonObject, py_val: PythonObject) raises -> Float64:
    """Convert Python number to Float64 (handles int or float in JSON). py = builtins."""
    return atof(String(py.str(Python.float(py_val))))


fn _py_int(py: PythonObject, py_val: PythonObject) raises -> Int:
    """Convert Python number to Int. py = builtins."""
    return Int(py.int(py_val))


fn load_dmrg_reference(filepath: String) raises -> DMRGReference:
    """Load DMRG reference from JSON file exported by C (or Mojo) dmrg_results_to_json."""
    var py = Python.import_module("builtins")
    var json_module = Python.import_module("json")

    var f = py.open(filepath, "r")
    var content = f.read()
    f.close()

    var data = json_module.loads(content)

    var impl = String(py.str(data["impl"]))
    var model = String(py.str(data["model"]))

    var params = data["params"]
    var nsites = _py_int(py, params["nsites"])
    var d = _py_int(py, params["d"])
    var J = _py_float(py, params["J"])
    var D = _py_float(py, params["D"])
    var h = _py_float(py, params["h"])
    var num_sweeps = _py_int(py, params["num_sweeps"])
    var maxiter_lanczos = _py_int(py, params["maxiter_lanczos"])
    var chi_max = _py_int(py, params["chi_max"])
    # tol_split optional (single-site JSON omits it)
    var tol_split: Float64 = -1.0
    var param_keys = py.list(params.keys())
    for i in range(len(param_keys)):
        if String(py.str(param_keys[i])) == "tol_split":
            tol_split = _py_float(py, params["tol_split"])
            break

    var energy_final = _py_float(py, data["energy_final"])

    var en_sweeps = List[Float64]()
    var en_sweeps_py = data["en_sweeps"]
    for i in range(len(en_sweeps_py)):
        en_sweeps.append(_py_float(py, en_sweeps_py[i]))

    var bond_dims = List[Int]()
    var bond_dims_py = data["bond_dims"]
    for i in range(len(bond_dims_py)):
        bond_dims.append(_py_int(py, bond_dims_py[i]))

    var norm = _py_float(py, data["norm"])

    var entropy = List[Float64]()
    var data_keys = py.list(data.keys())
    for i in range(len(data_keys)):
        if String(py.str(data_keys[i])) == "entropy":
            var ent_py = data["entropy"]
            for j in range(len(ent_py)):
                entropy.append(_py_float(py, ent_py[j]))
            break

    return DMRGReference(
        impl^, model^,
        nsites, d, J, D, h,
        num_sweeps, maxiter_lanczos, chi_max, tol_split,
        energy_final, en_sweeps^, bond_dims^, norm, entropy^
    )


fn print_dmrg_reference_info(dmrg_ref: DMRGReference) -> None:
    """Print loaded DMRG reference summary."""
    print("DMRG Reference:")
    print("  impl:", dmrg_ref.impl)
    print("  model:", dmrg_ref.model)
    print("  nsites:", dmrg_ref.nsites)
    print("  d:", dmrg_ref.d)
    print("  params: J=" + String(dmrg_ref.J) + " D=" + String(dmrg_ref.D) + " h=" + String(dmrg_ref.h))
    print("  num_sweeps:", dmrg_ref.num_sweeps)
    print("  chi_max:", dmrg_ref.chi_max)
    print("  energy_final:", dmrg_ref.energy_final)
    print("  norm:", dmrg_ref.norm)
    var bd_str = String("  bond_dims: [")
    for i in range(len(dmrg_ref.bond_dims)):
        if i > 0:
            bd_str += ", "
        bd_str += String(dmrg_ref.bond_dims[i])
    bd_str += "]"
    print(bd_str)
