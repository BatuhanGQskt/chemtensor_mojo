"""
JSON loader for MPS reference data from C implementation.

Loads observables only (nsites, d, bond_dims, norm, state_vector) for
implementation-agnostic comparison. No per-site tensor data.
"""

from collections.list import List
from python import Python, PythonObject


@fieldwise_init
struct MPSReference(Copyable, Movable):
    """Reference MPS observables loaded from C implementation JSON export.

    Used to compare Mojo MPS against C via norm and overlap only,
    not raw tensor entries.
    """
    var nsites: Int
    var d: Int
    var bond_dims: List[Int]
    var norm: Float64
    var state_vector: List[Float64]


fn load_mps_reference(filepath: String) raises -> MPSReference:
    """Load MPS reference (observables + state vector) from JSON exported by C."""
    var py = Python.import_module("builtins")
    var json_module = Python.import_module("json")

    var f = py.open(filepath, "r")
    var content = f.read()
    f.close()

    var data = json_module.loads(content)

    var nsites = Int(py.int(data["nsites"]))
    var d = Int(py.int(data["d"]))

    var bond_dims = List[Int]()
    var bond_dims_py = data["bond_dims"]
    for i in range(len(bond_dims_py)):
        bond_dims.append(Int(py.int(bond_dims_py[i])))

    var norm_py: PythonObject = data["norm"]
    var norm_s = String(py.str(py.float(norm_py)))
    var norm_val = atof(norm_s)

    var state_vector = List[Float64]()
    var vec_py = data["state_vector"]
    for i in range(len(vec_py)):
        var py_val: PythonObject = vec_py[i]
        var s = String(py.str(Python.float(py_val)))
        state_vector.append(atof(s))

    return MPSReference(nsites, d, bond_dims^, norm_val, state_vector^)


fn print_mps_reference_info(mps_ref: MPSReference) -> None:
    """Print loaded MPS reference summary."""
    print("MPS Reference (observables):")
    print("  nsites:", mps_ref.nsites)
    print("  d:", mps_ref.d)
    var bond_str = String("  bond_dims: [")
    for i in range(len(mps_ref.bond_dims)):
        if i > 0:
            bond_str += ", "
        bond_str += String(mps_ref.bond_dims[i])
    bond_str += "]"
    print(bond_str)
    print("  norm:", mps_ref.norm)
    print("  state_vector length:", len(mps_ref.state_vector))
