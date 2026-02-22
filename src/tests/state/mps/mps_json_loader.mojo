"""
JSON loader for MPS reference data from C implementation.

Loads observables only (nsites, d, bond_dims, norm, state_vector) for
implementation-agnostic comparison. No per-site tensor data.

Also supports loading a full MPS from C perf export: nsites, d, bond_dims,
and sites with shape + data per site (for identical randomness in benchmarks).
"""

from collections.list import List
from python import Python, PythonObject
from gpu.host import DeviceContext
from src.m_tensor.dense_tensor import create_dense_tensor_from_data
from src.state.mps_state import MPSSite, MatrixProductState


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


fn load_mps_from_sites_json[dtype: DType = DType.float32](
    filepath: String,
    ctx: DeviceContext,
) raises -> MatrixProductState[dtype]:
    """Load MPS from C perf export JSON: nsites, d, bond_dims, sites with shape + data.

    C exports double; values are converted to dtype (e.g. float32) for Mojo.
    """
    var py = Python.import_module("builtins")
    var json_module = Python.import_module("json")

    var f = py.open(filepath, "r")
    var content = f.read()
    f.close()

    var data = json_module.loads(content)
    var sites_py = data["sites"]

    var sites = List[MPSSite[dtype]](capacity=len(sites_py))
    for i in range(len(sites_py)):
        var site_data = sites_py[i]
        var shape_py = site_data["shape"]
        var shape = List[Int]()
        for j in range(len(shape_py)):
            shape.append(Int(py.int(shape_py[j])))

        var data_py = site_data["data"]
        var tensor_data = List[Scalar[dtype]]()
        for j in range(len(data_py)):
            var py_val: PythonObject = data_py[j]
            var type_name = String(py.type(py_val).__name__)
            var to_float: PythonObject = py_val
            if type_name == "list":
                to_float = py_val[0]
            var py_float = Python.float(to_float)
            var s = String(py.str(py_float))
            var val_f64 = atof(s)

            @parameter
            if dtype == DType.float32:
                tensor_data.append(Scalar[dtype](Float32(val_f64)))
            else:
                tensor_data.append(Scalar[dtype](val_f64))

        var tensor = create_dense_tensor_from_data[dtype](ctx, tensor_data^, shape^)
        sites.append(MPSSite[dtype](tensor^))

    return MatrixProductState[dtype](sites^)


fn mps_sites_json_exists(filepath: String) raises -> Bool:
    """Return True if path exists and is readable (for C perf export JSON)."""
    var py = Python.import_module("builtins")
    var os = Python.import_module("os")
    return Bool(py.bool(os.path.exists(filepath)))
