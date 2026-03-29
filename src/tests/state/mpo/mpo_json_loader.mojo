"""
JSON loader for MPO reference data from C implementation.

This module provides utilities to parse JSON files exported from the C implementation
and load them into Mojo data structures for comparison testing.
"""

from collections.list import List
from gpu.host import DeviceContext
from src.m_tensor.dense_tensor import DenseTensor, create_dense_tensor_from_data
from python import Python, PythonObject


@fieldwise_init
struct MPOReference[dtype: DType](Copyable, Movable):
    """Reference MPO data loaded from C implementation JSON export.
    
    Attributes:
        nsites: Number of sites in the MPO.
        d: Physical dimension (local Hilbert space dimension).
        bond_dims: List of bond dimensions [length = nsites + 1].
        qsite: Quantum numbers for physical sites.
        site_tensors: List of dense tensors for each site [shape: Wl, d, d, Wr].
    """
    var nsites: Int
    var d: Int
    var bond_dims: List[Int]
    var qsite: List[Int]
    var site_tensors: List[DenseTensor[dtype]]


fn load_mpo_reference[dtype: DType](
    filepath: String,
    ctx: DeviceContext
) raises -> MPOReference[dtype]:
    """Load MPO reference data from JSON file exported by C implementation.
    
    Args:
        filepath: Path to the JSON file containing MPO reference data.
        ctx: Device context for tensor allocation.
    
    Returns:
        MPOReference structure containing the loaded MPO data.
    
    Raises:
        Error if the file cannot be read or parsed.
    """
    var py = Python.import_module("builtins")
    var json_module = Python.import_module("json")
    
    # Read JSON file
    var f = py.open(filepath, "r")
    var content = f.read()
    f.close()
    
    # Parse JSON
    var data = json_module.loads(content)
    
    # Parse metadata (convert Python numbers to Mojo Int via py.int())
    var nsites = Int(py.int(data["nsites"]))
    var d = Int(py.int(data["d"]))
    
    # Parse bond dimensions
    var bond_dims = List[Int]()
    var bond_dims_py = data["bond_dims"]
    for i in range(len(bond_dims_py)):
        bond_dims.append(Int(py.int(bond_dims_py[i])))
    
    # Parse quantum numbers
    var qsite = List[Int]()
    var qsite_py = data["qsite"]
    for i in range(len(qsite_py)):
        qsite.append(Int(py.int(qsite_py[i])))
    
    # Parse site tensors
    var site_tensors = List[DenseTensor[dtype]]()
    var sites_py = data["sites"]
    for i in range(len(sites_py)):
        var site_data = sites_py[i]
        
        # Parse shape
        var shape_py = site_data["shape"]
        var shape = List[Int]()
        for j in range(len(shape_py)):
            shape.append(Int(py.int(shape_py[j])))

        var data_py = site_data["data"]
        var tensor_data = List[Scalar[dtype]]()
        for j in range(len(data_py)):
            # C export is real-only (flat numbers). Legacy JSON may have [re, im] lists; use first element.
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
        
        # Create dense tensor
        var tensor = create_dense_tensor_from_data[dtype](ctx, tensor_data^, shape^)
        site_tensors.append(tensor^)
    
    return MPOReference[dtype](nsites, d, bond_dims^, qsite^, site_tensors^)


fn print_mpo_reference_info[D: DType](mpo_ref: MPOReference[D]) -> None:
    """Print information about loaded MPO reference data.
    
    Args:
        mpo_ref: MPO reference structure to print.
    """
    print("MPO Reference Data:")
    print("  Number of sites:", mpo_ref.nsites)
    print("  Physical dimension:", mpo_ref.d)
    
    var bond_str = String("  Bond dimensions: [")
    for i in range(len(mpo_ref.bond_dims)):
        if i > 0:
            bond_str += ", "
        bond_str += String(mpo_ref.bond_dims[i])
    bond_str += "]"
    print(bond_str)
    
    var qsite_str = String("  Quantum numbers: [")
    for i in range(len(mpo_ref.qsite)):
        if i > 0:
            qsite_str += ", "
        qsite_str += String(mpo_ref.qsite[i])
    qsite_str += "]"
    print(qsite_str)
    
    print("\n  Site tensors:")
    for i in range(len(mpo_ref.site_tensors)):
        var shape = mpo_ref.site_tensors[i].shape.copy()
        var shape_str = String("    Site ") + String(i) + ": ["
        for j in range(len(shape)):
            if j > 0:
                shape_str += ", "
            shape_str += String(shape[j])
        shape_str += "]"
        print(shape_str)
