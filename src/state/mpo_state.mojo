from collections.list import List
from gpu.host import DeviceContext
from src.m_tensor.dynamic_tensor import (
    DynamicTensor,
    create_dynamic_tensor,
    create_dynamic_tensor_from_data,
)


@fieldwise_init
struct MPOSite[dtype: DType](Writable, Movable, ImplicitlyCopyable):
    """Single site tensor inside an MPO (Matrix Product Operator).
    
    Each site is stored as a rank-4 DynamicTensor with layout
    [left_bond, phys_in, phys_out, right_bond] or [Wl, d_in, d_out, Wr].
    
    This convention matches the standard ChemTensor Python implementation:
    - Contract phys_in with ket physical index
    - phys_out becomes the new physical index
    """
    var tensor: DynamicTensor[dtype]
    
    fn rank(self) -> Int:
        return len(self.tensor.shape)
    
    fn shape(self) -> List[Int]:
        return self.tensor.shape.copy()
    
    fn left_bond_dim(self) raises -> Int:
        self._assert_rank4()
        return self.tensor.shape[0]
    
    fn physical_in_dim(self) raises -> Int:
        self._assert_rank4()
        return self.tensor.shape[1]
    
    fn physical_out_dim(self) raises -> Int:
        self._assert_rank4()
        return self.tensor.shape[2]

    fn right_bond_dim(self) raises -> Int:
        self._assert_rank4()
        return self.tensor.shape[3]
    
    fn _assert_rank4(self) raises -> None:
        var rank = len(self.tensor.shape)
        if rank != 4:
            raise Error(
                "MPOSite expects rank-4 tensors [Wl, d_in, d_out, Wr], got rank "
                + String(rank)
            )
    
    fn write_to[W: Writer](self, mut writer: W) -> None:
        self.tensor.write_to(writer)


struct MatrixProductOperator[dtype: DType](Writable, Movable, ImplicitlyCopyable):
    """Matrix Product Operator (MPO) representation of quantum operators.
    
    An MPO represents an operator on a many-body Hilbert space as a network
    of local tensors. Each site tensor has shape [Wl, d_in, d_out, Wr]:
    - Wl, Wr: left/right virtual bond dimensions (operator space)
    - d_in: input physical dimension (acts on ket)
    - d_out: output physical dimension (produces new ket)
    
    For Hermitian operators acting on real states, d_in == d_out.
    
    Typical usage:
        ```mojo
        # Create a Heisenberg Hamiltonian MPO
        var mpo = create_heisenberg_mpo(ctx, num_sites=10, J=1.0)
        mpo.describe()
        ```
    """
    var sites: List[MPOSite[dtype]]
    var physical_in_dim: Int
    var physical_out_dim: Int
    var length: Int
    var bond_dims: List[Int]  # Length = num_sites + 1, bond_dims[i] = W_i (between site i-1 and i)
    
    fn __init__(out self, var sites: List[MPOSite[dtype]]) raises:
        if len(sites) == 0:
            raise Error("MatrixProductOperator requires at least one site tensor")
        
        var first_site: MPOSite[dtype] = sites[0]
        var phys_in = first_site.physical_in_dim()
        var phys_out = first_site.physical_out_dim()
        var bonds: List[Int] = List[Int](capacity=len(sites) + 1)
        bonds.append(first_site.left_bond_dim())
        
        for idx in range(len(sites)):
            var site = sites[idx]
            if site.physical_in_dim() != phys_in:
                raise Error("All MPO sites must have the same physical input dimension")
            if site.physical_out_dim() != phys_out:
                raise Error("All MPO sites must have the same physical output dimension")
            
            if idx > 0:
                var expected = sites[idx - 1].right_bond_dim()
                if site.left_bond_dim() != expected:
                    raise Error(
                        "Bond mismatch between sites "
                        + String(idx - 1)
                        + " and "
                        + String(idx)
                    )
            
            bonds.append(site.right_bond_dim())
        
        self.sites = sites^
        self.physical_in_dim = phys_in
        self.physical_out_dim = phys_out
        self.length = len(self.sites)
        self.bond_dims = bonds^
    
    fn __copyinit__(out self, existing: Self):
        self.sites = existing.sites.copy()
        self.physical_in_dim = existing.physical_in_dim
        self.physical_out_dim = existing.physical_out_dim
        self.length = existing.length
        self.bond_dims = existing.bond_dims.copy()
    
    fn num_sites(self) -> Int:
        return self.length
    
    fn bond_dimension(self, index: Int) -> Int:
        return self.bond_dims[index]
    
    fn site_shape(self, index: Int) -> List[Int]:
        return self.sites[index].shape()
    
    fn describe(self) -> None:
        print(
            "MatrixProductOperator(length=",
            self.length,
            ", physical_in=",
            self.physical_in_dim,
            ", physical_out=",
            self.physical_out_dim,
            ")",
        )
        
        var bond_str = String("  bond dims = [")
        for i in range(len(self.bond_dims)):
            if i > 0:
                bond_str += ", "
            bond_str += String(self.bond_dims[i])
        bond_str += "]"
        print(bond_str)
        
        for idx in range(self.length):
            var shape = self.sites[idx].shape()
            print(
                "  site ",
                idx,
                ": [",
                shape[0],
                ", ",
                shape[1],
                ", ",
                shape[2],
                ", ",
                shape[3],
                "]",
            )
    
    fn write_to[W: Writer](self, mut writer: W) -> None:
        writer.write("MatrixProductOperator(length=")
        writer.write(self.length)
        writer.write(", physical_in=")
        writer.write(self.physical_in_dim)
        writer.write(", physical_out=")
        writer.write(self.physical_out_dim)
        writer.write(", bond_dims=[")
        for i in range(len(self.bond_dims)):
            if i > 0:
                writer.write(", ")
            writer.write(self.bond_dims[i])
        writer.write("], sites=[")
        for idx in range(self.length):
            if idx > 0:
                writer.write(", ")
            var shape = self.sites[idx].shape()
            writer.write("(")
            writer.write(shape[0])
            writer.write(", ")
            writer.write(shape[1])
            writer.write(", ")
            writer.write(shape[2])
            writer.write(", ")
            writer.write(shape[3])
            writer.write(")")
        writer.write("])")


fn create_identity_mpo[dtype: DType = DType.float32](
    ctx: DeviceContext,
    num_sites: Int,
    physical_dim: Int,
) raises -> MatrixProductOperator[dtype]:
    """Create an MPO representing the identity operator.
    
    All bond dimensions are 1 (unentangled operator).
    Each site tensor is the identity matrix reshaped to [1, d, d, 1].
    
    Args:
        ctx: GPU device context.
        num_sites: Number of sites in the chain.
        physical_dim: Local Hilbert space dimension (e.g., 2 for qubits).
    
    Returns:
        MatrixProductOperator representing the identity.
    """
    if num_sites < 1:
        raise Error("num_sites must be >= 1")
    if physical_dim < 1:
        raise Error("physical_dim must be >= 1")
    
    var sites = List[MPOSite[dtype]](capacity=num_sites)
    
    for i in range(num_sites):
        # Shape: [Wl=1, d_in, d_out, Wr=1]
        var shape = List[Int](1, physical_dim, physical_dim, 1)
        var total_size = physical_dim * physical_dim
        var data = List[Scalar[dtype]](capacity=total_size)
        
        # Create identity matrix: I[i,j] = delta_{ij}
        # Layout in memory: (Wl=0) -> row=p_in -> col=p_out -> (Wr=0)
        # Since Wl and Wr are 1, this is just row-major [p_in, p_out]
        for p_in in range(physical_dim):
            for p_out in range(physical_dim):
                if p_in == p_out:
                    data.append(Scalar[dtype](1.0))
                else:
                    data.append(Scalar[dtype](0.0))
        
        var site_tensor = create_dynamic_tensor_from_data[dtype](ctx, data, shape^)
        sites.append(MPOSite[dtype](site_tensor^))
    
    return MatrixProductOperator[dtype](sites^)
