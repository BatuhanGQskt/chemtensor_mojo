from collections.list import List
from gpu.host import DeviceContext
from m_tensor.dynamic_tensor import (
    DynamicTensor,
    create_dynamic_tensor,
    create_dynamic_tensor_from_data,
)


@fieldwise_init
struct MPSSite[dtype: DType](Writable, Movable, ImplicitlyCopyable):
    """Single site tensor inside an MPS.

    Each site is stored as a rank-3 DynamicTensor with layout
    [left_bond, physical, right_bond].
    """
    var tensor: DynamicTensor[dtype]

    fn rank(self) -> Int:
        return len(self.tensor.shape)

    fn shape(self) -> List[Int]:
        return self.tensor.shape.copy()

    fn physical_dim(self) raises -> Int:
        self._assert_rank3()
        return self.tensor.shape[1]

    fn left_bond_dim(self) raises -> Int:
        self._assert_rank3()
        return self.tensor.shape[0]

    fn right_bond_dim(self) raises -> Int:
        self._assert_rank3()
        return self.tensor.shape[2]

    fn _assert_rank3(self) raises -> None:
        var rank = len(self.tensor.shape)
        if rank != 3:
            raise Error(
                "MPSSite expects rank-3 tensors [bond_left, physical, bond_right], got rank "
                + String(rank)
            )

    fn write_to[W: Writer](self, mut writer: W) -> None:
        self.tensor.write_to(writer)


struct MatrixProductState[dtype: DType](Writable, Movable, ImplicitlyCopyable):
    """Matrix Product State built on top of DynamicTensor.

    Typical usage:
        ```mojo
        with DeviceContext() as ctx:
            var basis = List[Int](0, 1, 0)
            var psi = create_product_mps(ctx, 2, basis^)
            psi.describe()  # Prints bond dims and site shapes
        ```
    """
    var sites: List[MPSSite[dtype]]
    var physical_dim: Int
    var length: Int
    var bond_dims: List[Int]

    fn __init__(out self, var sites: List[MPSSite[dtype]]) raises:
        if len(sites) == 0:
            raise Error("MatrixProductState requires at least one site tensor")

        var first_site: MPSSite[dtype] = sites[0]
        var phys_dim = first_site.physical_dim()
        var bonds: List[Int] = List[Int](capacity=len(sites))
        bonds.append(first_site.left_bond_dim())

        for idx in range(len(sites)):
            var site = sites[idx]
            if site.physical_dim() != phys_dim:
                raise Error("All MPS sites must have the same physical dimension")

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
        self.physical_dim = phys_dim
        self.length = len(self.sites)
        self.bond_dims = bonds^

    fn __copyinit__(out self, existing: Self):
        self.sites = existing.sites.copy()
        self.physical_dim = existing.physical_dim
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
            "MatrixProductState(length=",
            self.length,
            ", physical_dim=",
            self.physical_dim,
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
                "]",
            )

    fn write_to[W: Writer](self, mut writer: W) -> None:
        writer.write("MatrixProductState(length=")
        writer.write(self.length)
        writer.write(", physical_dim=")
        writer.write(self.physical_dim)
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
            writer.write(")")
        writer.write("])")

    # TODO: Review this function
    fn left_canonicalize(inout self) raises:
        """Bring the MPS to left-canonical form using QR decomposition.
        
        Sweeps from left to right, orthogonalizing each site and absorbing the R factor to the next.
        Assumes DynamicTensor has qr() -> (Q, R), reshape, contract, and norm methods.
        """
        var ortho_center = 0
        while ortho_center < self.length - 1:
            var site = self.sites[ortho_center]
            var left_dim = site.left_bond_dim()
            var phys_dim = site.physical_dim()
            var right_dim = site.right_bond_dim()
            
            # Reshape to matrix: [left * phys, right]
            var mat = site.tensor.reshape(List[Int](left_dim * phys_dim, right_dim))
            
            # QR decomposition
            var Q, R = mat.qr()
            
            # Normalize Q if needed (optional, but ensures unit norm)
            var q_norm = Q.norm()
            if q_norm > 0:
                Q /= q_norm  # Assume DynamicTensor supports in-place scalar divide
            
            # Reshape Q back to [left, phys, right] (right dim from Q cols)
            var new_right_dim = Q.shape[1]  # May truncate if QR does
            self.sites[ortho_center].tensor = Q.reshape(List[Int](left_dim, phys_dim, new_right_dim))
            self.bond_dims[ortho_center + 1] = new_right_dim  # Update bond dim
            
            # Absorb R into next site: next_tensor = R @ next_tensor (contract on right_dim)
            var next_site = self.sites[ortho_center + 1]
            # Assume contract(axes: List[Tuple[Int, Int]]) or similar; here, R (old_right, new_right) @ next (old_left=old_right, phys, right)
            next_site.tensor = R.contract(next_site.tensor, List[Tuple[Int, Int]]((1, 0)))  # Contract R.col with next.left
            
            # Update next left bond dim
            self.bond_dims[ortho_center + 1] = new_right_dim  # Already done above
            self.sites[ortho_center + 1] = next_site  # If needed, but since inout self, direct assign ok
            
            ortho_center += 1
        
        # Final norm absorption (scale last site)
        var last_site = self.sites[self.length - 1]
        var last_norm = last_site.tensor.norm()
        if last_norm > 0:
            last_site.tensor /= last_norm
        self.sites[self.length - 1] = last_site


fn create_uniform_mps[dtype: DType = DType.float32](
    ctx: DeviceContext,
    num_sites: Int,
    physical_dim: Int,
    bond_dims: List[Int],
    init_value: Optional[Scalar[dtype]] = None,
) raises -> MatrixProductState[dtype]:
    """Allocate an MPS with constant entries in every site tensor."""
    if num_sites < 1:
        raise Error("num_sites must be >= 1")
    if physical_dim < 1:
        raise Error("physical_dim must be >= 1")
    if len(bond_dims) != num_sites + 1:
        raise Error("bond_dims must have length num_sites + 1")

    var sites = List[MPSSite[dtype]](capacity=num_sites)
    for i in range(num_sites):
        var left_dim = bond_dims[i]
        var right_dim = bond_dims[i + 1]
        if left_dim < 1 or right_dim < 1:
            raise Error("Bond dimensions must be >= 1")

        var shape = List[Int](left_dim, physical_dim, right_dim)
        var site_tensor: DynamicTensor[dtype]
        if init_value is None:
            site_tensor = DynamicTensor[dtype].random(ctx, shape^)  # Assume random factory
        else:
            site_tensor = create_dynamic_tensor[dtype](
                ctx, shape^, row_major=True, init_value=init_value.value()
            )
        sites.append(MPSSite[dtype](site_tensor^))
    return MatrixProductState[dtype](sites^)


fn create_product_mps[dtype: DType = DType.float32](
    ctx: DeviceContext,
    physical_dim: Int,
    basis: List[Int],
) raises -> MatrixProductState[dtype]:
    """Create a product-state MPS from a list of local basis choices.

    Args:
        ctx: GPU device context.
        physical_dim: Local Hilbert-space dimension (e.g., 2 for qubits).
        basis: List of integers (length = num_sites) specifying |basis[i]> at each site.

    Returns:
        MatrixProductState where all internal bonds are 1 (unentangled).
    """
    var num_sites = len(basis)
    if num_sites < 1:
        raise Error("basis must contain at least one element")
    if physical_dim < 1:
        raise Error("physical_dim must be >= 1")

    var sites = List[MPSSite[dtype]](capacity=num_sites)
    for i in range(num_sites):
        var choice = basis[i]
        if choice < 0 or choice >= physical_dim:
            raise Error(
                "Invalid basis index "
                + String(choice)
                + " at site "
                + String(i)
                + " (must be in [0, physical_dim))"
            )

        var shape = List[Int](1, physical_dim, 1)
        var total_size = physical_dim
        var data = List[Scalar[dtype]](capacity=total_size)

        for p in range(physical_dim):
            if p == choice:
                data.append(Scalar[dtype](1.0))
            else:
                data.append(Scalar[dtype](0.0))

        var site_tensor = create_dynamic_tensor_from_data[dtype](ctx, data, shape^)
        sites.append(MPSSite[dtype](site_tensor^))

    return MatrixProductState[dtype](sites^)

