"""
Sanity checks for MPO / MPS correctness (small-N, dense reference).

Run with:
  mojo run src/tests/test_mpo_mps_sanity.mojo

This test is intentionally "small and exact":
- It checks the local tensor *blocks* produced by `create_ising_1d_mpo` match the
  standard bond-dimension-3 Ising MPO layout.
- It contracts the MPO to a dense matrix for N <= 6 and compares against an
  exact dense Hamiltonian generated directly in the computational basis.
- It contracts a product-state MPS to a dense vector and checks it matches the
  expected basis vector (and is normalized).
"""

from sys import has_accelerator
from collections.list import List
from gpu.host import DeviceContext

from src.state.hamiltonians import create_ising_1d_mpo
from src.state.mps_state import create_product_mps, MatrixProductState, MPSSite
from src.state.mpo_state import MatrixProductOperator, MPOSite
from src.m_tensor.dynamic_tensor import DynamicTensor


@fieldwise_init
struct PackedRank4(Movable):
    """Many rank-4 tensors packed into one flat array.

    We pack each site tensor's raw storage (flattened) into `data`, and keep:
    - offsets[site]: starting flat index of that site within `data`
    - s0..s3[site]: strides for that site tensor (matching the original DynamicTensor.stride)
    """
    var data: List[Float64]
    var offsets: List[Int]
    var s0: List[Int]
    var s1: List[Int]
    var s2: List[Int]
    var s3: List[Int]

    fn get(self, site: Int, i0: Int, i1: Int, i2: Int, i3: Int) -> Float64:
        var base = self.offsets[site]
        var idx = base + i0 * self.s0[site] + i1 * self.s1[site] + i2 * self.s2[site] + i3 * self.s3[site]
        return self.data[idx]


@fieldwise_init
struct PackedRank3(Movable):
    """Many rank-3 tensors packed into one flat array."""
    var data: List[Float64]
    var offsets: List[Int]
    var s0: List[Int]
    var s1: List[Int]
    var s2: List[Int]

    fn get(self, site: Int, i0: Int, i1: Int, i2: Int) -> Float64:
        var base = self.offsets[site]
        var idx = base + i0 * self.s0[site] + i1 * self.s1[site] + i2 * self.s2[site]
        return self.data[idx]


fn _bit_msb(state_index: Int, site: Int, nsites: Int) -> Int:
    # Site 0 = most-significant bit.
    return (state_index >> (nsites - 1 - site)) & 1


fn _bit_lsb(state_index: Int, site: Int, nsites: Int) -> Int:
    # Site 0 = least-significant bit.
    return (state_index >> site) & 1


fn _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


fn assert_close(name: String, got: Float64, expected: Float64, tol: Float64) raises -> None:
    var diff = _abs(got - expected)
    if diff > tol:
        raise Error(
            name
            + " mismatch: got="
            + String(got)
            + " expected="
            + String(expected)
            + " |diff|="
            + String(diff)
            + " tol="
            + String(tol)
        )


fn assert_true(name: String, cond: Bool) raises -> None:
    if not cond:
        raise Error("Assertion failed: " + name)


fn pack_rank4_sites_f64(
    sites: List[MPOSite[DType.float64]],
    ctx: DeviceContext,
) raises -> PackedRank4:
    var n = len(sites)
    var offsets = List[Int](capacity=n)
    var s0 = List[Int](capacity=n)
    var s1 = List[Int](capacity=n)
    var s2 = List[Int](capacity=n)
    var s3 = List[Int](capacity=n)

    # First pass: compute total size and record per-site metadata
    var total: Int = 0
    for i in range(n):
        var t = sites[i].tensor
        if len(t.shape) != 4:
            raise Error("Expected rank-4 site tensor, got rank " + String(len(t.shape)))
        offsets.append(total)
        s0.append(t.stride[0])
        s1.append(t.stride[1])
        s2.append(t.stride[2])
        s3.append(t.stride[3])
        total += t.size

    var data = List[Float64](capacity=total)

    # Second pass: copy raw storage into the packed buffer
    for i in range(n):
        var t = sites[i].tensor
        var host = ctx.enqueue_create_host_buffer[DType.float64](t.size)
        ctx.enqueue_copy(host, t.storage)
        ctx.synchronize()
        for k in range(t.size):
            data.append(Float64(host[k]))

    return PackedRank4(data^, offsets^, s0^, s1^, s2^, s3^)


fn pack_rank3_sites_f64(
    sites: List[MPSSite[DType.float64]],
    ctx: DeviceContext,
) raises -> PackedRank3:
    var n = len(sites)
    var offsets = List[Int](capacity=n)
    var s0 = List[Int](capacity=n)
    var s1 = List[Int](capacity=n)
    var s2 = List[Int](capacity=n)

    var total: Int = 0
    for i in range(n):
        var t = sites[i].tensor
        if len(t.shape) != 3:
            raise Error("Expected rank-3 site tensor, got rank " + String(len(t.shape)))
        offsets.append(total)
        s0.append(t.stride[0])
        s1.append(t.stride[1])
        s2.append(t.stride[2])
        total += t.size

    var data = List[Float64](capacity=total)
    for i in range(n):
        var t = sites[i].tensor
        var host = ctx.enqueue_create_host_buffer[DType.float64](t.size)
        ctx.enqueue_copy(host, t.storage)
        ctx.synchronize()
        for k in range(t.size):
            data.append(Float64(host[k]))

    return PackedRank3(data^, offsets^, s0^, s1^, s2^)


fn build_dense_ising_reference(
    nsites: Int,
    J: Float64,
    h: Float64,
    g: Float64,
    msb_first: Bool,
) raises -> List[Float64]:
    # Dense H for:
    #   H = -J Σ Z_i Z_{i+1} - g Σ X_i - h Σ Z_i
    var dim = 1 << nsites
    var H = List[Float64](capacity=dim * dim)
    for _ in range(dim * dim):
        H.append(0.0)

    # Diagonal: ZZ + Z terms
    for ket in range(dim):
        var diag: Float64 = 0.0
        for i in range(nsites):
            var bi = _bit_msb(ket, i, nsites) if msb_first else _bit_lsb(ket, i, nsites)
            var zi: Float64 = 1.0
            if bi == 1:
                zi = -1.0
            diag += -h * zi
        for i in range(nsites - 1):
            var bi = _bit_msb(ket, i, nsites) if msb_first else _bit_lsb(ket, i, nsites)
            var bj = _bit_msb(ket, i + 1, nsites) if msb_first else _bit_lsb(ket, i + 1, nsites)
            var zi: Float64 = 1.0
            var zj: Float64 = 1.0
            if bi == 1:
                zi = -1.0
            if bj == 1:
                zj = -1.0
            diag += -J * zi * zj
        H[ket * dim + ket] = diag

    # Off-diagonal: X flips
    for ket in range(dim):
        for i in range(nsites):
            var flip_mask = (1 << (nsites - 1 - i)) if msb_first else (1 << i)
            var bra = ket ^ flip_mask
            H[bra * dim + ket] += -g

    return H^


fn contract_mpo_to_dense(
    mpo: MatrixProductOperator[DType.float64],
    ctx: DeviceContext,
    msb_first: Bool,
) raises -> List[Float64]:
    var nsites = mpo.num_sites()
    var d_in = mpo.physical_in_dim
    var d_out = mpo.physical_out_dim
    if d_in != 2 or d_out != 2:
        raise Error("This sanity test expects d_in=d_out=2 for Ising MPO")

    # Copy all local MPO tensors to host once (packed).
    var W = pack_rank4_sites_f64(mpo.sites, ctx)

    var dim = 1 << nsites
    var Hd = List[Float64](capacity=dim * dim)
    for _ in range(dim * dim):
        Hd.append(0.0)

    for bra in range(dim):
        for ket in range(dim):
            # DP over MPO bond index
            var wl_dim0 = mpo.bond_dimension(0)
            var dp = List[Float64](capacity=wl_dim0)
            for _ in range(wl_dim0):
                dp.append(0.0)
            dp[0] = 1.0

            for site in range(nsites):
                var wl_dim = mpo.bond_dimension(site)
                var wr_dim = mpo.bond_dimension(site + 1)
                var dp_next = List[Float64](capacity=wr_dim)
                for _ in range(wr_dim):
                    dp_next.append(0.0)

                var s_in = _bit_msb(ket, site, nsites) if msb_first else _bit_lsb(ket, site, nsites)
                var s_out = _bit_msb(bra, site, nsites) if msb_first else _bit_lsb(bra, site, nsites)

                for wl in range(wl_dim):
                    var coeff = dp[wl]
                    if coeff == 0.0:
                        continue
                    for wr in range(wr_dim):
                        # Modified call order: [wl, s_in, s_out, wr]
                        dp_next[wr] += coeff * W.get(site, wl, s_in, s_out, wr)

                dp = dp_next^

            # Right boundary bond dim should be 1
            Hd[bra * dim + ket] = dp[0]

    return Hd^


fn contract_mps_to_dense(
    mps: MatrixProductState[DType.float64],
    ctx: DeviceContext,
    msb_first: Bool,
) raises -> List[Float64]:
    var nsites = mps.num_sites()
    var d = mps.physical_dim
    if d != 2:
        raise Error("This sanity test expects physical_dim=2 for MPS")

    var A = pack_rank3_sites_f64(mps.sites, ctx)

    var dim = 1 << nsites
    var psi = List[Float64](capacity=dim)
    for _ in range(dim):
        psi.append(0.0)

    for ket in range(dim):
        var dl0 = mps.bond_dimension(0)
        var dp = List[Float64](capacity=dl0)
        for _ in range(dl0):
            dp.append(0.0)
        dp[0] = 1.0

        for site in range(nsites):
            var Dl = mps.bond_dimension(site)
            var Dr = mps.bond_dimension(site + 1)
            var dp_next = List[Float64](capacity=Dr)
            for _ in range(Dr):
                dp_next.append(0.0)

            var s = _bit_msb(ket, site, nsites) if msb_first else _bit_lsb(ket, site, nsites)
            for dl in range(Dl):
                var coeff = dp[dl]
                if coeff == 0.0:
                    continue
                for dr in range(Dr):
                    dp_next[dr] += coeff * A.get(site, dl, s, dr)

            dp = dp_next^

            psi[ket] = dp[0]

    return psi^


fn dot_dense(a: List[Float64], b: List[Float64]) raises -> Float64:
    if len(a) != len(b):
        raise Error("dot_dense: length mismatch")
    var acc: Float64 = 0.0
    for i in range(len(a)):
        acc += a[i] * b[i]
    return acc


fn matvec_dense(H: List[Float64], v: List[Float64]) raises -> List[Float64]:
    var dim = len(v)
    if len(H) != dim * dim:
        raise Error("matvec_dense: matrix shape mismatch")
    var out = List[Float64](capacity=dim)
    for _ in range(dim):
        out.append(0.0)
    for i in range(dim):
        var acc: Float64 = 0.0
        for j in range(dim):
            acc += H[i * dim + j] * v[j]
        out[i] = acc
    return out^


fn max_abs_diff(a: List[Float64], b: List[Float64]) raises -> Float64:
    if len(a) != len(b):
        raise Error("max_abs_diff: length mismatch")
    var m: Float64 = 0.0
    for i in range(len(a)):
        var d = _abs(a[i] - b[i])
        if d > m:
            m = d
    return m


fn test_ising_mpo_local_blocks(ctx: DeviceContext) raises -> None:
    var nsites = 6
    var J = 1.0
    var h = 0.0
    var g = 1.0
    var mpo = create_ising_1d_mpo[DType.float64](ctx, nsites, J=J, h_longitudinal=h, g_transverse=g)

    # Shapes / bond dims
    assert_true("mpo.num_sites == nsites", mpo.num_sites() == nsites)
    assert_true("mpo bond left boundary = 1", mpo.bond_dimension(0) == 1)
    assert_true("mpo bond right boundary = 1", mpo.bond_dimension(nsites) == 1)
    for i in range(1, nsites):
        assert_true("mpo bulk bond dim = 3 at i=" + String(i), mpo.bond_dimension(i) == 3)

    # Check key operator blocks match the canonical Ising MPO layout.
    # We only check a few entries per block (enough to detect index/layout bugs).
    #
    # Pauli matrices:
    #   I = [[1,0],[0,1]]
    #   X = [[0,1],[1,0]]
    #   Z = [[1,0],[0,-1]]
    # On-site term: -(g X + h Z)
    var tol = 1e-12

    # First site: shape [1,3,2,2] -> [Wl=1, d=2, d=2, Wr=3]
    var Wpack = pack_rank4_sites_f64(mpo.sites, ctx)
    # I block at W[0,0] (Wl=0, Wr=0)
    # get(site, Wl, d_in, d_out, Wr)
    assert_close("W0[0,0,0,0]", Wpack.get(0, 0, 0, 0, 0), 1.0, tol)
    assert_close("W0[0,1,1,0]", Wpack.get(0, 0, 1, 1, 0), 1.0, tol)
    assert_close("W0[0,0,1,0]", Wpack.get(0, 0, 0, 1, 0), 0.0, tol)
    
    # Z block at W[0,1]
    assert_close("W0[0,0,0,1]", Wpack.get(0, 0, 0, 0, 1), 1.0, tol)
    assert_close("W0[0,1,1,1]", Wpack.get(0, 0, 1, 1, 1), -1.0, tol)
    
    # On-site: -(gX+hZ) with h=0 => -gX at W[0,2]
    assert_close("W0[0,0,1,2]", Wpack.get(0, 0, 0, 1, 2), -g, tol)
    assert_close("W0[0,1,0,2]", Wpack.get(0, 0, 1, 0, 2), -g, tol)
    assert_close("W0[0,0,0,2]", Wpack.get(0, 0, 0, 0, 2), 0.0, tol)

    # One bulk site: shape [3,3,2,2] -> [Wl=3, d=2, d=2, Wr=3]
    # row0: [I, Z, -(gX+hZ)]
    # W[0,0] = I
    assert_close("Wb[0,0,0,0]", Wpack.get(1, 0, 0, 0, 0), 1.0, tol)
    assert_close("Wb[0,1,1,0]", Wpack.get(1, 0, 1, 1, 0), 1.0, tol)
    
    # W[0,1] = Z (row 0, col 1) -> Wl=0, Wr=1
    assert_close("Wb[0,0,0,1]", Wpack.get(1, 0, 0, 0, 1), 1.0, tol)
    assert_close("Wb[0,1,1,1]", Wpack.get(1, 0, 1, 1, 1), -1.0, tol)
    
    # W[1,2] = -J Z (row 1, col 2) -> Wl=1, Wr=2
    assert_close("Wb[1,0,0,2]", Wpack.get(1, 1, 0, 0, 2), -J, tol)
    assert_close("Wb[1,1,1,2]", Wpack.get(1, 1, 1, 1, 2), J, tol)
    
    # W[2,2] = I (row 2, col 2) -> Wl=2, Wr=2
    assert_close("Wb[2,0,0,2]", Wpack.get(1, 2, 0, 0, 2), 1.0, tol)
    assert_close("Wb[2,1,1,2]", Wpack.get(1, 2, 1, 1, 2), 1.0, tol)

    # Last site: shape [3,1,2,2] -> [Wl=3, d=2, d=2, Wr=1]
    # col0: [-(gX+hZ); -JZ; I]
    # W[0,0] = -(gX+hZ) (row 0, col 0) -> Wl=0, Wr=0
    assert_close("WN[0,0,1,0]", Wpack.get(nsites - 1, 0, 0, 1, 0), -g, tol)
    
    # W[1,0] = -J Z (row 1, col 0) -> Wl=1, Wr=0
    assert_close("WN[1,0,0,0]", Wpack.get(nsites - 1, 1, 0, 0, 0), -J, tol)
    
    # W[2,0] = I (row 2, col 0) -> Wl=2, Wr=0
    assert_close("WN[2,1,1,0]", Wpack.get(nsites - 1, 2, 1, 1, 0), 1.0, tol)


fn test_ising_mpo_dense_matches_reference(ctx: DeviceContext) raises -> None:
    var nsites = 6
    var J = 1.0
    var h = 0.0
    var g = 1.0
    var mpo = create_ising_1d_mpo[DType.float64](ctx, nsites, J=J, h_longitudinal=h, g_transverse=g)

    var tol = 1e-12

    # Try both basis orderings; MPO correctness should hold in at least one,
    # depending on the codebase convention for mapping basis index -> site bits.
    var H_ref_msb = build_dense_ising_reference(nsites, J, h, g, msb_first=True)
    var H_ref_lsb = build_dense_ising_reference(nsites, J, h, g, msb_first=False)

    var H_mpo_msb = contract_mpo_to_dense(mpo^, ctx, msb_first=True)
    var H_mpo_lsb = contract_mpo_to_dense(mpo^, ctx, msb_first=False)

    var err_msb = max_abs_diff(H_ref_msb^, H_mpo_msb^)
    var err_lsb = max_abs_diff(H_ref_lsb^, H_mpo_lsb^)

    print("Dense MPO check: err(msb_first)=", err_msb, " err(lsb_first)=", err_lsb)

    if err_msb <= tol:
        return
    if err_lsb <= tol:
        return

    # Neither matched: hard failure.
    raise Error(
        "MPO dense mismatch (both conventions): err(msb_first)="
        + String(err_msb)
        + " err(lsb_first)="
        + String(err_lsb)
    )


fn test_product_mps_dense_vector(ctx: DeviceContext) raises -> None:
    var nsites = 6
    var basis = List[Int](capacity=nsites)
    # |010101>
    for i in range(nsites):
        basis.append(i & 1)
    var psi = create_product_mps[DType.float64](ctx, 2, basis^)

    # For MPS vectorization, try both conventions and accept either (but print which).
    var v_msb = contract_mps_to_dense(psi^, ctx, msb_first=True)
    var v_lsb = contract_mps_to_dense(psi^, ctx, msb_first=False)
    var dim = 1 << nsites

    var tol = 1e-12
    # Expected basis index:
    # - msb_first: bits are appended left-to-right
    # - lsb_first: site0 toggles the least-significant bit
    var expected_msb = 0
    for i in range(nsites):
        expected_msb = (expected_msb << 1) | (i & 1)
    var expected_lsb = 0
    for i in range(nsites):
        expected_lsb = expected_lsb | ((i & 1) << i)

    var ok_msb = True
    for i in range(dim):
        var expected = 0.0
        if i == expected_msb:
            expected = 1.0
        if _abs(v_msb[i] - expected) > tol:
            ok_msb = False
            break

    var ok_lsb = True
    for i in range(dim):
        var expected = 0.0
        if i == expected_lsb:
            expected = 1.0
        if _abs(v_lsb[i] - expected) > tol:
            ok_lsb = False
            break

    print("Dense MPS check: matches(msb_first)=", ok_msb, " matches(lsb_first)=", ok_lsb)

    assert_true("product MPS matches at least one basis convention", ok_msb or ok_lsb)

    if ok_msb:
        var norm2 = dot_dense(v_msb^, v_msb^)
        assert_close("||psi||^2 (msb_first)", norm2, 1.0, tol)
    if ok_lsb:
        var norm2 = dot_dense(v_lsb^, v_lsb^)
        assert_close("||psi||^2 (lsb_first)", norm2, 1.0, tol)


fn test_dense_energy_consistency(ctx: DeviceContext) raises -> None:
    # Optional extra check: for a simple product state, energy computed via dense contraction
    # is consistent with the MPO dense we built.
    var nsites = 6
    var J = 1.0
    var h = 0.0
    var g = 1.0
    var mpo = create_ising_1d_mpo[DType.float64](ctx, nsites, J=J, h_longitudinal=h, g_transverse=g)

    var basis = List[Int](capacity=nsites)
    for _ in range(nsites):
        basis.append(0)  # |000...0>
    var psi = create_product_mps[DType.float64](ctx, 2, basis^)

    # Keep this consistent with whichever convention made the MPO dense check pass.
    var tol = 1e-12

    var H_ref_msb = build_dense_ising_reference(nsites, J, h, g, msb_first=True)
    var H_ref_lsb = build_dense_ising_reference(nsites, J, h, g, msb_first=False)
    var H_mpo_msb = contract_mpo_to_dense(mpo^, ctx, msb_first=True)
    var H_mpo_lsb = contract_mpo_to_dense(mpo^, ctx, msb_first=False)

    var v_msb = contract_mps_to_dense(psi^, ctx, msb_first=True)
    var v_lsb = contract_mps_to_dense(psi^, ctx, msb_first=False)

    var err_msb = max_abs_diff(H_ref_msb^, H_mpo_msb^)
    var err_lsb = max_abs_diff(H_ref_lsb^, H_mpo_lsb^)

    if err_msb <= tol:
        var Hv_ref = matvec_dense(H_ref_msb^, v_msb^)
        var Hv_mpo = matvec_dense(H_mpo_msb^, v_msb^)
        var errHv = max_abs_diff(Hv_ref^, Hv_mpo^)
        if errHv > tol:
            raise Error("Dense matvec mismatch (msb_first): max|H_ref v - H_mpo v| = " + String(errHv))
        var e_ref = dot_dense(v_msb^, Hv_ref^)
        var e_mpo = dot_dense(v_msb^, Hv_mpo^)
        assert_close("<psi|H|psi> ref vs mpo (msb_first)", e_mpo, e_ref, tol)
        return

    if err_lsb <= tol:
        var Hv_ref = matvec_dense(H_ref_lsb^, v_lsb^)
        var Hv_mpo = matvec_dense(H_mpo_lsb^, v_lsb^)
        var errHv = max_abs_diff(Hv_ref^, Hv_mpo^)
        if errHv > tol:
            raise Error("Dense matvec mismatch (lsb_first): max|H_ref v - H_mpo v| = " + String(errHv))
        var e_ref = dot_dense(v_lsb^, Hv_ref^)
        var e_mpo = dot_dense(v_lsb^, Hv_mpo^)
        assert_close("<psi|H|psi> ref vs mpo (lsb_first)", e_mpo, e_ref, tol)
        return

    raise Error("Energy consistency test skipped: MPO dense did not match reference in either convention")


fn main():
    try:
        print("\n" + "#" * 70)
        print("# MPO / MPS SANITY TESTS (Ising, small N)")
        print("#" * 70)
        
        with DeviceContext() as ctx:
            test_ising_mpo_local_blocks(ctx)
            print("✓ MPO local block structure looks correct")

            test_ising_mpo_dense_matches_reference(ctx)
            print("✓ MPO contracts to the correct dense Ising Hamiltonian")

            test_product_mps_dense_vector(ctx)
            print("✓ Product MPS contracts to the expected basis vector")

            test_dense_energy_consistency(ctx)
            print("✓ Dense energy/matvec consistency checks passed")

        print("#" * 70)
        print("# ALL SANITY TESTS PASSED")
        print("#" * 70 + "\n")
    except e:
        print("Tests failed: " + String(e))
