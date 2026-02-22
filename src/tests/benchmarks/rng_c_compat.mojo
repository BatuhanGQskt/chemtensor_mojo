"""
C-compatible RNG for benchmark parity without exporting MPS data.

Replicates ChemTensor C rng.c + pcg_basic.c: PCG32 seeding and randnf()
so that the same seed produces the same sequence in Mojo and C.
Used by bench_contractions to build identical random MPS at any scale.
"""
from collections.list import List
from math import sqrt, log, sin
from gpu.host import DeviceContext
from src.state.mps_state import MatrixProductState, MPSSite
from src.m_tensor.dense_tensor import create_dense_tensor_from_data

alias PCG32_MULT: UInt64 = 6364136223846793005

@fieldwise_init
struct Pcg32StepResult(Copyable, Movable, ImplicitlyCopyable):
    var state: UInt64
    var value: UInt32

fn _pcg32_step(state: UInt64, inc: UInt64) -> Pcg32StepResult:
    """One PCG32 step: returns new_state and random_u32."""
    var new_state = state * PCG32_MULT + inc
    var xorshifted = ((state >> 18) ^ state) >> 27
    var rot = state >> 59
    var u = UInt32(xorshifted)
    var r = UInt32(rot)
    var neg_rot = UInt32((-Int32(rot)) & 31)
    var out_u32 = (u >> r) | (u << neg_rot)
    return Pcg32StepResult(new_state, out_u32)


@fieldwise_init
struct Pcg32Random(Copyable, Movable, ImplicitlyCopyable):
    var state: UInt64
    var inc: UInt64

    fn random_u32(self) -> Pcg32RandomResult:
        var result = _pcg32_step(self.state, self.inc)
        return Pcg32RandomResult(Pcg32Random(result.state, self.inc), result.value)


@fieldwise_init
struct Pcg32RandomResult(Copyable, Movable, ImplicitlyCopyable):
    var next: Pcg32Random
    var value: UInt32


fn pcg32_seed(initstate: UInt64, initseq: UInt64) -> Pcg32Random:
    """C-compatible pcg32_srandom_r: returns seeded Pcg32Random (no __init__)."""
    var inc = (initseq << 1) | 1
    var state: UInt64 = 0
    var step1 = _pcg32_step(state, inc)
    state = step1.state + initstate
    var step2 = _pcg32_step(state, inc)
    return Pcg32Random(step2.state, inc)


@fieldwise_init
struct RngStateCCompat(Copyable, Movable, ImplicitlyCopyable):
    """State for C-compatible RNG (pcg32x2; we only use gen0 for randnf)."""
    var gen0: Pcg32Random
    var gen1: Pcg32Random

    fn rand_uint32(self) -> RngStateU32Result:
        var r0 = self.gen0.random_u32()
        return RngStateU32Result(RngStateCCompat(r0.next, self.gen1), r0.value)

    fn rand_uint64(self) -> RngStateU64Result:
        """C-compatible pcg32x2_random_r: combines gen0 and gen1 into a 64-bit value."""
        var r0 = self.gen0.random_u32()
        var r1 = self.gen1.random_u32()
        var val = (UInt64(r0.value) << 32) | UInt64(r1.value)
        return RngStateU64Result(RngStateCCompat(r0.next, r1.next), val)

    fn rand_interval(self, bound: UInt64) -> RngStateU64Result:
        """C-compatible pcg32x2_boundedrand_r: uniform random in [0, bound)."""
        var threshold = (~bound + 1) % bound  # unsigned -bound % bound
        var state = self
        while True:
            var r = state.rand_uint64()
            state = r.next
            if r.value >= threshold:
                return RngStateU64Result(state, r.value % bound)

    fn rand_choice_advance(self, bound: Int, num_samples: Int) -> RngStateCCompat:
        """Advance RNG state exactly as C rand_choice (Floyd's algorithm).

        Replicates the RNG consumption pattern of C's rand_choice() without
        storing results (quantum numbers are all zero when qsite=0)."""
        var state = self
        for i in range(num_samples):
            var j = UInt64(bound - num_samples + 1 + i)
            var result = state.rand_interval(j)
            state = result.next
        return state

    fn randnf(self) -> RngRandnfResult:
        """Uniform [0,1) then Box-Muller for standard normal. Matches C randnf()."""
        var inv_2p32: Float32 = 2.32830644e-10
        var r1 = self.rand_uint32()
        var u1 = Float32(r1.value) * inv_2p32
        var r2 = r1.next.rand_uint32()
        var u2 = Float32(r2.value) * inv_2p32
        if u1 == 0.0:
            u1 = 1.0
        var g = sqrt(-2.0 * log(u1)) * sin(6.2831853071795864769 * u2)
        return RngRandnfResult(r2.next, Float32(g))


@fieldwise_init
struct RngStateU32Result(Copyable, Movable, ImplicitlyCopyable):
    var next: RngStateCCompat
    var value: UInt32


@fieldwise_init
struct RngStateU64Result(Copyable, Movable, ImplicitlyCopyable):
    var next: RngStateCCompat
    var value: UInt64


@fieldwise_init
struct RngRandnfResult(Copyable, Movable, ImplicitlyCopyable):
    var next: RngStateCCompat
    var value: Float32


fn seed_rng_state(seed: UInt64) -> RngStateCCompat:
    """C-compatible seed_rng_state: returns seeded RngStateCCompat (no __init__)."""
    var mult: UInt64 = 6364136223846793005
    var inc: UInt64 = 1442695040888963407
    var rng_seed = pcg32_seed(seed, mult * seed + inc)
    var r1 = rng_seed.random_u32()
    var r2 = r1.next.random_u32()
    var seed1 = (UInt64(r1.value) << 32) | UInt64(r2.value)
    var r3 = r2.next.random_u32()
    var r4 = r3.next.random_u32()
    var seed2 = (UInt64(r3.value) << 32) | UInt64(r4.value)
    var r5 = r4.next.random_u32()
    var r6 = r5.next.random_u32()
    var seq1 = (UInt64(r5.value) << 32) | UInt64(r6.value)
    var r7 = r6.next.random_u32()
    var r8 = r7.next.random_u32()
    var seq2 = (UInt64(r7.value) << 32) | UInt64(r8.value)
    var mask: UInt64 = 0x7FFF_FFFF_FFFF_FFFF
    if (seq1 & mask) == (seq2 & mask):
        seq2 = ~seq2
    return RngStateCCompat(pcg32_seed(seed1, seq1), pcg32_seed(seed2, seq2))


fn bond_dims_c_style(nsites: Int, d: Int, chi_max: Int) -> List[Int]:
    """Bond dimensions matching C construct_random_mps with qsite=0, qnum_sector=0.
    Left half: b[0]=1, b[l]=min(b[l-1]*d, chi_max) for l=1..mid-1.
    Right half: b[nsites]=1, b[l]=min(b[l+1]*d, chi_max) for l=nsites-1..mid."""
    var b = List[Int](capacity=nsites + 1)
    var mid = (nsites + 1) // 2
    b.append(1)
    for l in range(1, mid):
        var next_dim = b[l - 1] * d
        b.append(min(next_dim, chi_max))
    # Right half: build from right end. right_end[0]=1, right_end[i]=min(right_end[i-1]*d, chi_max).
    var right_end = List[Int](capacity=nsites - mid + 1)
    right_end.append(1)
    for i in range(1, nsites - mid + 1):
        var next_dim = right_end[i - 1] * d
        right_end.append(min(next_dim, chi_max))
    # b[mid..nsites] = right_end[nsites-mid], right_end[nsites-mid-1], ..., right_end[0] (so last is 1)
    for i in range(nsites - mid, -1, -1):
        b.append(right_end[i])
    return b^


fn create_random_mps_c_compatible[dtype: DType = DType.float32](
    ctx: DeviceContext,
    nsites: Int,
    d: Int,
    chi_max: Int,
    seed: UInt64,
) raises -> MatrixProductState[dtype]:
    """Build random MPS with same bond dims and RNG as C construct_random_mps.

    Replicates the exact C bond-construction algorithm including rand_choice()
    calls that advance the RNG state when dim_full > max_vdim.  This ensures
    the subsequent randnf() calls for tensor fills produce identical values
    at any nsites/d/chi_max combination (not just when bonds stay below chi_max).
    """
    var rng = seed_rng_state(seed)

    # --- Bond dimension computation (mirrors C construct_random_mps) ----------
    var dim_bonds = List[Int](capacity=nsites + 1)
    # Initialize all entries; boundaries first.
    for _ in range(nsites + 1):
        dim_bonds.append(0)
    dim_bonds[0] = 1
    dim_bonds[nsites] = 1

    var mid = (nsites + 1) // 2

    # Left half: l = 1 .. mid-1
    for l in range(1, mid):
        var dim_full = dim_bonds[l - 1] * d
        dim_bonds[l] = min(dim_full, chi_max)
        if dim_full > chi_max:
            # C calls rand_choice(dim_full, chi_max, rng, ...) here
            rng = rng.rand_choice_advance(dim_full, chi_max)

    # Right half: l = nsites-1 down to mid
    for l in range(nsites - 1, mid - 1, -1):
        var dim_full = dim_bonds[l + 1] * d
        dim_bonds[l] = min(dim_full, chi_max)
        if dim_full > chi_max:
            # C calls rand_choice(dim_full, chi_max, rng, ...) here
            rng = rng.rand_choice_advance(dim_full, chi_max)

    # --- Fill MPS tensors (identical to C block_sparse_tensor_fill_random_normal) ---
    var sites = List[MPSSite[dtype]](capacity=nsites)
    for i in range(nsites):
        var left_dim = dim_bonds[i]
        var right_dim = dim_bonds[i + 1]
        var nelem = left_dim * d * right_dim
        var scale = 1.0 / sqrt(Float64(nelem))
        var data = List[Scalar[dtype]](capacity=nelem)
        for _ in range(nelem):
            var result = rng.randnf()
            rng = result.next
            data.append(Scalar[dtype](result.value * Float32(scale)))
        var shape = List[Int](left_dim, d, right_dim)
        var tensor = create_dense_tensor_from_data[dtype](ctx, data^, shape^)
        sites.append(MPSSite[dtype](tensor^))
    return MatrixProductState[dtype](sites^)
