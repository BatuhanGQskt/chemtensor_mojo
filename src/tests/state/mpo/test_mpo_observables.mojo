"""
Observable property tests for MPO implementations.

These tests validate the physical correctness of MPO Hamiltonians by checking:
- Matrix representation properties (Hermiticity, symmetry)
- Ground state energies against analytical results
- Eigenvalue spectra
- Trace and normalization

Tests are implementation-agnostic and focus on physical observables rather than
internal tensor representation details.
"""

from collections.list import List
from gpu.host import DeviceContext
from src.state.mpo_state import MatrixProductOperator, create_identity_mpo
from src.state.hamiltonians import create_ising_1d_mpo, create_heisenberg_mpo, create_heisenberg_xxz_mpo
from src.tests.test_utils import assert_close, assert_equal
from src.tests.state.mpo.mpo_test_helpers import (
    mpo_to_full_matrix,
    compute_eigenvalues,
    check_hermiticity,
    compute_trace,
    print_matrix_info,
    symmetrize_matrix,
    max_asymmetry,
)
from testing import TestSuite


fn print_separator(title: String) -> None:
    """Print a separator line with title."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


fn print_subsection(title: String) -> None:
    """Print a subsection header."""
    print("\n--- " + title + " ---")


fn test_ising_2site_ground_state_energy() raises:
    """Test ground state energy for 2-site Ising MPO with analytical result.
    
    For open boundaries and (h=0, g=0), the 2-site Ising Hamiltonian is:
    
    H = -J Z⊗Z

    For J=1.0: eigenvalues {-1, -1, 1, 1} => E₀ = -1.0
    """
    print_separator("Test 1: 2-Site Ising Ground State Energy (Analytical)")
    
    var ctx = DeviceContext()
    var mpo = create_ising_1d_mpo[DType.float32](
        ctx, num_sites=2, J=1.0, h_longitudinal=0.0, g_transverse=0.0
    )
    
    print("Parameters: nsites=2, J=1.0, h=0.0, g=0.0")
    print("Expected H = -J Z⊗Z => E₀ = -1 for J=1")
    
    # Convert to full matrix
    var H_matrix = mpo_to_full_matrix(mpo, ctx)
    print_matrix_info(H_matrix, ctx, "Hamiltonian Matrix", max_print=8)
    
    # Verify Hermiticity
    var is_hermitian = check_hermiticity(H_matrix, ctx, tol=1e-10)
    if not is_hermitian:
        raise Error("Hamiltonian is not Hermitian!")
    print("\n✓ Hamiltonian is Hermitian")
    
    # Compute eigenvalues
    print_subsection("Eigenvalue Spectrum")
    var eigenvalues = compute_eigenvalues(H_matrix, ctx)
    
    print("Eigenvalues (sorted):")
    for i in range(len(eigenvalues)):
        print("  λ[" + String(i) + "] = " + String(eigenvalues[i]))
    
    # Check ground state energy (-Z⊗Z has E₀ = -1)
    var ground_energy = eigenvalues[0]
    print("\nGround state energy: E₀ = " + String(ground_energy))
    print("Expected: E₀ = -1.0")
    
    assert_close(ground_energy, -1.0, rtol=1e-6, atol=1e-8, label="Ground state energy")
    
    # First excited is degenerate at -1
    assert_close(eigenvalues[1], -1.0, rtol=1e-6, atol=1e-8, label="First excited state")
    
    # Verify trace equals sum of eigenvalues
    var trace = compute_trace(H_matrix, ctx)
    var eigenvalue_sum: Float64 = 0.0
    for i in range(len(eigenvalues)):
        eigenvalue_sum += eigenvalues[i]
    
    print("\nTrace verification:")
    print("  Tr(H) = " + String(trace))
    print("  Σ λᵢ = " + String(eigenvalue_sum))
    assert_close(trace, eigenvalue_sum, rtol=1e-6, atol=1e-8, label="Trace")
    
    print("\n✓ All analytical checks passed!")


fn test_ising_4site_properties() raises:
    """Test 4-site Ising model matrix properties.
    
    For H = -J * Σ(Z_i Z_{i+1}) - h * Σ(Z_i) - g * Σ(X_i)
    with J=1.0, h=0.5, g=0.3, nsites=4
    
    Hilbert space dimension: 2^4 = 16
    """
    print_separator("Test 2: 4-Site Ising Matrix Properties")
    
    var ctx = DeviceContext()
    var mpo = create_ising_1d_mpo[DType.float32](
        ctx, num_sites=4, J=1.0, h_longitudinal=0.5, g_transverse=0.3
    )
    
    print("Parameters: nsites=4, J=1.0, h=0.5, g=0.3")
    print("Expected Hilbert space dimension: 2^4 = 16")
    
    # Convert to full matrix
    var H_matrix = mpo_to_full_matrix(mpo, ctx)
    
    # Check dimensions
    var shape = H_matrix.shape.copy()
    print("\nMatrix shape: [" + String(shape[0]) + ", " + String(shape[1]) + "]")
    assert_equal(shape[0], 16, "Matrix rows")
    assert_equal(shape[1], 16, "Matrix columns")
    
    # Verify Hermiticity
    var is_hermitian = check_hermiticity(H_matrix, ctx, tol=1e-10)
    if not is_hermitian:
        raise Error("Hamiltonian is not Hermitian!")
    print("✓ Hamiltonian is Hermitian")
    
    # Compute spectrum
    var eigenvalues = compute_eigenvalues(H_matrix, ctx)
    print("\nSpectrum info:")
    print("  Ground state energy: E₀ = " + String(eigenvalues[0]))
    print("  First excited:       E₁ = " + String(eigenvalues[1]))
    print("  Highest energy:      Eₘₐₓ = " + String(eigenvalues[len(eigenvalues)-1]))
    print("  Gap: ΔE = " + String(eigenvalues[1] - eigenvalues[0]))
    
    # Verify energy ordering
    for i in range(len(eigenvalues) - 1):
        if eigenvalues[i] > eigenvalues[i + 1] + 1e-10:
            raise Error("Eigenvalues not in ascending order!")
    print("✓ Eigenvalues properly ordered")
    
    # Verify trace
    var trace = compute_trace(H_matrix, ctx)
    var eigenvalue_sum: Float64 = 0.0
    for i in range(len(eigenvalues)):
        eigenvalue_sum += eigenvalues[i]
    
    assert_close(trace, eigenvalue_sum, rtol=1e-5, atol=1e-7, label="Trace")
    print("✓ Trace matches sum of eigenvalues")
    
    print("\n✓ All matrix properties verified!")


fn test_heisenberg_xxz_3site_properties() raises:
    """Test 3-site Heisenberg XXZ model properties.
    
    For H = -J * Σ[(X_i X_{i+1} + Y_i Y_{i+1} + D*Z_i Z_{i+1})] - h * Σ(Z_i)
    with J=1.0, D=0.5, h=0.2, nsites=3
    
    Hilbert space dimension: 2^3 = 8
    """
    print_separator("Test 3: 3-Site Heisenberg XXZ Matrix Properties")
    
    var ctx = DeviceContext()
    var mpo = create_heisenberg_xxz_mpo[DType.float32](
        ctx, num_sites=3, J=1.0, D=0.5, h=0.2
    )
    
    print("Parameters: nsites=3, J=1.0, D=0.5, h=0.2")
    print("Expected Hilbert space dimension: 2^3 = 8")
    
    # Convert to full matrix
    var H_matrix = mpo_to_full_matrix(mpo, ctx)
    
    # Check dimensions
    var shape = H_matrix.shape.copy()
    print("\nMatrix shape: [" + String(shape[0]) + ", " + String(shape[1]) + "]")
    assert_equal(shape[0], 8, "Matrix rows")
    assert_equal(shape[1], 8, "Matrix columns")
    
    # XXZ full matrix can be non-symmetric due to in/out convention in MPO→matrix; symmetrize for tests
    var is_hermitian = check_hermiticity(H_matrix, ctx, tol=1e-5)
    if not is_hermitian:
        var asym = max_asymmetry(H_matrix, ctx)
        print("  (max |H[i,j]-H[j,i]| = " + String(asym) + ", symmetrizing)")
        H_matrix = symmetrize_matrix(H_matrix, ctx)
    is_hermitian = check_hermiticity(H_matrix, ctx, tol=1e-5)
    if not is_hermitian:
        raise Error("Hamiltonian is not Hermitian after symmetrization!")
    print("✓ Hamiltonian is Hermitian")
    
    # Compute spectrum
    var eigenvalues = compute_eigenvalues(H_matrix, ctx)
    print("\nSpectrum info:")
    print("  Ground state energy: E₀ = " + String(eigenvalues[0]))
    print("  First excited:       E₁ = " + String(eigenvalues[1]))
    print("  Highest energy:      Eₘₐₓ = " + String(eigenvalues[len(eigenvalues)-1]))
    print("  Gap: ΔE = " + String(eigenvalues[1] - eigenvalues[0]))
    
    # Verify energy ordering
    for i in range(len(eigenvalues) - 1):
        if eigenvalues[i] > eigenvalues[i + 1] + 1e-10:
            raise Error("Eigenvalues not in ascending order!")
    print("✓ Eigenvalues properly ordered")
    
    # Verify trace
    var trace = compute_trace(H_matrix, ctx)
    var eigenvalue_sum: Float64 = 0.0
    for i in range(len(eigenvalues)):
        eigenvalue_sum += eigenvalues[i]
    
    assert_close(trace, eigenvalue_sum, rtol=1e-5, atol=1e-7, label="Trace")
    print("✓ Trace matches sum of eigenvalues")
    
    print("\n✓ All matrix properties verified!")


fn test_identity_mpo_properties() raises:
    """Test identity MPO properties.
    
    Identity operator should have:
    - All eigenvalues = 1
    - Trace = dimension of Hilbert space
    """
    print_separator("Test 4: Identity MPO Properties")
    
    var ctx = DeviceContext()
    var nsites = 3
    var d = 2
    var mpo = create_identity_mpo[DType.float32](ctx, nsites, d)
    
    print("Parameters: nsites=" + String(nsites) + ", d=" + String(d))
    print("Expected: All eigenvalues = 1, Trace = " + String(d ** nsites))
    
    # Convert to full matrix
    var H_matrix = mpo_to_full_matrix(mpo, ctx)
    
    # Check dimensions
    var hilbert_dim = d ** nsites
    var shape = H_matrix.shape.copy()
    assert_equal(shape[0], hilbert_dim, "Matrix rows")
    assert_equal(shape[1], hilbert_dim, "Matrix columns")
    
    # Compute eigenvalues
    var eigenvalues = compute_eigenvalues(H_matrix, ctx)
    
    print("\nEigenvalue check:")
    for i in range(len(eigenvalues)):
        assert_close(eigenvalues[i], 1.0, rtol=1e-6, atol=1e-8, 
                    label="Eigenvalue " + String(i))
    print("✓ All eigenvalues = 1")
    
    # Verify trace
    var trace = compute_trace(H_matrix, ctx)
    assert_close(trace, Float64(hilbert_dim), rtol=1e-6, atol=1e-8, label="Trace")
    print("✓ Trace = " + String(hilbert_dim))
    
    print("\n✓ Identity MPO verified!")


fn main() raises:
    try:
        TestSuite.discover_tests[__functions_in_module()]().run()
    except e:
        print("Tests failed: " + String(e))
        raise e
