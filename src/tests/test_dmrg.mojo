"""Example: DMRG for Transverse-Field Ising Model

This example demonstrates how to use the DMRG implementation to find
the ground state of a 1D transverse-field Ising model.

Hamiltonian: H = -J * sum_i (Z_i Z_{i+1}) - h * sum_i X_i

For J=1.0, h=0.5, the system is in the ferromagnetic phase.
"""

from testing import TestSuite, assert_true
from gpu.host import DeviceContext
from src.state.mps_state import create_product_mps
from src.state.hamiltonians import create_transverse_ising_mpo
from src.algorithms.dmrg import dmrg_two_site, DMRGParams


fn test_dmrg_ising() raises:
    """Run DMRG on transverse-field Ising model."""
    
    print("=" * 60)
    print("DMRG Example: Transverse-Field Ising Model")
    print("=" * 60)
    
    with DeviceContext() as ctx:
        # System parameters
        var num_sites = 10
        var physical_dim = 2  # Qubits
        
        # Hamiltonian parameters
        var J = 1.0   # Ferromagnetic coupling
        var h = 0.5   # Transverse field
        
        print("\nSystem setup:")
        print("  Number of sites:", num_sites)
        print("  Coupling J:", J)
        print("  Field h:", h)
        
        # Create Hamiltonian MPO
        print("\nCreating Hamiltonian MPO...")
        var H = create_transverse_ising_mpo(ctx, num_sites, J, h)
        H.describe()
        
        # Create initial state (product state |000...0>)
        print("\nCreating initial MPS (product state)...")
        var basis = List[Int](capacity=num_sites)
        for _ in range(num_sites):
            basis.append(0)  # All spins up (|0> state)
        
        var psi_initial = create_product_mps(ctx, physical_dim, basis^)
        psi_initial.describe()
        
        # Set DMRG parameters
        var params = DMRGParams(
            num_sweeps=10,
            chi_max=50,
            eps_trunc=1e-10,
            max_krylov_iter=20,
            krylov_tol=1e-8,
            energy_tol=1e-6,
            two_site=True,
            reorthogonalize=True,
            verbose=True
        )
        
        print("\nDMRG Parameters:")
        print("  Max sweeps:", params.num_sweeps)
        print("  Max bond dimension:", params.chi_max)
        print("  Truncation threshold:", params.eps_trunc)
        print("  Energy convergence:", params.energy_tol)
        
        # Run DMRG
        print("\n" + "=" * 60)
        print("Running DMRG optimization...")
        print("=" * 60)
        
        var result = dmrg_two_site(ctx, H^, psi_initial^, params)
        var ground_energy = result[0]
        var ground_state = result[1]
        
        # Display results
        print("\n" + "=" * 60)
        print("DMRG Results")
        print("=" * 60)
        print("Ground state energy:", ground_energy)
        print("Ground state energy per site:", ground_energy / Float64(num_sites))
        
        print("\nFinal MPS structure:")
        ground_state.describe()
        
        print("\n" + "=" * 60)
        print("Example completed successfully!")
        print("=" * 60)

        # Simple assertion: Energy should be negative and reasonably close to expected ground state
        # For N=10, J=1, h=0.5, E_gs is approximately -10.6
        assert_true(ground_energy < -10.0, "Energy should be less than -10.0")
        assert_true(ground_energy > -11.0, "Energy should be greater than -11.0")

fn main():
    try:
        TestSuite.discover_tests[__functions_in_module()]().run()
    except e:
        print("Tests failed: " + String(e))
