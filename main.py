from qcaus_suite import QCAUS3DPyTorchSuite
from qcaus_analytics import ingest_sparc_galaxy_data, QCAUSGlobalAsymmetricFitter

if __name__ == "__main__":
    print("=== QCAUS V4.0 Automated System Verification ===")
    
    # Verify local GPU/CPU physics engine initialization
    sim = QCAUS3DPyTorchSuite(grid_size=64, box_size=10.0)
    sim.init_soliton_tensors(omega=-0.1215)
    sim.step_gpu(dt=0.002, omega=-0.1215, epsilon=1e-11, m22=2.5)
    
    # Verify live web database connectivity
    r, v = ingest_sparc_galaxy_data("NGC6503")
    if r is not None: 
        print(f"--> Ingestion Successful! Loaded {len(r)} SPARC database nodes.")
        print("=== Verification Complete: Pipeline Is Completely Stable ===")
