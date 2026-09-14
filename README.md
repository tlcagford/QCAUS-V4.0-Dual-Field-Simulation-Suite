# QCAUS V4.0 Dual-Field Simulation Suite

GPU-accelerated 3D pseudo-spectral fluid dynamics solver and multi-galaxy parameter optimization pipeline for analyzing Tony Eugene Ford's Vector-Portal Dark Matter models.

## Architecture Components
* `qcaus_suite.py`: 3D split-step fast-Fourier transform (FFT) solver executing on CUDA VRAM.
* `qcaus_analytics.py`: Automated web scraper and multi-parameter MCMC-like database fitter.
* `main.py`: Top-level validation and pipeline orchestration wrapper.

## How to Run Locally
Ensure you have PyTorch, NumPy, and SciPy installed, then execute:
```bash
python main.py
``
Duel License see 
