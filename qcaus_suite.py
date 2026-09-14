import os
import torch
import numpy as np
import torch.fft as tfft

class QCAUS3DPyTorchSuite:
    def __init__(self, grid_size=64, box_size=10.0, device="cuda"):
        """GPU-Accelerated 3D Pseudo-Spectral Solver Core"""
        self.N = grid_size
        self.L = box_size
        self.dx = box_size / grid_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        x = torch.linspace(-self.L/2, self.L/2, self.N, device=self.device)
        self.X, self.Y, self.Z = torch.meshgrid(x, x, x, indexing='ij')
        self.r2 = self.X**2 + self.Y**2 + self.Z**2
        
        k = torch.fft.fftfreq(self.N, d=self.dx, device=self.device) * 2 * np.pi
        self.KX, self.KY, self.KZ = torch.meshgrid(k, k, k, indexing='ij')
        self.K2 = torch.clamp(self.KX**2 + self.KY**2 + self.KZ**2, min=1.0)

    def init_soliton_tensors(self, omega=0.5):
        amp = torch.exp(-self.r2 / 2.0)
        self.psi_t = amp.to(torch.complex128)
        self.psi_d = (amp * torch.complex(torch.cos(0.1*self.X), torch.sin(0.1*self.X))).to(torch.complex128)

    def solve_poisson_gpu(self, rho):
        return torch.real(tfft.ifftn(-4.0 * np.pi * tfft.fftn(rho) / self.K2))

    def step_gpu(self, dt, omega=0.5, epsilon=1e-3, m22=1.0):
        rho_tot = torch.abs(self.psi_t)**2 + torch.abs(self.psi_d)**2 + 2*omega*torch.real(torch.conj(self.psi_t)*self.psi_d)
        k_t = torch.exp(-1j*(dt/2.0)*self.solve_poisson_gpu(rho_tot+epsilon*torch.abs(self.psi_d)**2))
        k_d = torch.exp(-1j*(dt/2.0)*self.solve_poisson_gpu(rho_tot+epsilon*torch.abs(self.psi_t)**2))
        
        self.psi_t, self.psi_d = self.psi_t*k_t, self.psi_d*k_d
        self.psi_t = tfft.ifftn(tfft.fftn(self.psi_t)*torch.exp(-1j*dt*self.K2/(2.0*m22)))
        self.psi_d = tfft.ifftn(tfft.fftn(self.psi_d)*torch.exp(-1j*dt*self.K2/(2.0*m22)))
        self.psi_t, self.psi_d = self.psi_t*k_t, self.psi_d*k_d
        print("--> 3D spectral step advanced successfully.")
