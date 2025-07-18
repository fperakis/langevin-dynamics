"""
Langevin Dynamics Toolkit

Provides simulation and analysis tools for Langevin dynamics with
state-dependent friction. 

Modules:
- simulation: Langevin integrators (explicit, semi-implicit)
- analysis: MSD, NGP, ISF, overlap function, chi4, etc.
- fitting: stretched exponential and multi-component model fitting
- visualization: tools for plotting trajectories
"""

from .simulation import run_langevin_simulation
from .analysis import (
    compute_msd,
    estimate_diffusion_from_msd,
    compute_local_diffusion
    compute_ngp,
    compute_isf,
    compute_overlap_function,
    compute_chi4_overlap
)
from .fitting import (
    stretched_exponential,
    double_exponential,
    two_step_stretched, 
    VFT_gamma, 
    log_weighted_residuals, 
    fit_model
)
from .visualization import (
    plot_trajectory_colored_by_diffusion
)

__all__ = [
    "run_langevin_simulation",
    "compute_msd",
    "estimate_diffusion_from_msd",
    "compute_local_diffusion",
    "compute_ngp",
    "compute_isf",
    "compute_overlap_function",
    "compute_chi4_overlap", 
    "stretched_exponential",
    "double_exponential",
    "two_step_stretched", 
    "VFT_gamma", 
    "log_weighted_residuals", 
    "fit_model",
    "plot_trajectory_colored_by_diffusion"
]

__version__ = "0.1.0"
