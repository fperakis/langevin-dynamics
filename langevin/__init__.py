"""
Langevin Dynamics Toolkit

Provides simulation and analysis tools for Langevin dynamics with
state-dependent friction. Supports symmetric and temperature-dependent switching.

Modules:
- simulation: Langevin integrators (explicit, semi-implicit)
- analysis: MSD, NGP, ISF, overlap function, chi4, etc.
- fitting: stretched exponential and multi-component model fitting
- plotting: visualization tools
"""

from .simulation import run_langevin_simulation
from .analysis import (
    compute_msd,
    compute_ngp_ensemble_time_avg,
    compute_intermediate_scattering_function,
    compute_overlap_function,
    compute_chi4
)
from .fitting import (
    fit_stretched_exponential,
    fit_two_step_relaxation
)
from .visualization import (
    plot_trajectory_colored_by_diffusion,
    compute_local_diffusion
)

__all__ = [
    "run_langevin_simulation",
    "run_langevin_simulation_symmetric",
    "compute_msd",
    "compute_ngp_ensemble_time_avg",
    "compute_intermediate_scattering_function",
    "compute_overlap_function",
    "compute_chi4",
    "fit_stretched_exponential",
    "fit_two_step_relaxation",
    "plot_trajectory_colored_by_diffusion",
    "plot_isf_fit"
]

__version__ = "0.1.0"
