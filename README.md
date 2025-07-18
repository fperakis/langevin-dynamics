# Langevin Dynamics 

A modular and extensible Python package for simulating and analyzing overdamped and underdamped Langevin dynamics with stochastic switching between different frictional states. Includes tools for computing mean-squared displacement (MSD), non-Gaussian parameter (NGP), intermediate scattering functions (ISF), overlap functions, and four-point dynamical susceptibility (χ₄).

---

## Features

- Semi-implicit and explicit Langevin integrators  
- Two-state switching between liquid-like and glass like friction coefficients 
- Analysis tools: MSD, NGP, ISF, τₐ (alpha-relaxation), χ₄  
- Visualization of dynamic heterogeneity and local diffusion  
- Stretched exponential fitting for relaxation dynamics  
- Modular code structure with examples

---
## Installation 
```
pip install git+https://github.com/fperakis/langevin-dynamics.git
```
---

## Getting started

Run a basic Langevin simulation:

```
from langevin.simulation import run_langevin_simulation

positions, velocities, gamma_record = run_langevin_simulation(
    T=1.0,           # temperature
    tau=1.0,         # residence time in each state
    n_steps=100000,  # number of time steps
    n_particles=100, # number of independent particles (no crossterm)
    dt=0.01,         # time step
    gamma_l=1.0,     # friction coeff. of liquid-like state 
    gamma_g=100.0    # friction coeff. of glass-like state 
)
```

Compute and plot the MSD

```
from langevin.analysis import compute_msd
import matplotlib.pyplot as plt

time, msd = compute_msd(positions, dt=0.01)

plt.loglog(time, msd)
plt.xlabel("Time")
plt.ylabel("MSD")
plt.show()
```

## Example Analyses

- Non-Gaussian Parameter (NGP): Quantifies deviations from Gaussian displacement statistics.
- Intermediate Scattering Function (ISF): Direction-averaged ISF for extracting τα and cage dynamics.
- Overlap Function & χ₄ Susceptibility: Tools for identifying dynamical heterogeneity and cooperative motion.

 --

## Project structure
```
 langevin/
├── simulation.py      # Core Langevin integrators
├── analysis.py        # MSD, NGP, ISF, etc.
├── fitting.py         # Stretched exponential and multi-component fits
└── visualization.py   # Visualization tools
```

⸻

## License

MIT License

⸻

## Acknowledgments

Developed by F. Perakis and collaborators at the Institute of Molecular Science (IMS), Japan, and Stockholm University, Sweden.
Inspired by research on stochastic dynamics, glassy systems, and molecular simulations.

Feel free to open an issue or pull request for feedback or contributions.
 
