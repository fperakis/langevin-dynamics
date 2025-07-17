import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_msd(times, msd, ax=None, label=None, loglog=True):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(times, msd, label=label)
    ax.set_xlabel("Time")
    ax.set_ylabel("MSD")
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    if label:
        ax.legend()
    ax.grid(True)
    return ax


def plot_ngp(times, alpha2, ax=None, label=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(times, alpha2, label=label)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\alpha_2(t)$")
    ax.set_xscale("log")
    ax.grid(True)
    if label:
        ax.legend()
    return ax


def plot_isf(times, isf, ax=None, label=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(times, isf, label=label)
    ax.set_xlabel("Time")
    ax.set_ylabel("ISF")
    ax.set_xscale("log")
    ax.grid(True)
    if label:
        ax.legend()
    return ax


def compute_local_diffusion(pos, dt, window_size):
    """
    Computes local diffusion coefficient from trajectory.

    Parameters:
        pos : array of shape (n_steps, 2)
            2D position trajectory.
        dt : float
            Time step.
        window_size : int
            Size of the time window to compute local diffusion.

    Returns:
        D_local : array of shape (n_steps - window_size,)
            Estimated local diffusion coefficients.
        mid_points : array of shape (n_steps - window_size, 2)
            Corresponding midpoints of the trajectory segments.
    """
    n_steps = len(pos)
    D_local = []
    mid_points = []
    for i in range(n_steps - window_size):
        delta = pos[i + window_size] - pos[i]
        msd = np.sum(delta**2)
        D = msd / (4 * (window_size * dt))  # 2D MSD formula
        D_local.append(D)
        mid = (pos[i + window_size] + pos[i]) / 2
        mid_points.append(mid)
    return np.array(D_local), np.array(mid_points)


def plot_trajectory_colored_by_diffusion(
    pos,
    D_local,
    mid_points,
    vmin=None,
    vmax=None,
    cmap='viridis'):
    """
    Plots trajectory with color coding based on local diffusion.

    Parameters:
        pos : array of shape (n_steps, 2)
            Particle trajectory.
        D_local : array of local diffusion coefficients.
        mid_points : array of shape (n_steps - window_size, 2)
            Midpoints of each local segment.
        vmin : float, optional
            Minimum value for color scale.
        vmax : float, optional
            Maximum value for color scale.
        cmap : str
            Colormap name (default is 'viridis').
    """
    pos_centered = pos - np.mean(pos, axis=0)

    segments = np.array([
        [pos_centered[i], pos_centered[i + 1]]
        for i in range(len(pos_centered) - 1)
    ])

    if len(D_local) < len(segments):
        D_local = np.pad(D_local, (0, len(segments) - len(D_local)), mode='edge')

    lc = LineCollection(segments, cmap=cmap, linewidth=1,
                        norm=plt.Normalize(vmin=vmin, vmax=vmax))
    lc.set_array(D_local)

    fig, ax = plt.subplots(figsize=(4, 5))
    ax.add_collection(lc)

    x_min, x_max = np.min(pos_centered[:, 0]), np.max(pos_centered[:, 0])
    y_min, y_max = np.min(pos_centered[:, 1]), np.max(pos_centered[:, 1])
    _min = min(x_min, y_min)
    _max = max(x_max, y_max)
    pad = (_max - _min) / 3
    ax.set_xlim(_min - pad, _max + pad)
    ax.set_ylim(_min - pad, _max + pad)
    ax.set_aspect('equal', 'box')

    ax.tick_params(direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.grid(True, linestyle='--', alpha=0.5)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(lc, cax=cax)
    cbar.set_label("D(t)")

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    plt.tight_layout(pad=0.2)
    plt.show()
