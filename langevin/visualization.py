import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_trajectory_colored_by_diffusion(
    pos,
    D_local,
    mid_points,
    vmin=None,
    vmax=None,
    cmap='viridis',
    savefig=False):
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
    if savefig==True:
        plt.savefig('trajectory.pdf')
    plt.show()
