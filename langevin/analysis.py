import numpy as np
from scipy.optimize import curve_fit

def compute_msd(positions, dt=1e-2, n_lags=100, min_lag=1):
    """
    Computes ensemble-averaged MSD using log-spaced lags.

    Parameters:
        positions : ndarray (n_particles, n_steps, 2)
        dt        : float, timestep
        n_lags    : number of lag points
        min_lag   : first lag point

    Returns:
        times : array of lag times
        msd   : array of MSD values
    """
    x,y = positions[:,:,0],positions[:,:,1]
    n_particles, n_steps = x.shape
    max_lag = n_steps // 2
    log_lags = np.unique(np.logspace(np.log10(min_lag), np.log10(max_lag), n_lags).astype(int))

    msd = np.zeros(len(log_lags))
    
    for i, lag in enumerate(log_lags):
        disp_sq = (x[:, lag:] - x[:, :-lag])**2 + (y[:, lag:] - y[:, :-lag])**2
        msd[i] = np.mean(disp_sq)

    times = log_lags*dt  
    return times, msd


def estimate_diffusion_from_msd(time_lags, msd, fit_start_ratio=0.5, fit_end_ratio=0.7):
    """
    Estimate the effective diffusion coefficient from the MSD curve by linear fitting.

    Parameters:
        time_lags        : ndarray
            Array of time lags (e.g. from log spacing).
        msd              : ndarray
            Mean squared displacement values corresponding to time_lags.
        fit_start_ratio  : float
            Fraction of the data range to start the linear fit (default 0.5).
        fit_end_ratio    : float
            Fraction of the data range to end the linear fit (default 0.7).

    Returns:
        D_eff  : float
            Estimated diffusion coefficient (in 2D: MSD = 4Dt).
        popt   : tuple
            Optimal fit parameters.
        pcov   : ndarray
            Covariance of the fit parameters.
    """
    def diffusion_linear_fit(t, D):
        return 4 * D * t  # For 2D diffusion

    start_idx = int(len(time_lags) * fit_start_ratio)
    end_idx = int(len(time_lags) * fit_end_ratio)
    popt, pcov = curve_fit(diffusion_linear_fit, time_lags[start_idx:end_idx], msd[start_idx:end_idx])
    D_eff = popt[0]
    return D_eff, popt, pcov


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

def compute_ngp(positions, dt=0.01, n_lags=100):
    """
    Computes the Non-Gaussian parameter α₂(t) using ensemble + time average.

    Returns:
        times  : ndarray of times
        alpha2 : ndarray of α₂(t)
    """
    n_particles, n_steps, dim = positions.shape
    max_lag = n_steps // 2
    lag_steps = np.unique(np.logspace(0, np.log10(max_lag), n_lags).astype(int))

    alpha2 = np.zeros(len(lag_steps))
    for i, lag in enumerate(lag_steps):
        disp = positions[:, lag:, :] - positions[:, :-lag, :]
        dr2 = np.sum(disp**2, axis=-1)
        msd = np.mean(dr2)
        fourth_moment = np.mean(dr2**2)
        alpha2[i] = fourth_moment / (2 * msd**2) - 1 if msd > 0 else 0.0

    times = lag_steps * dt
    return times, alpha2


def compute_isf(
    positions, q=1.0, dt=0.01, stride=1, n_lags=100, lag_min=1
):
    """
    Computes the self-intermediate scattering function with log-spaced time lags.

    Parameters:
        positions : array of shape (n_particles, n_steps, 2)
            Particle trajectories.
        q         : float
            Wavevector magnitude.
        dt        : float
            Time step.
        stride    : int
            Temporal downsampling for fast computation.
        n_lags    : int
            Number of log-spaced lags to compute.
        lag_min   : int
            Minimum lag (in steps).

    Returns:
        times     : array of time lags (in physical time units)
        Fs_qt     : ISF evaluated at those time lags
    """
    pos = positions[:, ::stride]  # downsample in time
    n_particles, n_steps, _ = pos.shape

    # Generate log-spaced lag steps (unique and sorted)
    lag_steps = np.unique(np.logspace(
        np.log10(lag_min), np.log10(n_steps // 2), n_lags
    ).astype(int))

    Fs_qt = np.zeros_like(lag_steps, dtype=float)

    for i, lag in enumerate(lag_steps):
        if lag >= n_steps:
            break
        dr = pos[:, lag:, :] - pos[:, :-lag, :]
        dq = np.sum(dr, axis=2) * q  # projection along q-direction (approx)
        Fs_qt[i] = np.mean(np.cos(dq))

    time_lags = lag_steps * dt * stride
    return time_lags, Fs_qt


def compute_overlap_function(positions, a=0.3, dt=0.01, min_lag=1, max_lag=None, stride=1, num_lags=100):
    """
    Fast version of the self-overlap function Q(t) with vectorized implementation.
    """
    n_particles, n_steps, _ = positions.shape
    if max_lag is None:
        max_lag = n_steps // 2

    lag_steps = np.unique(np.logspace(np.log10(min_lag), np.log10(max_lag), num=num_lags, dtype=int))
    times = lag_steps * dt
    Q_t = np.zeros(len(lag_steps))

    for i, lag in enumerate(lag_steps):
        t0_max = n_steps - lag
        if t0_max <= 0:
            continue
        # displacement: shape (n_particles, t0_max)
        dr = positions[:, lag:] - positions[:, :-lag]  # shape: (n_particles, t0_max, 2)
        dr2 = np.sum(dr**2, axis=-1)
        overlap = dr2 < a**2
        Q_t[i] = np.mean(overlap)

    return times, Q_t
    

def compute_chi4_overlap(
    positions,
    a=0.3,
    dt=0.01,
    n_lags=100,
    min_lag=1
):
    """
    Computes the dynamical susceptibility χ₄(t) using log-spaced time lags.

    Parameters:
        positions : ndarray (n_particles, n_steps, dim)
            Particle trajectories.
        a         : float
            Overlap threshold distance.
        dt        : float
            Time step size.
        n_lags    : int
            Number of log-spaced lag steps.
        min_lag   : int
            Minimum lag (in steps).

    Returns:
        time_lags : ndarray (n_lags,)
            Log-spaced lag times in physical units.
        chi4      : ndarray (n_lags,)
            Dynamical susceptibility χ₄(t).
    """
    n_particles, n_steps, _ = positions.shape
    max_lag = n_steps // 2
    lag_steps = np.unique(np.logspace(
        np.log10(min_lag), np.log10(max_lag), n_lags).astype(int))

    chi4 = np.zeros(len(lag_steps))

    for i, lag in enumerate(lag_steps):
        displacements = positions[:, lag:] - positions[:, :-lag]
        dr2 = np.sum(displacements**2, axis=-1)  # shape: (n_particles, n_steps - lag)
        overlap = (dr2 < a**2).astype(float)     # binary overlap matrix

        # Average over time origins for each particle
        q_i_t = np.mean(overlap, axis=1)         # shape: (n_particles,)

        # χ₄(t): variance of overlap over particles
        chi4[i] = n_particles * np.var(q_i_t)

    time_lags = lag_steps * dt
    return time_lags, chi4

    def vacf_from_velocities(vel, dt, max_lag=None):
    """
    Compute velocity autocorrelation function <v_x(t0+τ) v_x(t0)> averaged
    over particles and time origins.
    """
    vx = vel[:, :, 0]  # x-component
    n_particles, n_steps = vx.shape
    if max_lag is None:
        max_lag = n_steps // 2
    vx = vx - np.mean(vx, axis=1, keepdims=True)  # subtract mean per trajectory

    Cvv = np.zeros(max_lag)
    for lag in range(max_lag):
        # average over particles and time origins
        prod = vx[:, :-lag or None] * vx[:, lag:]
        Cvv[lag] = np.mean(prod)
    t_lags = np.arange(max_lag) * dt
    return t_lags, Cvv

def mobility_from_vacf(Cvv, t_lags, T, kB=1.0, t_cut=None):
    """
    Compute mobility via Green–Kubo integral.
    """
    if t_cut is None:
        t_cut = 0.1 * t_lags[-1]
    mask = t_lags <= t_cut
    mu_gk = (1.0 / (kB * T)) * np.trapezoid(Cvv[mask], t_lags[mask])
    return mu_gk

#import numpy as np

def vacf_from_velocities_fft(vel, dt, max_lag=None, normalize=True):
    """
    Fast FFT-based velocity autocorrelation function.
    Computes Cvv(tau) = <v_x(t0+tau) v_x(t0)> averaged over particles.

    Parameters
    ----------
    vel : ndarray (n_particles, n_steps, 2)
        Velocity trajectories.
    dt : float
        Timestep.
    max_lag : int or None
        Maximum lag to return (default: half the trajectory).
    normalize : bool
        If True, divide by number of contributing points.

    Returns
    -------
    t_lags : ndarray
        Array of lag times.
    Cvv : ndarray
        Velocity autocorrelation function averaged over particles.
    """
    vx = vel[:, :, 0]               # (n_particles, n_steps)
    n_particles, n_steps = vx.shape
    if max_lag is None:
        max_lag = n_steps // 2

    # subtract mean per trajectory
    vx = vx - np.mean(vx, axis=1, keepdims=True)

    # --- FFT-based autocorrelation for each particle ---
    n_fft = 2 * n_steps  # zero-padding to avoid circular convolution
    Cvv_all = np.zeros((n_particles, n_steps))

    for i in range(n_particles):
        f = np.fft.fft(vx[i], n=n_fft)
        acf = np.fft.ifft(f * np.conjugate(f)).real
        Cvv_all[i] = acf[:n_steps]  # keep only positive lags

    # average over particles
    Cvv = np.mean(Cvv_all, axis=0)

    if normalize:
        # divide by number of pairs contributing to each lag
        norm = np.arange(n_steps, 0, -1)
        Cvv /= norm

    t_lags = np.arange(n_steps) * dt
    return t_lags[:max_lag], Cvv[:max_lag]
