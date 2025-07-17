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
    n_particles, n_steps = x.shape
    max_lag = n_steps // 2
    log_lags = np.unique(np.logspace(np.log10(min_lag), np.log10(max_lag), n_lags).astype(int))

    msd = np.zeros(len(log_lags))
    x,y = positions[:,:,0],positions[:,:,1]
    
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


def compute_isf(positions, q=1.0, dt=0.01, n_lags=100, stride=1):
    """
    Computes direction-averaged self-intermediate scattering function (ISF).

    Returns:
        times : lag times
        isf   : ISF as a function of time
    """
    n_particles, n_steps, dim = positions.shape
    q_vectors = [np.array([q, 0.0]), np.array([0.0, q])]
    max_lag = n_steps // 2
    lag_steps = np.unique(np.logspace(0, np.log10(max_lag), n_lags).astype(int))

    isf_q = np.zeros(len(lag_steps))

    for q_vec in q_vectors:
        iq = np.zeros(len(lag_steps))
        for i, lag in enumerate(lag_steps):
            dot_prod = []
            for start in range(0, n_steps - lag, stride):
                disp = positions[:, start + lag, :] - positions[:, start, :]
                phase = np.dot(disp, q_vec)
                dot_prod.append(np.mean(np.cos(phase)))
            iq[i] = np.mean(dot_prod)
        isf_q += iq / len(q_vectors)

    times = lag_steps * dt * stride
    return times, isf_q


def compute_overlap_function(positions, a=0.3, dt=0.01, n_lags=100):
    """
    Computes overlap function Q(t).

    Returns:
        times : lag times
        Q     : overlap function Q(t)
    """
    n_particles, n_steps, _ = positions.shape
    max_lag = n_steps // 2
    lag_steps = np.unique(np.logspace(0, np.log10(max_lag), n_lags).astype(int))

    Q = np.zeros(len(lag_steps))
    for i, lag in enumerate(lag_steps):
        disp = positions[:, lag:, :] - positions[:, :-lag, :]
        dr2 = np.sum(disp**2, axis=-1)
        Q[i] = np.mean(dr2 < a**2)

    times = lag_steps * dt
    return times, Q


def compute_chi4_overlap(positions, a=0.3, dt=0.01, n_lags=100):
    """
    Computes four-point dynamical susceptibility χ₄(t) based on the overlap function.

    Parameters:
        positions : ndarray (n_particles, n_steps, dim)
        a         : threshold distance for overlap (float)
        dt        : time step
        n_lags    : number of log-spaced lag points

    Returns:
        times : ndarray of lag times
        chi4  : dynamic susceptibility χ₄(t)
    """
    n_particles, n_steps, dim = positions.shape
    max_lag = n_steps // 2
    lag_steps = np.unique(np.logspace(0, np.log10(max_lag), n_lags).astype(int))

    chi4 = np.zeros(len(lag_steps))
    for i, lag in enumerate(lag_steps):
        disp = positions[:, lag:, :] - positions[:, :-lag, :]
        dr2 = np.sum(disp**2, axis=-1)
        q_i = (dr2 < a**2).astype(float)  # shape: (n_particles, n_steps - lag)
        Q_t = np.mean(q_i, axis=0)        # average per frame
        chi4[i] = n_particles * np.var(Q_t)

    times = lag_steps * dt
    return times, chi4
