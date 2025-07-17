import numpy as np
from scipy.optimize import least_squares

def stretched_exponential(t, tau, beta, A=1.0):
    """
    Stretched exponential model: A * exp(-(t / tau)^beta)
    """
    return A * np.exp(-(t / tau) ** beta)


def double_exponential(t, tau1, tau2, A=0.5):
    """
    Two-step exponential model: (1-A) * exp(-t / tau1) + A * exp(-t / tau2)
    """
    return (1 - A) * np.exp(-t / tau1) + A * np.exp(-t / tau2)


def two_step_stretched(t, tau1, tau2, beta, A=0.5):
    """
    Two-step model with stretched second decay:
    (1-A) * exp(-t / tau1) + A * exp(-(t / tau2)^beta)
    """
    return (1 - A) * np.exp(-(t / tau1)**2) + A * np.exp(-(t / tau2) ** beta)

def VFT_gamma(T, T_0=0.1, A=1, B = 0.1):
    """
    VFT-like divergence of friction coefficient:
        gamma_s(T) = exp[ (delta * T_0 / (T - T_0))^exponent ]
    Diverges as T → T_0 from above.
    """
    return A*np.exp((B / (T - T_0)))

def log_weighted_residuals(params, model_func, t, y, weights=None):
    """
    Compute residuals in log space (optional weights).
    """
    y_model = model_func(t, *params)
    residuals = np.log(np.maximum(y_model, 1e-12)) - np.log(np.maximum(y, 1e-12))
    if weights is not None:
        residuals *= weights
    return residuals


def fit_model(t, y, model_func, p0, bounds=None, log_weights=True):
    """
    Fit data using least squares in log space.

    Parameters:
        t          : array
            Time points
        y          : array
            Function values to fit
        model_func : callable
            Model function to fit to (e.g. stretched_exponential)
        p0         : list
            Initial guess for parameters
        bounds     : tuple
            (lower_bounds, upper_bounds)
        log_weights : bool
            Whether to use logarithmic weighting (default True)

    Returns:
        fit_params : optimized parameters
        y_fit      : model evaluated with fitted parameters

    Example of usage: 
        from fitting import fit_model, stretched_exponential
        fit_params, fit_curve = fit_model(t, F_qt, model_func=stretched_exponential, p0=[1.0, 0.5])
        
    """
    if log_weights:
        weights = np.logspace(0, 1, len(t))
        weights /= weights.max()
    else:
        weights = None

    res = least_squares(
        log_weighted_residuals,
        x0=p0,
        args=(model_func, t, y, weights),
        bounds=bounds if bounds is not None else (-np.inf, np.inf),
    )

    return res.x, model_func(t, *res.x)
