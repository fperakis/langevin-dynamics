import numpy as np


def run_langevin_simulation(
    T=1.0,
    tau=1.0,
    n_steps=100000,
    n_particles=100,
    dt=0.001,
    gamma_l=1.0,
    gamma_g=1e2,
    m=1.0,
    kB=1.0,
    seed=2,
    scheme="semi-implicit",
):
    """
    Two-state underdamped Langevin simulation with continuous-time (Poisson) switching
    between low- and high-friction states.

    Parameters
    ----------
    T : float
        Temperature.
    tau : float
        Mean switching time (⟨Δt_switch⟩).
    n_steps : int
        Number of integration steps.
    n_particles : int
        Number of independent trajectories.
    dt : float
        Time step (should satisfy dt << min(tau, m/gamma_g, m/gamma_l)).
    gamma_l, gamma_g : float
        Friction coefficients for liquid-like and glass-like states.
    m : float
        Particle mass.
    kB : float
        Boltzmann constant.
    seed : int
        Random seed.
    scheme : str
        Integration scheme: "semi-implicit" (default) or "explicit".

    Returns
    -------
    positions : ndarray, shape (n_particles, n_steps, 2)
    velocities : ndarray, shape (n_particles, n_steps, 2)
    gamma_record : ndarray, shape (n_particles, n_steps)
        Time series of friction coefficients for each particle.
    """
    np.random.seed(seed)

    # --- Initialize states and arrays ---
    gamma_states = np.random.rand(n_particles) < 0.5  # True=liquid, False=glass
    gamma_array = np.where(gamma_states, gamma_l, gamma_g)

    positions = np.zeros((n_particles, n_steps, 2))
    velocities = np.zeros((n_particles, n_steps, 2))
    gamma_record = np.zeros((n_particles, n_steps))
    gamma_record[:, 0] = gamma_array

    rand_force = np.random.normal(size=(n_steps, n_particles, 2))

    # --- Precompute Poisson switching times for each particle ---
    t_max = n_steps * dt
    n_events_est = int(3 * t_max / tau) + 10
    switch_times = np.cumsum(
        np.random.exponential(tau, size=(n_particles, n_events_est)), axis=1
    )
    next_switch_idx = np.zeros(n_particles, dtype=int)
    next_switch_time = switch_times[np.arange(n_particles), next_switch_idx]

    # --- Integration loop ---
    time = 0.0
    for step in range(1, n_steps):
        time += dt

        # Handle switching events (continuous time)
        to_flip = time >= next_switch_time
        if np.any(to_flip):
            gamma_states[to_flip] = ~gamma_states[to_flip]
            gamma_array[to_flip] = np.where(gamma_states[to_flip], gamma_l, gamma_g)
            next_switch_idx[to_flip] += 1
            # resample new event times if we run out
            exhausted = next_switch_idx >= n_events_est
            if np.any(exhausted):
                switch_times[exhausted] += np.cumsum(
                    np.random.exponential(tau, size=(np.sum(exhausted), n_events_est)),
                    axis=1,
                )
                next_switch_idx[exhausted] = 0
            next_switch_time = switch_times[
                np.arange(n_particles),
                np.clip(next_switch_idx, 0, n_events_est - 1),
            ]

        gamma_record[:, step] = gamma_array

        # --- Langevin velocity update ---
        R = rand_force[step]
        sigma_v = np.sqrt(2 * kB * T * gamma_array * dt) / m  # velocity noise amplitude

        if scheme.lower() in ["semi-implicit", "semiimplicit"]:
            # Stable velocity-form semi-implicit scheme
            damping = 1.0 / (1.0 + (gamma_array * dt) / m)
            velocities[:, step] = damping[:, None] * (
                velocities[:, step - 1] + sigma_v[:, None] * R
            )

        elif scheme.lower() == "explicit":
            # Euler–Maruyama integrator (simple, requires dt << m/gamma)
            velocities[:, step] = (
                velocities[:, step - 1]
                - (gamma_array[:, None] / m) * velocities[:, step - 1] * dt
                + sigma_v[:, None] * R
            )

        else:
            raise ValueError("Unknown scheme: choose 'semi-implicit' or 'explicit'.")

        # --- Position update ---
        positions[:, step] = positions[:, step - 1] + velocities[:, step] * dt

    return positions, velocities, gamma_record