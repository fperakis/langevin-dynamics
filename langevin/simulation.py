import numpy as np

def run_langevin_simulation(
    T=1.0,
    tau=1.0,
    n_steps=100000,
    n_particles=100,
    dt=0.01,
    gamma_l=1.0,
    gamma_g=1e2,
    m=1.0,
    kB=1.0,
    seed=2,
    scheme="semi-implicit"
):
    """
    Langevin simulation with symmetric two-state switching (fast <-> slow),
    using explicit or semi-implicit velocity update and random initial states.

    Parameters:
        T         : Temperature
        tau       : Mean switching time
        n_steps   : Number of time steps
        n_particles : Number of particles
        dt        : Time step size
        gamma_l   : Friction in 'fast' (liquid-like) state
        gamma_g   : Friction in 'slow' (glassy-like) state
        m         : Particle mass
        kB        : Boltzmann constant
        seed      : Random seed
        scheme    : "semi-implicit" (default) or "explicit" integration scheme

    Returns:
        positions     : (n_particles, n_steps, 2)
        velocities    : (n_particles, n_steps, 2)
        gamma_record  : (n_particles, n_steps)
    """
    np.random.seed(seed)
    switch_prob = dt / tau

    # Random initial state: True = fast (liquid), False = slow (glass)
    gamma_states = np.random.rand(n_particles) < 0.5
    gamma_array = np.where(gamma_states, gamma_l, gamma_g)

    positions = np.zeros((n_particles, n_steps, 2))
    velocities = np.zeros((n_particles, n_steps, 2))
    gamma_record = np.zeros((n_particles, n_steps))
    gamma_record[:, 0] = gamma_array

    rand_switch = np.random.rand(n_steps, n_particles)
    rand_force = np.random.normal(size=(n_steps, n_particles, 2))

    for step in range(1, n_steps):
        # Symmetric switching
        flip = rand_switch[step] < switch_prob
        gamma_states[flip] = ~gamma_states[flip]
        gamma_array[:] = np.where(gamma_states, gamma_l, gamma_g)
        gamma_record[:, step] = gamma_array

        # Langevin velocity update
        sigma = np.sqrt(2 * kB * T * gamma_array / dt) / m
        noise = sigma[:, np.newaxis] * rand_force[step]

        if scheme == "semi-implicit":
            damping = 1.0 / (1.0 + dt * gamma_array / m)
            velocities[:, step] = (velocities[:, step - 1] + dt * noise / m) * damping[:, np.newaxis]
        elif scheme == "explicit":
            accel = (-gamma_array[:, np.newaxis] * velocities[:, step - 1] + noise) / m
            velocities[:, step] = velocities[:, step - 1] + accel * dt
        else:
            raise ValueError("Unknown scheme. Choose 'semi-implicit' or 'explicit'.")

        positions[:, step] = positions[:, step - 1] + velocities[:, step] * dt

    return positions, velocities, gamma_record
