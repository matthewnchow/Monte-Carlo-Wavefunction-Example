import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)   # Using the same seed to get consistent results. Comment out this line for random seed

# Problem 1a
# Model of decaying Rabi oscillations for a 2 lvl system. Working in the {|g> , |e>} basis


def run_1a():
    t_initial = 0                            # time variable, in units of 1/Gamma, where Gamma is the lifetime of |e>
    t_final = 4                              # Molmer's plots go out to 4*(1/Gamma)

    Omega = 6                                       # Rabi frequency
    t_steps = 1000                                  # Number of time steps per unravelling
    ts = np.linspace(t_initial, t_final, t_steps)   # Array of times
    psi_0 = np.array([1, 0])                        # Assume atom starts in the ground state
    psi = 1j*np.zeros((t_steps, 2))                 # Wavefunction at each time step
    psi[0] = psi_0
    sigma_x = np.array([[0, 1], [1, 0]])
    proj_e = np.array([[0, 0], [0, 1]])
    H_eff = -0.5j * proj_e - 1 * Omega*0.5*sigma_x            # Effective Hamiltonian


    def unravel_rabi(h_e=H_eff, steps=t_steps, c_0=psi_0):
        # Function for calculating a single unravelling
        dt = float(t_final - t_initial) / steps  # time step
        c = 1j * np.zeros((steps, 2))  # Variable for storing coefficients of wavefunction at each time step
        c[0] = c_0

        # Take a bunch of time steps in loop below to propagate wvfn
        for i in range(1, steps):
            prob_jump = np.abs(c[i-1, 1])**2 * dt               # Prob = (Gamma=1) * |ce|^2 * dt
            jump = np.random.rand() < prob_jump
            if jump:                             # Would have to figure out which kind of jump if there were multiple
                c[i] = np.array([1, 0])        # All population moves to ground state if there is a jump
            else:                                # Evolve according to H_effective and re-normalize
                c[i] = c[i-1] - 1j * dt * np.matmul(h_e, c[i - 1])
                c[i] = c[i]/(np.dot(c[i], np.conjugate(c[i])))
        return c

    psi = unravel_rabi()
    p_g = np.abs(psi[:, 0])**2
    p_e = np.abs(psi[:, 1])**2

    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.suptitle('Problem 1a: Decaying Rabi Oscillations')

    ax[0].plot(ts, p_e)
    ax[0].set_title("Singe Unravelling")
    ax[0].set_xlabel("Time (1/Gamma)")
    ax[0].set_ylabel("Prob to be in |e>")
    ax[0].set_ylim(0, 1)

    m_trials = 100                 # Molmer averages 100 wvfns
    p_es = np.zeros((m_trials, len(p_e)))
    p_es[0] = p_e

    for m in range(1, m_trials):
        psi = unravel_rabi()
        p_es[m] = np.abs(psi[:, 1])**2

    uncertainty = np.std(p_es, 0)/np.sqrt(m_trials)

    ax[1].errorbar(ts, np.mean(p_es, 0), uncertainty)
    ax[1].plot(ts, np.mean(p_es, 0), 'r')  # Center line plotted in red for clarity
    ax[1].set_title("Average of %i unravellings" % m_trials)
    ax[1].set_xlabel("Time (1/Gamma)")
    ax[1].set_ylim(0, 1)


# Problem 1b


def run_1b():
    t_initial = 0                            # time variable, in units of 1/Gamma, where Gamma is the lifetime of |e>
    t_final = 4                              # Molmer's plots go out to 4*(1/Gamma)

    Omega = 6                                       # Rabi frequency
    t_steps = 1000                                  # Number of time steps per unravelling
    ts = np.linspace(t_initial, t_final, t_steps)   # Array of times
    psi_0 = np.array([1, 0])                        # Assume atom starts in the ground state
    psi = 1j*np.zeros((t_steps, 2))                 # Wavefunction at each time step
    psi[0] = psi_0
    sigma_x = np.array([[0, 1], [1, 0]])
    proj_e = np.array([[0, 0], [0, 1]])
    H_eff = -0.5j * proj_e - 1 * Omega*0.5*sigma_x            # Effective Hamiltonian


    def unravel_rabi_wait(h_e=H_eff, steps=t_steps, c_0=psi_0):
        # Function for calculating a single unravelling
        dt = float(t_final - t_initial) / steps  # time step
        c = 1j * np.zeros((steps, 2))  # Variable for storing coefficients of wavefunction at each time step
        c[0] = c_0
        eta = np.random.rand()
        c_t = c_0*(-1j)**2

        # Take a bunch of time steps in loop below to propagate wvfn
        for i in range(1, steps):
            # calculate the norm of the wvfn and compare to chosen random number
            norm2 = np.abs(np.dot(c_t, np.conjugate(c_t)))
            jump = (norm2 <= eta)
            if jump:                             # Would have to figure out which kind of jump if there were multiple
                c[i] = np.array([1, 0])          # All population moves to ground state if there is a jump

                eta = np.random.rand()           # Also pick a new random number
                c_t = c[i]                       # And reset numerical integration state
            else:                                # Evolve according to H_effective and re-normalize
                c[i] = c[i-1] - 1j * dt * np.matmul(h_e, c[i - 1])
                c[i] = c[i]/(np.dot(c[i], np.conjugate(c[i])))
                c_t -= 1j * dt * np.matmul(h_e, c_t)  # Performing numerical integration of differential propagator
        return c

    psi = unravel_rabi_wait()
    p_g = np.abs(psi[:, 0])**2
    p_e = np.abs(psi[:, 1])**2

    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.suptitle('Problem 1b: Decaying Rabi Oscillations, Calculated with wait time distribution')

    ax[0].plot(ts, p_e)
    ax[0].set_title("Singe Unravelling")
    ax[0].set_xlabel("Time (1/Gamma)")
    ax[0].set_ylabel("Prob to be in |e>")
    ax[0].set_ylim(0, 1)

    m_trials = 100                 # Molmer averages 100 wvfns
    p_es = np.zeros((m_trials, len(p_e)))
    p_es[0] = p_e

    for m in range(1, m_trials):
        psi = unravel_rabi_wait()
        p_es[m] = np.abs(psi[:, 1])**2

    uncertainty = np.std(p_es, 0)/np.sqrt(m_trials)

    ax[1].errorbar(ts, np.mean(p_es, 0), uncertainty)
    ax[1].plot(ts, np.mean(p_es, 0), 'r')  # Center line plotted in red for clarity
    ax[1].set_title("Average of %i unravellings" % m_trials)
    ax[1].set_xlabel("Time (1/Gamma)")
    ax[1].set_ylim(0, 1)


run_1a()
run_1b()
plt.show()
