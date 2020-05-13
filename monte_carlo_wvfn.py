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
    h_eff = -0.5j * proj_e - 1 * Omega*0.5*sigma_x            # Effective Hamiltonian

    def unravel_rabi(h_e=h_eff, steps=t_steps, c_0=psi_0):
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
    ax[0].set_title("Single Trajectory")
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
    ax[1].set_title("Average of %i Trajectories" % m_trials)
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
    h_eff = -0.5j * proj_e - 1 * Omega*0.5*sigma_x            # Effective Hamiltonian

    def unravel_rabi_wait(h_e=h_eff, steps=t_steps, c_0=psi_0):
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
    ax[0].set_title("Single Trajectory")
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
    ax[1].set_title("Average of %i Trajectories" % m_trials)
    ax[1].set_xlabel("Time (1/Gamma)")
    ax[1].set_ylim(0, 1)


def run_1c():
    # trap_w = 0.1 / 5  # ~100 kHz trap frequency shown, divided by 5MHz (approx linewidth of Cs)
    t_initial = 0  # time variable, in units of 1/Gamma, where Gamma is the lifetime of |e>
    t_final = 100  # Go out to steady state times (1/Gamma)

    Omega = 6  # Rabi frequency
    t_steps = 2500  # Number of time steps per unravelling
    ts = np.linspace(t_initial, t_final, t_steps)  # Array of times
    psi_0 = np.array([1, 0])  # Assume atom starts in the ground state
    psi = 1j * np.zeros((t_steps, 2))  # Wavefunction at each time step
    psi[0] = psi_0
    sigma_x = np.array([[0, 1], [1, 0]])
    proj_e = np.array([[0, 0], [0, 1]])
    h_eff = -0.5j * proj_e - 1 * Omega * 0.5 * sigma_x  # Effective Hamiltonian, need to add in the trap part and detuning

    def unravel_rabi_wait(h_e=h_eff, steps=t_steps, c_0=psi_0):
        # Function for calculating a single unravelling
        dt = float(t_final - t_initial) / steps  # time step
        c = 1j * np.zeros((steps, 2))  # Variable for storing coefficients of wavefunction at each time step
        c[0] = c_0
        eta = np.random.rand()
        c_t = c_0 * (-1j) ** 2
        photons = 0

        # Take a bunch of time steps in loop below to propagate wvfn
        for i in range(1, steps):
            # calculate the norm of the wvfn and compare to chosen random number
            norm2 = np.abs(np.dot(c_t, np.conjugate(c_t)))
            jump = (norm2 <= eta)
            if jump:  # Would have to figure out which kind of jump if there were multiple
                c[i] = np.array([1, 0])  # All population moves to ground state if there is a jump
                eta = np.random.rand()  # Also pick a new random number
                c_t = c[i]  # And reset numerical integration state
                photons += 1
            else:  # Evolve according to H_effective and re-normalize
                c[i] = c[i - 1] - 1j * dt * np.matmul(h_e, c[i - 1])
                c[i] = c[i] / (np.dot(c[i], np.conjugate(c[i])))
                c_t -= 1j * dt * np.matmul(h_e, c_t)  # Performing numerical integration of differential propagator
        return c, photons


    psi = unravel_rabi_wait()
    p_g = np.abs(psi[:, 0]) ** 2
    p_e = np.abs(psi[:, 1]) ** 2

    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.suptitle('Problem 1b: Decaying Rabi Oscillations, Calculated with wait time distribution')

    ax[0].plot(ts, p_e)
    ax[0].set_title("Single Trajectory")
    ax[0].set_xlabel("Time (1/Gamma)")
    ax[0].set_ylabel("Prob to be in |e>")
    ax[0].set_ylim(0, 1)

    m_trials = 100  # Molmer averages 100 wvfns
    p_es = np.zeros((m_trials, len(p_e)))
    p_es[0] = p_e

    for m in range(1, m_trials):
        psi = unravel_rabi_wait()
        p_es[m] = np.abs(psi[:, 1]) ** 2

    uncertainty = np.std(p_es, 0) / np.sqrt(m_trials)

    ax[1].errorbar(ts, np.mean(p_es, 0), uncertainty)
    ax[1].plot(ts, np.mean(p_es, 0), 'r')  # Center line plotted in red for clarity
    ax[1].set_title("Average of %i Trajectories" % m_trials)
    ax[1].set_xlabel("Time (1/Gamma)")
    ax[1].set_ylim(0, 1)


# Problem 2
# Model of CPT for atom driven on |g, J=1> to |e, J=1 excited>.
# Basis states: {|g, m=-1> , |g, m=0>, |g, m=1>, |e, m=-1>, |e, m=0>, |e, m=1>}
# First, work in the z basis, where dark state = 1/sqrt(2) * (|g, mz=-1> + |g, mz=+1>)
# Then, work in the y basis, where dark state = |g, my=0>


def run_2():
    t_initial = 0                             # time variable, in units of 1/Gamma, where Gamma is the lifetime of |e>
    t_final = 10                              # Molmer's plots go out to 10*(1/Gamma)

    Omega = 3                                       # Rabi frequency
    t_steps = 1000                                  # Number of time steps per unravelling
    ts = np.linspace(t_initial, t_final, t_steps)   # Array of times

    # First work in the Z basis
    psi_0 = np.array([1, 0, 0, 0, 0, 0])                     # Assume atom starts in the mz = -1 state
    psi_dark = np.array([1, 0, 1, 0, 0, 0])/np.sqrt(2)       # dark state in the Z basis
    psi = 1j*np.zeros((t_steps, len(psi_0)))                 # Wavefunction at each time step
    psi[0] = psi_0

    d_y = 1j*np.zeros((len(psi_0), len(psi_0)))
    d_y[0, 4] = d_y[1, 5] = 1j                               # Note, this is really Dy dagger in Z basis
    d_y[1, 3] = d_y[2, 4] = -1j
    d_y *= 0.5
    # print(d_y)
    proj_e = 1j*np.zeros((len(psi_0), len(psi_0)))
    proj_e[3, 3] = proj_e[4, 4] = proj_e[5, 5] = 1
    # print(proj_e)
    h_eff = -0.5j * proj_e - 1 * Omega*0.5*(d_y + np.transpose(np.conjugate(d_y)))          # Effective Hamiltonian
    # print(h_eff - np.transpose(np.conjugate(h_eff)))

    # Lindblad jump operators:
    sig_plus = 1j * np.zeros((len(psi_0), len(psi_0)))
    sig_plus[0, 4] = 1

    sig_min = 1j * np.zeros((len(psi_0), len(psi_0)))
    sig_min[2, 4] = 1

    l_z = [sig_plus, sig_min]

    def unravel_y(h=h_eff, steps=t_steps, c_0=psi_0, ls=l_z):
        # Function for calculating a single unravelling
        dt = float(t_final - t_initial) / steps  # time step
        c = 1j * np.zeros((steps, len(psi_0)))  # Variable for storing coefficients of wavefunction at each time step
        c[0] = c_0
        h_jump = h - np.transpose(np.conjugate(h))

        # Take a bunch of time steps in loop below to propagate wvfn
        for i in range(1, steps):
            dp = dt * 1j*np.dot(np.conjugate(c[i-1]), np.matmul(h_jump, c[i-1]))  # Jump prob
            epsilon = np.random.rand()
            jump = epsilon < dp
            if jump:
                cumulative_dpm = 0
                # Calculate each jump probability, if epsilon less than the sum of the dpm up to this point, select jump
                for L in ls:
                    c[i] = np.matmul(L, c[i-1])  # Temporary, just used to calc dp
                    dpm = dt*np.abs(np.dot(c[i], np.conjugate(c[i])))
                    cumulative_dpm += dpm
                    if epsilon < cumulative_dpm:
                        c[i] /= np.sqrt(dpm/dt)
                        break
            # c[i] = np.array([1, 0, 0, 0, 0, 0])          # All population moves to ground state if there is a jump
            else:                                # Evolve according to H_effective and re-normalize
                c[i] = c[i-1] - 1j * dt * np.matmul(h, c[i - 1])
                c[i] = c[i]/(np.dot(c[i], np.conjugate(c[i])))
        return c

    psi = unravel_y()
    p_d = np.abs(np.matmul(psi, psi_dark))**2  # probability of being in the dark state
    # print(np.abs(np.matmul(psi[1:5, :], psi_dark))**2)

    fig, ax = plt.subplots(nrows=2, ncols=2)
    # fig.suptitle('Problem 2: Simulation of Evolution to Dark State')

    ax[0, 0].plot(ts, p_d)
    ax[0, 0].set_title("Single Trajectory, Z basis")
    ax[0, 0].set_ylabel("Prob to be in |dark>")
    ax[0, 0].set_ylim(0, 1)

    m_trials = 100                 # Molmer averages 100 wvfns
    p_ds = np.zeros((m_trials, len(p_d)))
    p_ds[0] = p_d

    for m in range(1, m_trials):
        psi = unravel_y()
        p_ds[m] = np.abs(np.matmul(psi, psi_dark)) ** 2

    uncertainty = np.std(p_ds, 0)/np.sqrt(m_trials)

    ax[1, 0].errorbar(ts, np.mean(p_ds, 0), uncertainty)
    ax[1, 0].plot(ts, np.mean(p_ds, 0), 'r')  # Center line plotted in red for clarity
    ax[1, 0].set_title("Average of %i Trajectories" % m_trials)
    ax[1, 0].set_xlabel("Time (1/Gamma)")
    ax[1, 0].set_ylabel("Prob to be in |dark>")
    ax[1, 0].set_ylim(-0.05, 1)

    # Now repeat the problem in the y basis
    psi_0y = np.array([0.5, 1j/np.sqrt(2), 0.5, 0, 0, 0])   # Assume atom starts in the mz = -1 state
    psi_darky = np.array([0, 1, 0, 0, 0, 0])                 # dark state in the Z basis
    psi = 1j*np.zeros((t_steps, len(psi_0)))                 # Wavefunction at each time step
    psi[0] = psi_0

    d_yy = 1j*np.zeros((len(psi_0y), len(psi_0y)))
    d_yy[0, 3] = -1                                             # Note, this is really Dy dagger in Z basis
    d_yy[2, 5] = 1
    d_yy *= 0.5
    # print(d_yy)

    l_pi = 1j*np.zeros((len(psi_0y), len(psi_0y)))          # Lindblad jump operator for (both) pi transitions
    l_pi[0, 3] = l_pi[2, 5] = 1                                 # Puts excited +/-1 into +/-1 ground
    l_sig = 1j*np.zeros((len(psi_0y), len(psi_0y)))         # Lindblad jump operator for (both) sigma transitions
    l_sig[1, 3] = l_sig[1, 5] = 1                                # Puts excited +/-1 into 0 ground
    l_y = [l_pi, l_sig]
    print(l_y)

    h_effy = -0.5j * proj_e - 1 * Omega*0.5*(d_yy + np.transpose(np.conjugate(d_yy)))          # Effective Hamiltonian
    print(h_effy)
    # print(h_effy - np.transpose(np.conjugate(h_effy)))

    psi = unravel_y(h=h_effy, c_0=psi_0y, ls=l_y)
    p_d = np.abs(np.matmul(psi, psi_darky))**2  # probability of being in the dark state
    # print(np.abs(np.matmul(psi[1:5, :], psi_dark))**2)

    ax[0, 1].plot(ts, p_d)
    ax[0, 1].set_title("Single Trajectory, Y Basis")
    ax[0, 1].set_ylim(-0.05, 1)

    m_trials = 100                 # Molmer averages 100 wvfns
    p_ds = np.zeros((m_trials, len(p_d)))
    p_ds[0] = p_d

    for m in range(1, m_trials):
        psi = unravel_y()
        p_ds[m] = np.abs(np.matmul(psi, psi_darky)) ** 2

    uncertainty = np.std(p_ds, 0)/np.sqrt(m_trials)

    ax[1, 1].errorbar(ts, np.mean(p_ds, 0), uncertainty)
    ax[1, 1].plot(ts, np.mean(p_ds, 0), 'r')  # Center line plotted in red for clarity
    ax[1, 1].set_title("Average of %i Trajectories" % m_trials)
    ax[1, 1].set_xlabel("Time (1/Gamma)")
    ax[1, 1].set_ylim(-0.05, 1)



# run_1a()
# run_1b()
run_2()

plt.show()

