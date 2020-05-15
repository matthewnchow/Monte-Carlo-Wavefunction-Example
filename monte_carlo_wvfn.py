import numpy as np
import matplotlib.pyplot as plt
import scipy

np.random.seed(1)   # Use the same seed to get consistent results. Comment out this line for random seed

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
    sigma_x = np.array([[0, 1], [1, 0]])            # = Sigma+ + sigma-
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
    p_g = np.abs(psi[:, 0])**2      # Population in the ground state
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
    sigma_x = np.array([[0, 1], [1, 0]])
    proj_e = np.array([[0, 0], [0, 1]])
    h_eff = -0.5j * proj_e - 1 * Omega*0.5*sigma_x            # Effective Hamiltonian


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
        return c


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


def run_1c():
    t_initial = 0  # time variable, in units of 1/Gamma, where Gamma is the lifetime of |e>
    t_final = 25   # Go out to steady state times (1/Gamma) for the first interval

    Omega = 6  # Rabi frequency
    t_steps = 2500  # Number of time steps per unravelling per interval
    psi_0 = np.array([1, 0])  # Assume atom starts in the ground state

    sigma_x = np.array([[0, 1], [1, 0]])
    proj_e = np.array([[0, 0], [0, 1]])
    h_eff = -0.5j * proj_e - 1 * Omega * 0.5 * sigma_x  # Effective Hamiltonian


    def unravel_rabi(h_e=h_eff, c_0=psi_0, t_i=t_initial, t_f=t_final, steps=t_steps):
        # Function for calculating a single unravelling
        dt = float(t_f - t_i) / steps  # time step
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

    n_taus = 100
    tau_max = 0.5  # More than double the Rabi time.
    taus = np.linspace(0, tau_max, n_taus)
    Cs = np.zeros((n_taus, 1))
    m_trials = 50

    A = np.array([[0, 0], [1, 0]])
    B = np.array([[0, 1], [0, 0]])   # Note Adagger = B
    for i in range(n_taus):
        a0_bar = 0
        a1_bar = 0
        a2_bar = 0
        a3_bar = 0
        for m in range(m_trials):
            psi = unravel_rabi()
            psi_t = psi[len(psi)-1]
            chi0_0 = psi_t + np.matmul(B, psi_t)
            mu_0 = np.sqrt(np.abs(np.dot(chi0_0, np.conjugate(chi0_0))))
            chi0_0 = chi0_0/np.sqrt(mu_0)
            chi0_1 = psi_t - np.matmul(B, psi_t)
            mu_1 = np.sqrt(np.abs(np.dot(chi0_1, np.conjugate(chi0_1))))
            chi0_1 = chi0_1 / np.sqrt(mu_1)
            chi0_2 = psi_t + 1j*np.matmul(B, psi_t)
            mu_2 = np.sqrt(np.abs(np.dot(chi0_2, np.conjugate(chi0_2))))
            chi0_2 = chi0_2 / np.sqrt(mu_2)
            chi0_3 = psi_t - 1j*np.matmul(B, psi_t)
            mu_3 = np.sqrt(np.abs(np.dot(chi0_3, np.conjugate(chi0_3))))
            chi0_3 = chi0_3 / np.sqrt(mu_3)
            chi0 = unravel_rabi(c_0=chi0_0, t_f=taus[i], steps=int((taus[i]+1)*50))
            chi1 = unravel_rabi(c_0=chi0_1, t_f=taus[i], steps=int((taus[i]+1)*50))
            chi2 = unravel_rabi(c_0=chi0_2, t_f=taus[i], steps=int((taus[i]+1)*50))
            chi3 = unravel_rabi(c_0=chi0_3, t_f=taus[i], steps=int((taus[i]+1)*50))
            a0 = np.matmul(chi0, B)  # c+/- and cprime +/- in Molmer
            a0_bar += np.sum(chi0 * a0)/(len(a0)*m_trials)  # gives average over times t to tau and 0 to t
            a1_bar += np.sum(chi1 * np.matmul(chi1, B))/(len(a0)*m_trials)
            a2_bar += np.sum(chi2 * np.matmul(chi2, B))/(len(a0)*m_trials)
            a3_bar += np.sum(chi3 * np.matmul(chi3, B))/(len(a0)*m_trials)
        Cs[i] = 0.25 * (mu_0*a0_bar - mu_1*a1_bar + 1j*mu_2*a2_bar - 1j*mu_3*a3_bar)

    spectrum = scipy.fft.fft(Cs)
    freqs = scipy.fft.fftfreq(len(Cs), float(tau_max)/n_taus)
    plt.figure()
    plt.plot(freqs, np.abs(spectrum))  # Looking for a big peak around Omega = 6Gamma
    plt.show()

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

    # Lindblad jump operators for q = 0, +1, -1 (denoted 2 bc minus is hard in name)
    d_q0 = 1j*np.zeros((6, 6))
    d_q0[2, 5] = 1/np.sqrt(2)
    d_q0[0, 3] = -1/np.sqrt(2)
    d_q1 = 1j * np.zeros((6, 6))
    d_q1[0, 4] = d_q1[1, 5] = 1/np.sqrt(2)
    d_q2 = 1j * np.zeros((6, 6))
    d_q2[1, 3] = d_q2[2, 4] = -1/np.sqrt(2)
    d_q = [d_q0, d_q1, d_q2]

    # First work in the Z basis
    psi_0 = np.array([1, 0, 0, 0, 0, 0])                     # Assume atom starts in the mz = -1 state
    psi_dark = np.array([1, 0, 1, 0, 0, 0])/np.sqrt(2)       # dark state in the Z basis

    d_y = 1j*np.zeros((len(psi_0), len(psi_0)))
    d_y[0, 4] = d_y[1, 5] = 1j
    d_y[1, 3] = d_y[2, 4] = -1j
    d_y *= 0.5
    # print(d_y)
    proj_e = 1j*np.zeros((len(psi_0), len(psi_0)))
    proj_e[3, 3] = proj_e[4, 4] = proj_e[5, 5] = 1
    # print(proj_e)
    h_eff = -0.5j * proj_e - 1 * Omega*0.5*(d_y + np.transpose(np.conjugate(d_y)))          # Effective Hamiltonian
    # print(h_eff - np.transpose(np.conjugate(h_eff)))

    def unravel(h=h_eff, c_0=psi_0, ls=d_q, steps=t_steps, t_i=t_initial, t_f=t_final):
        # Function for calculating a single unravelling
        dt = float(t_f - t_i) / steps  # time step
        c = 1j * np.zeros((steps, len(psi_0)))  # Wavefunction coefficients
        c[0] = c_0
        h_jump = h - np.transpose(np.conjugate(h))

        # Take a bunch of time steps in loop below to propagate wvfn
        for i in range(1, steps):
            dp = dt * 1j * np.dot(np.conjugate(c[i - 1]), np.matmul(h_jump, c[i - 1]))  # Jump prob
            epsilon = np.random.rand()
            jump = epsilon < dp
            if jump:
                cumulative_dpm = 0
                # Calculate each jump probability
                # If epsilon < Sum of dpm up to this point, select jump
                for L in ls:
                    c[i] = np.matmul(L, c[i - 1])  # Temporary, used to calc dpm
                    dpm = dt * np.abs(np.dot(c[i], np.conjugate(c[i])))
                    cumulative_dpm += dpm
                    if epsilon < cumulative_dpm:
                        c[i] /= np.sqrt(dpm / dt)
                        break
            else:
                # Evolve according to H_effective and re-normalize
                c[i] = c[i - 1] - 1j * dt * np.matmul(h, c[i - 1])
                c[i] = c[i] / np.abs((np.dot(c[i], np.conjugate(c[i]))))
        return c

    psi = unravel()
    p_d = np.abs(np.matmul(psi, psi_dark))**2  # probability of being in the dark state

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
        psi = unravel()
        p_ds[m] = np.abs(np.matmul(psi, psi_dark)) ** 2

    uncertainty = np.std(p_ds, 0)/np.sqrt(m_trials)

    ax[1, 0].errorbar(ts, np.mean(p_ds, 0), uncertainty)
    ax[1, 0].plot(ts, np.mean(p_ds, 0), 'r')  # Center line plotted in red for clarity
    ax[1, 0].set_title("Average of %i Trajectories" % m_trials)
    ax[1, 0].set_xlabel("Time (1/Gamma)")
    ax[1, 0].set_ylabel("Prob to be in |dark>")
    ax[1, 0].set_ylim(-0.05, 1)

    # Now repeat the problem in the Y basis
    psi_0y = np.array([0.5, 1j/np.sqrt(2), 0.5, 0, 0, 0])    # Assume atom starts in the mz = -1 state
    psi_darky = np.array([0, 1, 0, 0, 0, 0])                 # dark state in the y basis
    psi = 1j*np.zeros((t_steps, len(psi_0)))                 # Wavefunction at each time step
    psi[0] = psi_0

    d_yy = 1j*np.zeros((len(psi_0y), len(psi_0y)))
    d_yy[0, 3] = -1
    d_yy[2, 5] = 1
    d_yy *= 0.5
    # print(d_yy)

    h_effy = -0.5j * proj_e - 1 * Omega*0.5*(d_yy + np.transpose(np.conjugate(d_yy)))          # Effective Hamiltonian
    # print(h_effy)

    m_trials = 100                # Molmer averages 100 wvfns
    p_ds = np.zeros((m_trials, t_steps))

    for m in range(m_trials):
        psi = unravel(h=h_effy, c_0=psi_0y, ls=d_q)
        # check_norm = np.abs(np.linalg.norm(psi, axis=1)) ** 2         # make sure that the norm is 1
        # p_not_d = np.abs(np.matmul(psi, np.array([1, 0, 1, 1, 1, 1]))) ** 2  # probability of not dark state
        p_d = np.abs(np.matmul(psi, psi_darky)) ** 2  # probability of being in the dark state
        p_0 = np.abs(np.matmul(psi, np.array([1, 0, 0, 0, 0, 0]))) ** 2  # probability of being in -1 ground
        p_3 = np.abs(np.matmul(psi, np.array([0, 0, 0, 1, 0, 0]))) ** 2  # probability of being in -1 excited
        if m < 1:  # Plot trajectories up to this value on the individual trajectory plot
            ax[0, 1].plot(ts, p_d)
            # plt.figure(m)  #  Debugging, plot to look at various populations and norm over time
            # plt.plot(ts, p_0)
            # plt.plot(ts, p_3)
            # plt.plot(ts, p_d, 'k')
            # plt.plot(ts, check_norm)
            # plt.ylim(-0.05, 1.05)
        p_ds[m] = p_d
    ax[0, 1].set_title("Single Trajectory, Y Basis")
    ax[0, 1].set_ylim(-0.05, 1.05)

    uncertainty = np.std(p_ds, 0)/np.sqrt(m_trials)

    ax[1, 1].errorbar(ts, np.mean(p_ds, 0), uncertainty)
    ax[1, 1].plot(ts, np.mean(p_ds, 0), 'r')  # Center line plotted in red for clarity
    ax[1, 1].set_title("Average of %i Trajectories" % m_trials)
    ax[1, 1].set_xlabel("Time (1/Gamma)")
    ax[1, 1].set_ylim(-0.05, 1)


# run_1a()
# run_1b()
# run_2()
run_1c()

plt.show()

