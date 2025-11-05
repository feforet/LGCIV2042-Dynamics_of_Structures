import numpy as np
import matplotlib.pyplot as plt


import Part1_Free_vibration as part1
import part_2_1 as part21



def processing_data_exp(data_ks, ks):
    m = len(data_ks)
    signals_exp = np.zeros(shape=m, dtype=part1.Signal)
    data_exp_length = np.zeros(m)
    for i in range(len(data_ks)):
        data_exp_length[i] = ks[i]['end'] - ks[i]['start']

        t = np.array(data_ks[i]['time'])
        a = np.array(data_ks[i]['x'])
        t = t - t[0]

        signal_a_exp = part1.Signal(a, t)
        signals_exp[i] = signal_a_exp

    return signals_exp, data_exp_length

###################################
# === Q2.2 STEADY-STATE ===
###################################

def analytic_steady_state(k, n, t_max):
    t = np.linspace(0, t_max, n)
    omega_bar = k * omega

    """u_static = (p0/k) * np.sin(omega_bar * t)
    a_static = - (p0/k) * np.cos(omega_bar * t)
    u = u_static * 1 / (1 - (omega_bar/omega)**2)
    a = a_static * 1 / (1 - (omega_bar/omega)**2)"""

    Rd = ( (1-(omega_bar/omega)**2)**2 + (2*xi*(omega_bar/omega))**2 ) **(-1/2)
    Ra = (omega_bar/omega)**2 * Rd
    phi = np.arctan( (2*xi*(omega_bar/omega)) / (1-(omega_bar/omega))**2)

    u = (p0/k) * Rd * np.sin(omega_bar*t - phi)
    a = - (p0/m) * Ra * np.sin(omega_bar*t - phi)

    u_th = part1.Signal(u, t)
    a_th = part1.Signal(a, t)

    return u_th, a_th


def plot_analytic_steady_state(k_val, n, t_max):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
    axes[0].set_title('Harmonic loading - analytical response')
    axes[0].set_xlabel("Time, t [s]")
    axes[0].set_ylabel("Displacement, u [m]")

    axes[1].set_xlabel("Time, t [s]")
    axes[1].set_ylabel("Acceleration, u [m/s²]")

    m = len(k_val)
    us_th = np.zeros(shape=m, dtype=part1.Signal)
    as_th = np.zeros(shape=m, dtype=part1.Signal)
    for i in range(m):
        k = k_val[i]
        omega_bar = k * omega
        u_th, a_th = analytic_steady_state(k, n, t_max)
        us_th[i] = u_th
        as_th[i] = a_th

        axes[0].plot(u_th.t, u_th.u, label=f"omega_bar={omega_bar:.2f} [rad/s]", lw=1)
        axes[1].plot(a_th.t, a_th.u, label=f"omega_bar={omega_bar:.2f} [rad/s]", lw=1)

    for axe in axes:
        axe.set_xlim(left=0)
        axe.legend(loc='upper right')
        axe.grid()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)  # space between plot
    if saveFig2:
        plt.savefig(f"{repository}/Q2.2_Steady_state_response.png")
    if showPlot2:
        plt.show()

    return us_th, as_th


def plot_compare_th_exp(k_val, data_exp, data_exp_length):
    if len(data_exp) != len(k_val):
        print("ERROR: different length for list u_s, a_s, data_ks")

    n = 200
    m = len(k_val)
    colours = ['red', 'blue', 'green', 'yellow', 'pink']


    fig, axes = plt.subplots(nrows=m, ncols=1, figsize=(8, 12))
    axes[0].set_title('Harmonic loading - comparison of the acceleration response')

    for i in range(m):
        colour = colours[i]
        k = k_val[i]
        omega_bar = k * omega

        a_exp = data_exp[i]
        _, a_th = analytic_steady_state(k, n, data_exp_length[i])

        axes[i].set_title(f'Acceleration for k = {k} & omega_bar = {omega_bar:.2f} [rad/s]')
        axes[i].plot(a_exp.t, a_exp.u, label="a_exp", color=colour, lw=1)
        axes[i].plot(a_th.t, a_th.u, label="a_th", color='orange', lw=1, ls='--')


    axes[-1].set_xlabel('Time [s]')
    for axe in axes:
        axe.set_xlim(left=0)
        axe.legend(loc='upper right')
        axe.grid()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)  # space between plot
    if saveFig2:
        plt.savefig(f"{repository}/Q2.2_compare_steady_state_with_exp.png")
    if showPlot2:
        plt.show()




# ========== PARAMETERS ==========
repository = "Figures/Part2"
showPlot2 = True
saveFig2 = False

p0 = 1  # Amplitude of the harmonic load [N]

# Retrieved parameters from Part 1 and Part 2.1
k = part1.k
m = part1.m
omega = part21.omega_n_exp  # take natural frequency from the experimental signal (Q2.1)
xi = part1.params_Part1["xi"]


print("\n=== Parameters part 1 ===")
print(f"stiffness, k = {k} [N/m] ; natural freq considered, omega_n = {omega} [rad/s] ; damping ratio, xi = {xi}")
print(f"Other Parameters: \n"
      f"Initial loading, p0 = {p0}")


# Retrieved parameters from Part 2.1
print("\n=== Parameters part 2.1 ===")

data_ks = part21.data_ks # acceleration signals recorded for each ki
ks = part21.ks # dico containing start & end of the complet signal for each ki
signals_exp, data_exp_length = processing_data_exp(data_ks, ks)
print(signals_exp)
print(data_exp_length)

# ========== Q2.2 Analytical steady-state response ==========
print("\n=== Q2.2 ===")
k_val = [0.5, 0.75, 1, 1.5, 2] # index such that omega_bar = k_val * omega_n

us_th, as_th = plot_analytic_steady_state(k_val, n=200, t_max=120)
plot_compare_th_exp(k_val, signals_exp, data_exp_length)


print("\nEnd")

