import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve


import Part1_Free_vibration as part1
from Part1_Free_vibration import Signal
import part_2_1 as part21

def remove_idx(lst, idx):
    new_lst = np.zeros_like(lst, shape=len(lst)-1)
    i = 0 ; j = 0
    while i < len(lst):
        if i != idx:
            new_lst[j] = lst[i]
            j += 1
        i += 1
    return new_lst

def processing_data_exp(data_ks, ks):
    m = len(data_ks)
    signals_exp = np.zeros(shape=m, dtype=Signal)
    data_exp_length = np.zeros(m)
    for i in range(len(data_ks)):
        k = ks[i]['k']
        data_exp_length[i] = ks[i]['end'] - ks[i]['start']

        # Make the signal begin at t=0
        t = np.array(data_ks[i]['time'])
        t = t - t[0]

        a = np.array(data_ks[i]['x'])
        # Center signal
        a = a - np.mean(a)
        # Remove the ground acceleration part

        signal_a_exp = Signal(a, t)
        signals_exp[i] = signal_a_exp

    return signals_exp, data_exp_length

###################################
# === Q2.2 STEADY-STATE ===
###################################

def analytic_steady_state(k, n, t_max):
    t = np.linspace(0, t_max, n)
    omega_bar = k * omega
    freq_ratio = omega_bar / omega

    p0 = - m * a_g0
    a_g = a_g0 * np.sin(omega_bar * t)
    u_g = - a_g / (omega_bar**2)
    p_eff = m * a_g

    Rd = ( (1-(omega_bar/omega)**2)**2 + (2*xi*(omega_bar/omega))**2 ) **(-1/2)
    Ra = (omega_bar/omega)**2 * Rd
    if freq_ratio == 1:
        phi = np.pi/2   # 90° & arctan(90°) = inf
    else:
        phi = np.arctan( (2*xi*(omega_bar/omega)) / (1-(omega_bar/omega)**2))

    disp =  (p0/k) * Rd * np.sin(omega_bar*t - phi)
    acc = - (p0/m) * Ra * np.sin(omega_bar*t - phi)

    u_th = Signal(disp, t)
    a_th = Signal(acc, t)

    return u_th, a_th

def compute_steady_state_k_val(k_val, n, t_max):
    m = len(k_val)
    us_th = np.zeros(shape=m, dtype=Signal)
    as_th = np.zeros(shape=m, dtype=Signal)
    for i in range(m):
        k = k_val[i]
        u_th, a_th = analytic_steady_state(k, n, t_max)
        us_th[i] = u_th
        as_th[i] = a_th

    return us_th, as_th

def plot_analytic_steady_state(us_th, as_th, k_val, colours, title=""):
    m = len(k_val)

    if showPlot2:
        # 1. Plot all k_val acceleration signals
        fig, axes = plt.subplots(nrows=m, ncols=1, figsize=(8, 14))

        for i in range(m):
            colour = colours[i]
            k = k_val[i]
            omega_bar = k * omega

            axes[i].set_title(f'Acceleration for k = {k} & omega_bar = {omega_bar:.2f} [rad/s]')
            axes[i].plot(as_th[i].t, as_th[i].u, label="a_th", color=colour, lw=1)

        axes[-1].set_xlabel('Time [s]')
        for axe in axes:
            axe.set_xlim(left=0)
            axe.legend(loc='upper right')
            axe.grid()

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)  # space between plot
        if saveFig2:
            plt.savefig(f"{repository}/Q2.2_Steady_state_response{title}.png")
        plt.show()


        # 2. 2 plots of the acceleration and the displcement for different k_val
        plt.figure(figsize=(8, 3))
        for i in range(m):
            plt.plot(as_th[i].t, as_th[i].u, label=f"k={k_val[i]}", color=colours[i], lw=1)

        plt.ylim(top=2.5, bottom=-2.5)
        plt.xlim(left=0)

        plt.title(f'Harmonic loading - analytical response (omega_n = {omega:.2f} [rad/s])')
        plt.xlabel("Time, t [s]")
        plt.ylabel("Acceleration, a [m/s²]")
        plt.legend(loc='upper right')
        plt.grid()
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)  # space between plot
        if saveFig2:
            plt.savefig(f"{repository}/Q2.2_Steady_state_response_compare_k_val{title}.png")
        plt.show()

    return us_th, as_th


def plot_compare_th_exp(k_val, n, data_exp, data_exp_length, colours):
    if len(data_exp) != len(k_val):
        print("ERROR: different length for list u_s, a_s, data_ks")

    m = len(k_val)
    if showPlot2:
        fig, axes = plt.subplots(nrows=m, ncols=1, figsize=(8, 12))

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
        plt.show()


def plot_freq_response_curve_compare_exp_th(k_val, amplitude_th, amplitude_exp, ylim=None, title=""):
    plt.figure(figsize=(8,6))
    plt.plot(k_val, amplitude_exp, label="Experimental", color="blue", marker='o')
    plt.plot(k_val, amplitude_th, label="Theorical", color="orange", marker='o')
    plt.ylim(0, ylim)

    plt.title("Comparison of the frequency response curve")
    plt.xlabel(r'Frequancy ratio, $\frac{\bar\omega} {\omega}$')
    plt.ylabel('Maximum peak of the acceleration signal [m/s²]')
    plt.grid()
    plt.legend()
    if showPlot2:
        plt.savefig(f"{repository}/Q2.2_Compare_freq_response_curve{title}.png")
    plt.show()


def Duhamel(a0, tp, n, t_max):
    # 1. Compute the time from 0 to t_max
    amplitude = a0 * g
    t = np.linspace(0, t_max, n)


    # 2. Build ground acceleration a_g of the two-sided acceleration impulse
    a_g = np.zeros_like(t)
    T_ag = 3*tp
    for i in range(n):
        phase = t[i] % T_ag
        if t[i] % T_ag < tp:
            a_g[i] = +amplitude
        elif phase >= tp:
            a_g[i] = -amplitude/2
    p_g = m * a_g

    # 3. Duhamel integral
    omega_d = omega * np.sqrt(1-xi**2)
    s = np.arange(0, n) * dt # interval (t - τ)
    # 3.1 Unit impulse-response function
    h = (1/(m*omega_d)) * np.exp(-xi*omega*s) * np.sin(omega_d*s)
    h[0] = 0

    # 3.2 Convolution: u(t) = - ∫0^t h(t-τ) p_g(τ) dτ
    u_rel = dt * fftconvolve(p_g, h)[:n]

    # 4. Compute relative velocity & acceleration
    v_rel = np.gradient(u_rel, dt)
    a_rel = np.gradient(v_rel, dt)

    # 5. Compute absolute acceleration
    a_abs = a_g + a_rel

    signal_a_abs = Signal(a_abs, t)
    signal_a_g = Signal(a_g, t)
    signal_a_rel = Signal(a_rel, t)
    signal_v_rel = Signal(v_rel, t)
    signal_u_rel = Signal(u_rel, t)

    return signal_a_abs, signal_a_g, signal_a_rel, signal_v_rel, signal_u_rel

def plot_Duhamel_steps(a_abs, a_g, a_rel, v_rel, u_rel):
    # Plot application of Duhamel'integral
    if showPlot2:
        fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(8, 8))
        axes[0].set_title(f"Signals computed using Duhamel's integral")

        axes[0].plot(a_abs.t, a_abs.u, label="a_abs", color='purple', lw=1)
        axes[1].plot(a_g.t,   a_g.u,   label="a_g",   color='brown',  lw=0.8, ls='--')
        axes[2].plot(a_rel.t, a_rel.u, label="a_rel", color='orange', lw=0.8, ls='--')
        axes[3].plot(v_rel.t, v_rel.u, label="v_rel", color='green',  lw=1)
        axes[4].plot(u_rel.t, u_rel.u, label="u_rel", color='blue',   lw=1)

        axes[-1].set_xlabel("Time, t [s]")
        for axe in axes:
            axe.set_xlim(left=0)
            axe.legend(loc='upper right')
            axe.grid()

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)  # space between plot
        if saveFig2:
            plt.savefig(f"{repository}/Q2.3_Duhamel_response_steps.png")
        plt.show()

def plot_Duhamel_disp(u_rel):
    plt.figure(figsize=(10, 3))
    plt.plot(u_rel.t, u_rel.u, label="u_rel", color='blue', lw=1)

    plt.title("Duhamel's response")
    plt.xlabel("Time, t [s]")
    plt.ylabel('Displacement, u [m]')
    plt.grid()
    plt.legend(loc='upper right')
    if showPlot2:
        plt.savefig(f"{repository}/Q2.3_Duhamel_disp.png")
    plt.show()


# ========== PARAMETERS ==========
repository = "Figures"
showPlot2 = False
saveFig2 = True

g = 9.81  # [m/s²]
a_g0 = 0.1 * g # [m/s²]

tp = 0.1  # time of pulse diff
a0 = 0.075 # Amplitude of the two-sided impulse

if __name__ == "__main__":
    showPlot2 = True

# Retrieved parameters from Part 1 and Part 2.1
dt = part1.dt
k = part1.k
m = part1.m
omega = part1.params_Part1["omega"]  # take natural frequency computed in part 1.4
T = part1.params_Part1["T"]
xi = part1.params_Part1["xi"]
c = part1.params_Part1["c"]

print("\n=== Parameters from part 1 ===")
print(f"natural freq considered, omega_n = {omega} [rad/s] ; natural period, T = {T}")
print(f"stiffness, k = {k} [N/m] ; damping ratio, xi = {xi} ; damping coeff, c = {c}")
print(f"Other Parameters: \n"
      f"Peak acceleration, a_g0 = {a_g0}")

# Retrieved parameters from Part 2.1
print("\n=== Parameters part 2.1 ===")

data_ks = part21.data_ks # acceleration signals recorded for each ki
ks = part21.ks # dico containing start & end of the complet signal for each ki
signals_exp, data_exp_length = processing_data_exp(data_ks, ks)


# ========== Q2.2 Analytical steady-state response ==========
print("\n=== Q2.2 ===")

k_val = [0.5, 0.75, 1, 1.5, 2] # index such that omega_bar = k_val * omega_n
colours = ['red', 'magenta', 'orange', 'blue', 'green']
t_max = 5
n = int( (t_max/T) * 50 ) # Nb points for the analytical signals

# ----- 1. Calculate steady-state for diff k_val -----
us_th, as_th = compute_steady_state_k_val(k_val, n, t_max)
us_max = [np.max(us_th[i].u) for i in range(len(us_th))]
as_max = [np.max(as_th[i].u) for i in range(len(as_th))]
print(f"Max amplitude us : {np.max(us_max)} and for as : {np.max(as_max)}")

# ----- 2. Plot steady-state response (acc/disp and all k_val) with and without k=1  -----
plot_analytic_steady_state(us_th, as_th, k_val, colours=colours, title="")
plot_analytic_steady_state(remove_idx(us_th, 2), remove_idx(as_th, 2), remove_idx(k_val, 2), colours=remove_idx(colours, 2), title="_without_k3")

# ----- 3. Compare exp and th acceleration response -----
plot_compare_th_exp(k_val, n, signals_exp, data_exp_length, colours)

# ----- 4. Compute and compare frequency response curve for exp and th response -----
amplitude_th = [ np.max(as_th[i].u) for i in range(len(as_th)) ]
amplitude_exp = [ np.max(signals_exp[i].u) for i in range(len(signals_exp)) ]
plot_freq_response_curve_compare_exp_th(k_val, amplitude_th, amplitude_exp)


# ========== Q2.3 Duhamel’s integral ==========
print("\n=== Q2.3 ===")
print(f"Impulse amplitude, a0 = {a0} [g] ; Impulse period, tp = {tp} [s]")

t_max = 40 #[s]
n = int( (t_max/tp)*10 )
a_abs, a_g, a_rel, v_rel, u_rel = Duhamel(a0, tp, n, t_max)
#plot_Duhamel_steps(a_abs, a_g, a_rel, v_rel, u_rel)
#plot_Duhamel_disp(u_rel)



print("\nEnd Part 2")