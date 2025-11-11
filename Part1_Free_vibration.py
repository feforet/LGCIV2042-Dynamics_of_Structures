import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sc


###########################################
# === CLASSES POUR SYSTEME ET SIGNAUX === #
###########################################

class system:
    """ Classe représentant un système dynamique (accélération, vitesse, déplacement).
    Peut aussi stocker la période T et le coefficient d’amortissement xi."""

    def __init__(self, a, v, u, T=None, xi=None):
        self.a = a
        self.v = v
        self.u = u

        self.T = T
        self.xi = xi


class Signal:
    """
    Classe représentant un signal temporel u(t).
    Attributs:
        u : valeurs du signal
        t : vecteur temps
        size : taille du signal
    """

    def __init__(self, u, t):
        self.u = u
        self.t = t
        self.size = len(u)
        if (len(u) != len(t)):
            print("ERROR: u and t must have the same length")
            self.size = -1

    def __len__(self):
        return self.size

    def copy(self):
        return Signal(np.copy(self.u), np.copy(self.t))

    def trim_index(self, index_0, index_f):
        """Découpe le signal entre deux indices."""
        return self.u[index_0:index_f], self.t[index_0:index_f]

    def trim_time(self, time_0, time_f, precision):
        """Découpe le signal entre deux instants temporels (avec tolérance de précision)."""
        index_0 = find_value(self.t, time_0, precision)
        index_f = find_value(self.t, time_f, precision)
        return self.u[index_0:index_f], self.t[index_0:index_f]

    def trim_signal(self, val_0, val_f, precision):
        """Découpe le signal entre deux valeurs (ex : amplitude)."""
        index_0 = find_value(self.u, val_0, precision)
        index_f = find_value(self.u, val_f, precision)
        return self.u[index_0:index_f], self.t[index_0:index_f]

    def set_to_zeros(self, index_0, index_f):
        """Met à zéro le signal entre deux indices donnés."""
        new_u = self.u
        for i in range(index_0, index_f):
            new_u[i] = 0
        return new_u

    def center_signal(self):
        """ Re-centre le signal par rapport à sa moyenne """
        u_old = self.u
        self.u = u_old - np.mean(u_old)

    def filter(self, freq, type='high'):
        """Filtrage Butterworth passe-haut ou passe-bas du signal."""
        b, a = sc.butter(2, 0.5 / (0.5 * freq), btype=type)
        new_u = sc.filtfilt(b, a, self.u)
        return new_u


###################################
# === FONCTIONS UTILITAIRES === #
###################################

def find_value(x, value, precision=0.0):
    """Recherche l’index d’une valeur donnée dans un vecteur, avec tolérance."""
    index = -1
    for i in range(len(x)):
        if (value - precision) < x[i] < (value + precision):
            index = i

    if index == -1:
        print(f"Value {value} not found in list x :\n{x}")
        raise Exception(f"Value {value} not found in list x")

    return index


def read_data(filename):
    data = pd.read_csv(filename, sep=';', header=0, dtype=float)
    return data


def plotter(data):
    """Affiche les signaux d’accélération bruts (ax, ay, az)."""
    if showPlot:
        for col in data.columns:
            if col.startswith('a'):
                plt.plot(data['time'], data[col], label=col[1])

        plt.title('Raw data')
        plt.xlabel('Time [s]')
        plt.ylabel('Acceleration [m/s^2]')
        plt.grid()
        plt.legend()
        if saveFig:
            plt.savefig("Figures/Plot_acceleration_init")
        print("\tPLOT - plotter")
        plt.show()


def data_to_signals(data):
    """Convertit les colonnes de DataFrame en objets signal."""
    ax = Signal(np.array(data['ax']), np.array(data['time']))
    ay = Signal(np.array(data['ay']), np.array(data['time']))
    az = Signal(np.array(data['az']), np.array(data['time']))
    return ax, ay, az


def clean_signals(signals, t0, tf, precision):
    """Découpe un ensemble de signaux sur la plage temporelle [t0, tf]."""
    new_signals = np.empty(len(signals), dtype=object)  # tableau d’objets
    for i in range(len(signals)):
        x = signals[i]
        u, t = x.trim_time(t0, tf, precision)
        new_signals[i] = Signal(u, t)
    return new_signals


def integrate_trapezoid(x, dt, x0=0.0):
    """Intégration trapézoïdale cumulée."""
    y = np.zeros(len(x))
    for i in range(len(x) - 1):
        y[i] = y[i - 1] + (x[i] + x[i - 1]) * dt / 2
    return y


#######################################
# === Q1.2 DERIVE DISPLACEMENT & u0 ===
#######################################

def get_ux(data, optionFilter):
    """Full pipeline to compute ux (displacement) from acceleration data."""
    ax_init, _, _, = data_to_signals(data)
    [ax] = clean_signals([ax_init], cleaning_t0, cleaning_tf, 0.001)

    # ----- 1.1 Traitement du signal ax -----
    ax.center_signal()
    if optionFilter:
        pass
        # ax.u = ax.filter(fs, 'high')

    # ----- 2. Integration numérique de ax -----
    velocity = integrate_trapezoid(ax.u, dt, 0.0)  # Vitesse (v(0) = 0)
    vx = Signal(velocity, ax.t)

    # ----- 2.1 Traitement du signal vx -----
    [vx_trim] = clean_signals([vx], cleaning_t0 + 5, cleaning_tf - 1, 0.001)
    vx.u = vx.u - np.mean(vx_trim.u)
    if optionFilter:
        vx.u = vx.filter(fs, 'high')

    # ----- 3. Integration numérique de vx -----
    disp = integrate_trapezoid(vx.u, dt, 0.0)  # Déplacement relatif
    ux = Signal(disp, vx.t)

    # ----- 3.1 Traitement du signal ux -----
    [ux_trim] = clean_signals([ux], cleaning_t0 + 10, cleaning_tf - 2, 0.001)
    ux.u = ux.u - np.mean(ux_trim.u)
    if optionFilter:
        pass
        # ux.u = ux.filter(fs, 'high')

    return ax, vx, ux


def plot_comparison_with_without_filter(ax_filter, vx_filter, ux_filter, ax_nonFilter, vx_nonFilter, ux_nonFilter):
    """ Compare signals obtained with and without filtering."""
    if showPlot:

        # Plot 3 signals
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 6))
        axes[0].set_title("Comparison of obtained signals with and without filtering")

        axes[0].plot(ax_filter.t, ax_filter.u, label="ax [m/s²] (filtered)", color='purple', lw=1)
        axes[0].plot(ax_nonFilter.t, ax_nonFilter.u, label="ax [m/s²] (non-filtered)", color='orange', lw=0.5,
                     linestyle='--')
        axes[1].plot(vx_filter.t, vx_filter.u, label="vx [m/s] (filtered)", color='green', lw=1)
        axes[1].plot(vx_nonFilter.t, vx_nonFilter.u, label="vx [m/s] (non-filtered)", color='orange', lw=0.5,
                     linestyle='--')
        axes[2].plot(ux_filter.t, ux_filter.u, label="ux [m] (filtered)", color='blue', lw=1)
        axes[2].plot(ux_nonFilter.t, ux_nonFilter.u, label="ux [m] (non-filtered)", color='orange', lw=0.5, linestyle='--')

        axes[-1].set_xlabel('Time [s]')
        for axe in axes:
            axe.set_xlim(left=0)
            axe.legend(loc='upper right')
            axe.grid()

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)  # space between plot
        if saveFig:
            plt.savefig(f"Figures/Q1.2_Compare_with_without_filter.png")
        plt.show()


def plot_signals(ax, vx, ux, title=""):
    # Compare 3 signals (ax, vx, ux)
    if showPlot:
        plt.figure(figsize=(8, 6))
        plt.plot(ax.t, ax.u, label="ax [m/s²]", color='purple', lw=1)
        plt.plot(vx.t, vx.u, label="vx [m/s]", color='green', lw=1)
        plt.plot(ux.t, ux.u, label="ux [m]", color='blue', lw=1)

        plt.title(f'{title} - Acceleration, velocity and displacement')
        plt.xlabel('Time [s]')
        plt.xlim(left=0)
        plt.legend(loc='upper right')
        plt.grid()
        if saveFig:
            plt.savefig(f"Figures/Q1.2_Compare_ax_vx_ux_together_{title}.png")
        plt.show()

        # Plot 3 signals
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 6))
        axes[0].set_title(title)
        axes[0].plot(ax.t, ax.u, label="ax [m/s²]", color='purple', lw=1)
        axes[1].plot(vx.t, vx.u, label="vx [m/s]", color='green', lw=1)
        axes[2].plot(ux.t, ux.u, label="ux [m]", color='blue', lw=1)

        axes[-1].set_xlabel('Time [s]')
        for axe in axes:
            axe.set_xlim(left=0)
            axe.legend(loc='upper right')
            axe.grid()

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)  # space between plot
        if saveFig:
            plt.savefig(f"Figures/Q1.2_Compute_ax_vx_ux_{title}.png")
        plt.show()


def get_u0(ux):
    """
    Estimate initial displacement u0 from displacement signal ux.
    Steps:
        - trim the signal to the stable oscillation zone
        - remove DC offset
        - take maximum absolute value
    """
    # Trim  signal of the instable oscillation due to integration errors
    # (at the beginning and the end)
    t0_trim = 11
    tf_trim = 119
    ux_trim, t_trim = ux.trim_time(t0_trim, tf_trim, 0.001)
    u_trim = Signal(ux_trim, t_trim)

    # Centering the trimmed signal
    ux_trim = ux_trim - np.mean(ux_trim)
    u0 = np.max(abs(ux_trim))

    return u0, u_trim, t0_trim, tf_trim


def plot_u0(u0, ux, ux_trim):
    if showPlot:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

        # Plot original signal
        axes[0].set_title('Displacement ux')
        axes[0].plot(ux.t, ux.u, label="ux [m/s²]", color='blue', lw=1)
        axes[0].axhline(y=u0, color='r', linestyle='--', label='u0 max={:.3f}'.format(u0), lw=1)
        axes[0].axhline(y=-u0, color='r', linestyle='--', lw=1)

        # Plot trimmed signal
        axes[1].set_title('Trimmed signal to estimate u0')
        axes[1].plot(ux_trim.t, ux_trim.u, label="ux_trim [m/s]", color='pink', lw=1)
        axes[1].axhline(y=u0, color='r', linestyle='--', label='u0 max={:.3f}'.format(u0), lw=1)
        axes[1].axhline(y=-u0, color='r', linestyle='--', lw=1)

        axes[-1].set_xlabel('Time [s]')
        for axe in axes:
            axe.set_xlim(left=0)
            axe.legend(loc='upper right')
            axe.grid()

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)  # space between plot
        if saveFig:
            plt.savefig("Figures/Q1.2_u0_estimation.png")
        plt.show()


################################################
# === Q1.4 UNDAMPED-SYST RESPONSE & GET FREQ ===
################################################

def get_natural_frequency(ax):
    """Estimate natural frequency from acceleration signal.
        :return
            T_mean = Average period of all the period values, T_values
            T_values = period values from differences between
            peaks & troughs = index of the peaks/troughs of signal ax
    """

    # Estimated period
    T_guess = T_th  # here ~0.409 [s]
    min_distance = int(0.8 * T_guess / dt)  # at least 80% of T between 2 pics
    # Detect maxima
    peaks, _ = sc.find_peaks(ax.u, distance=min_distance, prominence=0.05)
    # Detect minim (invert signal)
    troughs, _ = sc.find_peaks(-ax.u, distance=min_distance, prominence=0.05)

    T_values_peaks = np.diff(ax.t[peaks])  # periods between 2 peaks
    T_values_troughs = np.diff(ax.t[troughs])  # periods between 2 troughs
    T_values = np.concatenate([T_values_peaks, T_values_troughs])

    T_peaks_mean = np.mean(T_values_peaks)
    T_troughs_mean = np.mean(T_values_troughs)
    T_mean = np.mean(T_values)

    print(f"Estimated natural period Tn from peaks: {T_peaks_mean}")
    print(f"Estimated natural period Tn from troughs: {T_troughs_mean}")

    return T_mean, T_values, peaks, troughs


def plot_natural_freq(ax, T_values, peaks, troughs):
    """Affiche le signal avec ses pics/troughs détectés et histogramme des périodes."""
    [ax_trim] = clean_signals([ax], cleaning_t0, cleaning_tf, 0.001)

    # Plot signal with peaks and troughs
    if showPlot:
        plt.plot(ax_trim.t, ax_trim.u, label="ax")
        plt.plot(ax_trim.t[peaks], ax_trim.u[peaks], label="peaks")
        plt.plot(ax_trim.t[troughs], ax_trim.u[troughs], label="troughs")

        plt.title("Check for peaks and troughs in the signal")
        plt.xlabel('Time [s]')
        plt.ylabel('Acceleration ax [m/s²]')
        plt.grid()
        plt.legend()
        if showPlot:
            plt.savefig("Figures/Q1.4_peak_throughs.png")
        plt.show()

        # Histogram of T values
        T_mean = np.mean(T_values)
        plt.hist(T_values, bins=100, edgecolor='black')
        plt.plot([T_mean, T_mean], [0, 50], label=f'T_mean={T_mean:.3f}', color='red')
        plt.title(f'Histogram of periods T between peaks/troughs (ax signal)')
        plt.xlabel('Period T [s]')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid()
        if saveFig:
            plt.savefig("Figures/Q1.4_Histogram_T_values_from_ax.png")
        plt.show()


def compute_experimental_signals_on_dtVal(ax, vx, ux, dt_val, showPlot):
    """Estimate damping ratio xi using logarithmic decrement method."""
    # ===== 1. Experimental data =====
    # ----- 1. Trim the signals to [cleaning_t0, cleaning_tf] -----
    [ax_clean, vx_clean, ux_clean] = clean_signals([ax, vx, ux], cleaning_t0, cleaning_tf, 0.001)
    ax_clean.t = ax_clean.t - ax_clean.t[0]  # Rebase time to 0
    vx_clean.t = vx_clean.t - vx_clean.t[0]
    ux_clean.t = ux_clean.t - ux_clean.t[0]

    # ----- 2. Save old signals for comparison -----
    old_ax = ax_clean.copy()
    old_vx = vx_clean.copy()
    old_ux = ux_clean.copy()

    # ----- 3. Shift the signal so that the th and exp signals start at the same time  -----
    T_guess = T_th  # estimated period (here ~0.41 [s])
    min_distance = int(0.8 * T_guess / dt)  # at least 80% of T between 2 pics

    # ----- 3.1 Shift signal ax (begin at first troughs) -----
    troughs, _ = sc.find_peaks(-ax_clean.u, distance=min_distance, prominence=0.05)
    first_trough_ax = troughs[0]
    ax_clean.u = ax_clean.u[first_trough_ax:]
    ax_clean.t = ax_clean.t[first_trough_ax:]
    ax_clean.t = ax_clean.t - ax_clean.t[0]  # Rebase time to 0

    # ----- 3.2 Shift signal vx (begin at first troughs of ax) -----
    vx_clean.u = vx_clean.u[first_trough_ax:]
    vx_clean.t = vx_clean.t[first_trough_ax:]
    vx_clean.t = vx_clean.t - vx_clean.t[0]

    # ----- 3.3 Shift signal ux (begin at first troughs of ax) -----
    ux_clean.u = ux_clean.u[first_trough_ax:]
    ux_clean.t = ux_clean.t[first_trough_ax:]
    ux_clean.t = ux_clean.t - ux_clean.t[0]

    # ----- 4 Trim the signals to [0, dt_val] -----
    [ax_t20, vx_t20, ux_t20] = clean_signals([ax_clean, vx_clean, ux_clean], 0, dt_val, 0.001)
    [old_ax, old_vx, old_ux] = clean_signals([old_ax, old_vx, old_ux], 0, dt_val, 0.001)

    # ----- 5. Check shifted signals -----
    if showPlot:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 6))
        axes[0].set_title('Check experimental signals shifted to the first troughs of ax')

        axes[0].plot(ax_t20.t, ax_t20.u, label='shifted', color="purple", lw=1)
        axes[0].plot(old_ax.t, old_ax.u, label='old', color="orange", lw=1)
        # axes[0].plot(old_ax.t[troughs], old_ax.u[troughs], 'x', color='red', label='troughs ax', lw=1)
        axes[0].set_ylabel('Acceleration [m/s²]')

        axes[1].plot(vx_t20.t, vx_t20.u, label='shifted', color="green", lw=1)
        axes[1].plot(old_vx.t, old_vx.u, label='old', color="orange", lw=1)
        axes[1].set_ylabel('Velocity [m/s]')

        axes[2].plot(ux_t20.t, ux_t20.u, label='shifted', color="blue", lw=1)
        axes[2].plot(old_ux.t, old_ux.u, label='old', color="orange", lw=1)
        axes[2].set_ylabel('Displacement [m]')

        axes[-1].set_xlabel('Time [s]')
        for axe in axes:
            axe.set_xlim(left=0)
            axe.legend(loc='upper right')
            axe.grid()

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)  # space between plot
        if saveFig:
            plt.savefig("Figures/Q1.4_Experimental_signals_shifted_to_first_peak.png")
        plt.show()

    return ax_t20, vx_t20, ux_t20

def free_undamped_response(omega, dt_val,  u0, v0=0.0):
    # ----- Theoretical response of free undamped system -----
    t = np.linspace(0, dt_val, len(ux_t20))
    u = u0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)
    v = - u0 * omega * np.sin(omega * t) + v0 * np.cos(omega * t)
    a = - omega_th ** 2 * u

    a_th = Signal(a, t)
    v_th = Signal(v, t)
    u_th = Signal(u, t)

    return a_th, v_th, u_th

def plot_free_undamped_syst_comparison(ax_t20, vx_t20, ux_t20, omega, dt_val, u0, v0=0.0, saveFig=False):
    """Compute theorical response (with omega given) & compare therical-experiment ax, vx, ux"""
    # ----- Theoretical response of free undamped system -----
    a_th, v_th, u_th = free_undamped_response(omega, dt_val, u0, v0)

    # ----- Plotting comparison theory (with omega) and experiment -----
    if showPlot:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 6))
        axes[0].set_title(f'Free undamped system:\nComparison theorical (omega={omega:.3f}) - experimental response')

        # Plot 3 pairs of signals (th vs exp)
        axes[0].plot(ax_t20.t, ax_t20.u, label="ax_exp", color='purple', lw=1)
        axes[0].plot(a_th.t, a_th.u, label="ax_th", color='orange', lw=1, ls='--')
        axes[0].set_ylabel('Acceleration [m/s²]')

        axes[1].plot(vx_t20.t, vx_t20.u, label="vx_exp", color='green', lw=1)
        axes[1].plot(v_th.t, v_th.u, label="vx_th", color='orange', lw=1, ls='--')
        axes[1].set_ylabel('Velocity [m/s]')

        axes[2].plot(ux_t20.t, ux_t20.u, label="ux_exp", color='blue', lw=1)
        axes[2].plot(u_th.t, u_th.u, label="ux_th", color='orange', lw=1, ls='--')
        axes[2].set_ylabel('Displacement [m]')

        axes[-1].set_xlabel('Time [s]')
        for axe in axes:
            axe.set_xlim(left=0)
            axe.legend(loc='upper right')
            axe.grid()

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)  # space between plot
        if saveFig:
            plt.savefig(f"Figures/Q1.4_Free_undamped_response_comparison_(omega={omega:2f}).png")
        plt.show()

    return a_th, v_th, u_th


####################################
# === Q1.5 COMPUTE DAMPING RATIO ===
####################################

def get_damping_coeff(ax, peaks, troughs):
    """Calcule le rapport d’amortissement ξ par la méthode du décrément logarithmique.
    Utilise les maxima et minima du signal pour plus de robustesse."""

    n_peaks = len(peaks)
    n_troughs = len(troughs)

    delta_peaks = np.zeros(n_peaks)
    delta_troughs = np.zeros(n_troughs)

    for i in range(n_peaks - 1):
        delta_peaks[i] = np.log(abs(ax.u[peaks[i]]) / abs(ax.u[peaks[i + 1]]))

    for i in range(n_troughs - 1):
        delta_troughs[i] = np.log(abs(ax.u[troughs[i]]) / abs(ax.u[troughs[i + 1]]))

    xi_peaks = delta_peaks / (2 * np.pi)
    xi_troughs = delta_troughs / (2 * np.pi)
    xi_peaks_mean = np.mean(xi_peaks)
    xi_troughs_mean = np.mean(xi_troughs)

    xi_mean = np.mean([xi_peaks_mean, xi_troughs_mean])

    print(f"Estimated damping ratio xi from peaks: {xi_peaks_mean}")
    print(f"Estimated damping ratio xi from troughs: {xi_troughs_mean}")

    return xi_mean, xi_peaks, xi_troughs


def plot_hist_xi(xi_peaks, xi_troughs):
    """Plot histogram of damping ratios estimated from peaks and troughs."""
    xi_values = np.concatenate([xi_peaks, xi_troughs])
    xi_mean = np.mean(xi_values)
    if showPlot:
        # Histogram of xi values for peaks and troughs
        plt.plot([xi_mean, xi_mean], [0, 50], label=f'xi_mean={xi_mean:.5f}', color='red')
        plt.hist(xi_values, bins=100, edgecolor='black')
        # plt.hist(xi_peaks, bins=100, edgecolor='black', label="xi_peaks")
        # plt.hist(xi_troughs, bins=100, edgecolor='black', label="xi_troughs")

        plt.title('Histogram of damping ratio xi')
        plt.xlabel('Damping ratio xi []')
        plt.ylabel('Frequency')
        plt.grid()
        plt.legend(loc='upper right')
        if saveFig:
            plt.savefig("Figures/Q1.5_Histogram_damping_ratio_xi.png")
        plt.show()


###################################
# === Q1.6 DAMPED-SYST RESPONSE ===
###################################

def damped_response(dt_val, xi, omega, u0, v0=0.0):
    """
    Compute theoretical response of a damped oscillator.
    Parameters:
        dt_val (float): duration [s]
        xi (float): damping ratio
        omega (float): natural pulsation [rad/s]
        u0 (float): initial displacement
        v0 (float): initial velocity (default 0)
    Returns:
        a_theo, v_theo, u_theo as signal objects
    """
    t = np.linspace(0, dt_val, 1000)

    # Suppose under-damped
    omega_d = omega * np.sqrt(1 - xi ** 2)
    A = u0
    B = ((v0 + xi * omega * u0) / omega_d)

    u = np.exp(-xi * omega * t) * (A * np.cos(omega_d * t) + \
                                   B * np.sin(omega_d * t))

    v = np.exp(-xi * omega * t) * (A * (-omega_d * np.sin(omega_d * t) - xi * omega * np.cos(omega_d * t)) + \
                                   B * (omega_d * np.cos(omega_d * t) - xi * omega * np.sin(omega_d * t)))

    a = np.exp(-xi * omega * t) * (A * (
                -omega_d ** 2 * np.cos(omega_d * t) + 2 * xi * omega * omega_d * np.sin(omega_d * t) + (
                    xi * omega) ** 2 * np.cos(omega_d * t)) + \
                                   B * (-omega_d ** 2 * np.sin(omega_d * t) - 2 * xi * omega * omega_d * np.cos(
                omega_d * t) + (xi * omega) ** 2 * np.sin(omega_d * t)))

    u_new = Signal(u, t)
    v_new = Signal(v, t)
    a_new = Signal(a, t)

    return a_new, v_new, u_new


def plot_damped_response(ax, vx, ux, a_th, v_th, u_th, T):
    """
    Compare experimental and theoretical damped system responses.
    Displays acceleration, velocity, displacement vs time.
    """
    if showPlot:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 6))
        axes[0].set_title('Damped system:\nComparison theorical - experimental response')

        axes[0].plot(ax.t, ax.u, label="ax_exp", color='purple', lw=1)
        axes[0].plot(a_th.t, a_th.u, label="ax_th", color='orange', lw=1, ls='--')

        axes[1].plot(vx.t, vx.u, label="vx_exp", color='green', lw=1)
        axes[1].plot(v_th.t, v_th.u, label="vx_th", color='orange', lw=1, ls='--')

        axes[2].plot(ux.t, ux.u, label="ux_exp", color='blue', lw=1)
        axes[2].plot(u_th.t, u_th.u, label="ux_th", color='orange', lw=1, ls='--')

        axes[-1].set_xlabel('Time [s]')
        for axe in axes:
            axe.set_xlim(left=0)
            axe.legend(loc='upper right')
            axe.grid()

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)  # space between plot
        if saveFig:
            plt.savefig("Figures/Q1.6_Damped_system_response.png")
        plt.show()


# ========== PARAMETERS ==========
showPlot = False
saveFig = False
cleaning_t0 = 8.7  # Start time to remove the first unstable oscillations
cleaning_tf = 120  # End time to remove the last oscillations to get a better mean valueµ

dt = 1 / 1024  # [s]
fs = 1024  # [Hz]

# --------- Q1.1 Hypothesis ---------
rho = 7900  # [kg/m³]
m = 1.2071358000000003  # [kg]
L = 0.2511074280441355  # [m]
I = 7.15822e-12  # [m^4]

k = 284.81765851785224  # [N/m]
omega_th = 15.360501352872701  # [rad/s]
f_th = 2.4446997186794364  # [Hz]
T_th = 0.409048192037333  # [s]

params_Part1 = {"k": k,
                "omega": omega_th,
                "f": f_th,
                "T": T_th,
                "xi": None,
                "c": None}

if __name__ == "__main__":
    showPlot = True

print("============================")
print("========== PART 1 ==========")
print("============================")

# ========== RETREIVE DATA ==========

data = read_data("Free vibration_4 blocks.csv")
columns = data.columns
data['time'] = data['ticks'] * dt
data['ax'] = data[columns[1]]
data['ay'] = data[columns[2]]
data['az'] = data[columns[3]]

# plotter(data)


# ========== Q1.1 Dynamical properties ==========
print("\n=== Q1.1 ===")

print("Hypothesis:")
print(f"Density: rho = {rho} kg/m³ ; Mass: m = {m} [kg] ; Length: L = {L} [m] ; Inertia: I = {I} [m^4]")

print(f"Stiffness: k = {k} [N/m]")
print(f"Theoretical natural frequency: f_th = {f_th} [Hz] ; T_th = {T_th} [s] ; omega_th = {omega_th} [rad/s]")


# ========== Q1.2 Compute displacement, ux and u0 ==========
print("\n=== Q1.2 ===")

# ----- 1. Choose which signal we will work with (with/without filter) + Calibrate the trimming limits -----
ax_filter, vx_filter, ux_filter = get_ux(data, optionFilter=True)  # retrieve filter signals
ax_nonFilter, vx_nonFilter, ux_nonFilter = get_ux(data, optionFilter=False)
plot_comparison_with_without_filter(ax_filter, vx_filter, ux_filter, ax_nonFilter, vx_nonFilter, ux_nonFilter)

ax_chosen, vx_chosen, ux_chosen = ax_filter, vx_filter, ux_filter  # Choose which signal we keep
print(f"Trimming time for the signal ax: from {cleaning_t0} [s] to {cleaning_tf} [s]")

title = "Chosen signals"
plot_signals(ax_chosen, vx_chosen, ux_chosen, title)

# ----- 2. Determine u0 -----
u0, u_trim, t0_trim, tf_trim = get_u0(ux_chosen)
plot_u0(u0, ux_chosen, u_trim)

# ----- 3. Update trimming limits & compute final trimmed signals
cleaning_t0 = t0_trim  # Update cleaning _t0
cleaning_tf = tf_trim  # Update cleaning _tf

[ax, vx, ux] = clean_signals([ax_chosen, vx_chosen, ux_chosen], cleaning_t0, cleaning_tf, 0.001)
title = "Free vibration response"
plot_signals(ax, vx, ux, title)

print(f"Estimated u0: {u0} [m]")
print(f"Final trimming time for the signals: from {cleaning_t0} [s] to {cleaning_tf} [s]")


# ========== Q1.4 Free undamped system responce ==========
print("\n=== Q1.4 ===")

# ----- 1. Get natural frequency from experimental signal -----
Tn, T_values, peaks, troughs = get_natural_frequency(ax)
fn = 1.0 / Tn
omega_n = 2 * np.pi / Tn
print(f"Estimated natural frequency: {fn} [Hz] ; Tn = {Tn} [s] ; omega_n = {omega_n} [rad/s]")
plot_natural_freq(ax, T_values, peaks, troughs)

# ----- 2. Compute experimental signals on a dt_val (20s) time window -----
dt_val = 20  # [s]
ax_t20, vx_t20, ux_t20 = compute_experimental_signals_on_dtVal(ax, vx, ux, dt_val, showPlot=False)
# Plotting comparison theory (with omega_th) and experiment
plot_free_undamped_syst_comparison(ax_t20, vx_t20, ux_t20, omega_th, dt_val, u0, saveFig)
# Plotting comparison theory (with omega_n) and experiment
plot_free_undamped_syst_comparison(ax_t20, vx_t20, ux_t20, omega_n, dt_val, u0, saveFig)

# Choose which frequency to keep for the rest of the exercice
a_undamped_th, v_undamped_th, u_undamped_th = free_undamped_response(omega_n, dt_val, u0)


# ========== Q1.5 Damping coefficient ==========
print("\n=== Q1.5 ===")

xi_mean, xi_peaks, xi_troughs = get_damping_coeff(ax, peaks, troughs)
plot_hist_xi(xi_peaks, xi_troughs)
xi = xi_mean
c_c = 2 * m * omega_n  # critical damping [kg*m/s]
c = xi * c_c  # damping coeff [kg*m/s]
print(f"Estimated damping ratio: xi = {xi}")
print(f"Critical damping coefficient (with omega_n), c_c: {c_c}")
print(f"Damping coefficient (with omega_n), c: {c}")


# ========== Q1.6 Undamped-syst responce ==========
print("\n=== Q1.6 ===")

a_damped_th, v_damped_th, u_damped_th = damped_response(dt_val, xi, omega_n, u0, v0=0.0)
plot_damped_response(ax_t20, vx_t20, ux_t20, a_damped_th, v_damped_th, u_damped_th, Tn)


# ========== SAVE PARAMETERS PART 1 ==========
params_Part1["omega"] = omega_n
params_Part1["T"] = Tn
params_Part1["f"] = fn
params_Part1["xi"] = xi
params_Part1["c"] = c

print("\nEND, Parameters saved for Part 1:")


