import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import detrend, butter, filtfilt

class signal:
    def __init__(self, u, t):
        self.u = u
        self.t = t
        self.size = len(u)
        if (len(u) != len(t)):
            print("ERROR: u and t must have the same length")
            self.size = -1

    def __len__(self):
        return self.size

    def trim_index(self, index_0, index_f):
       return self.u[index_0:index_f], self.t[index_0:index_f]

    def trim_time(self, time_0, time_f, precision):
        index_0 = find_value(self.t, time_0, precision)
        index_f = find_value(self.t, time_f, precision)
        return self.u[index_0:index_f], self.t[index_0:index_f]
    
    def trim_signal(self, val_0, val_f, precision):
        index_0 = find_value(self.u, val_0, precision)
        index_f = find_value(self.u, val_f, precision)
        return self.u[index_0:index_f], self.t[index_0:index_f]

    def set_to_zeros(self, index_0, index_f):
        new_u = self.u
        for i in range(index_0, index_f):
            new_u[i] = 0
        return new_u
    

    """ Re-centre le signal par rapport à sa moyenne """
    def center_signal(self):
        u_old = self.u
        self.u = u_old - np.mean(u_old)

    def filter(self, freq, type='high'):
        b, a = butter(2, 0.5 / (0.5 * freq), btype=type)
        new_u = filtfilt(b, a, self.u)
        return new_u

def find_value(x, value, precision=0.0):
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
    for col in data.columns:
        if col.startswith('a'):
            plt.plot(data['time'], data[col], label=col[1])

    plt.title('Raw data')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s^2]')
    plt.grid()
    plt.legend()
    plt.savefig("Figures/Plot_acceleration_init")
    plt.show()

def data_processing(data):
    ax_init = data['ax']
    ay_init = data['ay']
    az_init = data['az']
    t_init = data['time']

    ax_vect = np.array(ax_init)
    ay_vect = np.array(ay_init)
    az_vect = np.array(az_init)
    t = np.array(t_init)

    ax = signal(ax_vect, t)
    ay = signal(ay_vect, t)
    az = signal(az_vect, t)

    return ax, ay, az

def cleen_signals(data, t0, tf):
    ax, ay, az, _ = data_processing(data)
    ax.u, ax.t = ax.trim_time(t0, tf)
    ay.u, ay.t = ay.trim_time(t0, tf)
    az.u, az.t = az.trim_time(t0, tf)

    return ax, ay, az, ax

    u.trim_time(t0, tf)

# Intégration trapézoïdale cumulée (vitesse et déplacement)
def integrate_trapezoid(x, dt, x0=0.0):
    y = np.zeros(len(x))
    for i in range(len(x)-1):
        y[i] = y[i-1] + (x[i] + x[i-1]) * dt/2
    return y

def get_u0(ux):
    ux_trim, t_trim = ux.trim_time(cleaning_t0, cleaning_tf, 0.001)
    ux_trim = ux_trim - np.mean(ux_trim)    # Centering the trimmed signal

    u0 = np.max(abs(ux_trim))

    if (showPlot):

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
        
        axes[0].set_title('Displacement ux')
        axes[0].plot(ux.t, ux.u, label="ux [m/s²]", color='purple', lw=1)
        axes[0].axhline(y=u0, color='r', linestyle='--', label='u0 max={:.2f}'.format(u0), lw=1) 
        axes[0].axhline(y=-u0, color='r', linestyle='--', lw=1)

        axes[1].set_title('Trimmed signal to estimate u0')
        axes[1].plot(t_trim, ux_trim, label="ux_trim [m/s]", color='pink', lw=1)
        axes[1].axhline(y=u0, color='r', linestyle='--', label='u0 max={:.2f}'.format(u0), lw=1) 
        axes[1].axhline(y=-u0, color='r', linestyle='--', lw=1)

        axes[1].set_xlabel('Time [s]')

        for axe in axes:
            axe.set_xlim(left=0)
            axe.legend()
            axe.grid()

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)  # space between plot
        plt.savefig("Figures/Q1.2_u0_estimation.png")
        plt.show()

    return u0

def get_ux(data):
    ax, _, _, = data_processing(data)

    # ----- 1.1 Suppression de la moyenne et tendance linéaire -----
    ax.center_signal()

    # ----- 1.2 Filtrage passe-haut pour supprimer le bruit basses fréquences -----
    ax.u = ax.filter(fs, 'high')

    # ----- 2. Integration numérique de ax -----
    velocity = integrate_trapezoid(ax.u, dt, 0.0)  # Vitesse (v(0) = 0)
    vx = signal(velocity, ax.t)

    # ----- 2.1 Traitement du signal vx -----
    vx.center_signal()
    vx.u = vx.filter(fs, 'high')

    # ----- 3. Integration numérique de vx -----
    disp = integrate_trapezoid(vx.u, dt, 0.0)  # Déplacement relatif
    ux = signal(disp, vx.t)

    # ----- 3.1 Traitement du signal ux -----
    ux.center_signal()
    #ux.u = ux.filter(fs, 'high')

    # ----- 4. Estimation de l'amplitude initiale, u0 -----
    u0 = get_u0(ux)


    # ----- Affichage des résultats -----
    if (showPlot):
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 6))
        axes[0].plot(ax.t, ax.u, label="ax [m/s²]", color='purple', lw=1)
        axes[1].plot(vx.t, vx.u, label="vx [m/s]", color='green', lw=1)
        axes[2].plot(ux.t, ux.u, label="ux [m]", color='blue', lw=1)

        axes[2].set_xlabel('Time [s]')

        for axe in axes:
            axe.set_xlim(left=0)
            axe.legend(loc='upper right')
            axe.grid()

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3) # space between plot
        plt.savefig("Figures/Q1.2_Compare_ax_vx_ux.png")
        plt.show()

    return ax, vx, ux, u0


def plot_free_undamped_syst(ax, vx, ux, u0, v0=0.0):
    dt_val = 20

    # ---- Experimental data ----
    ax_t20, t20 = ax.trim_time(cleaning_t0, cleaning_t0 + dt_val, 0.001)    # Get the real 20s of data
    ax_t20 = signal(ax_t20, t20)                                            
    ax_t20.t = ax_t20.t - ax_t20.t[0]                                       # Rebase time to 0s                   

    vx_t20, t20 = vx.trim_time(cleaning_t0, cleaning_t0 + dt_val, 0.001)
    vx_t20 = signal(vx_t20, t20)
    vx_t20.t = vx_t20.t - vx_t20.t[0]

    ux_t20, t20 = ux.trim_time(cleaning_t0, cleaning_t0 + dt_val, 0.001)
    ux_t20 = signal(ux_t20, t20)
    ux_t20.t = ux_t20.t - ux_t20.t[0]


    # ---- Theoretical response of free undamped system ----
    t = np.linspace(0, dt_val, len(ux_t20))
    u = u0 * np.cos(omega_n * t) + (v0 / omega_n) * np.sin(omega_n * t)
    v = - u0 * omega_n * np.sin(omega_n * t) + v0 * np.cos(omega_n * t)
    a = - omega_n**2 * u


    # ---- Plotting ----

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 6))
    axes[0].set_title('Response of free undamped system')

    axes[0].plot(ax_t20.t, ax_t20.u, label="ax_exp [m/s²]", color='purple', lw=1)
    axes[0].plot(t, a, label="ax_th [m/s²]", color='orange', lw=1, ls='--')

    axes[1].plot(vx_t20.t, vx_t20.u, label="vx_exp [m/s]", color='green', lw=1)
    axes[1].plot(t, v, label="vx_th [m/s]", color='orange', lw=1, ls='--')

    axes[2].plot(ux_t20.t, ux_t20.u, label="ux [m]", color='blue', lw=1)
    axes[2].plot(t, u, label="ux_th [m]", color='orange', lw=1, ls='--')

    axes[2].set_xlabel('Time [s]')

    for axe in axes:
        axe.set_xlim(left=0)
        axe.legend(loc='upper right')
        axe.grid()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)  # space between plot
    plt.savefig("Figures/Q1.4_Free_undamped_response_comparison.png")
    plt.show()

    return a, v, u, t



def get_damping_coeff(ax, ux):
    return


# ========== PARAMETERS ==========
showPlot = True
cleaning_t0 = 10.5                      # Start time to remove the first unstable oscillations
cleaning_tf = 115                        # End time to remove the last oscillations to get a better mean valueµ

dt = 1 / 1024     # [s]
fs = 1024         # [Hz]

# --------- Q1.1 Hypothesis ---------
rho = 7900 # [kg/m³]
m = 1.2071358000000003 #[kg]
L = 0.2511074280441355 #[m]
I = 7.15822e-12 #[m^4]


k = 284.81765851785224 #[N/m]
omega_n = 15.360501352872701 #[rad/s]
f_n = 2.4446997186794364 #[Hz]
T = 0.409048192037333 #[s]



# ========== RETREIVE DATA ==========

print("Begin\n")


data = read_data("Free vibration_4 blocks.csv")
columns = data.columns
data['time'] = data['ticks'] * dt
data['ax'] = data[columns[1]]
data['ay'] = data[columns[2]]
data['az'] = data[columns[3]]

#plotter(data)


# --------- Q1.2 Compute displacement, ux and u0 ---------
ax, vx, ux, u0  = get_ux(data)
print(f"Estimated u0: {u0} m")
print(f"Trimming time for the signal: from {cleaning_t0} s to {cleaning_tf} s")


# --------- Q1.4 Theorical response of free undamped system ---------
plot_free_undamped_syst(ax, vx, ux, u0)

print("\nEnd")


