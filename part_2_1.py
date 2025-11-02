#Using the accelerometer record, extract the amplitude of the steady-state acceleration response of the structure, plot it as a function of the ratio k and explain the curve obtained.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import part_1_1 as p11
import scipy.signal as sc

#####################
####  Constants  ####
#####################

displayPlots = True

omega_n_exp = 2.169491525423729
omega_n_theo = p11.omega_n()
dt = 1 / 1024

# TODO find start and end times for each k
k_1 = {"k": 0.5, "start": 5, "end": 101}
k_2 = {"k": 0.75, "start": 116, "end": 214}
k_3 = {"k": 1.0, "start": 224, "end": 322}
k_4 = {"k": 1.5, "start": 337, "end": 440}
k_5 = {"k": 2.0, "start": 453, "end": 550}

ks = [k_1, k_2, k_3, k_4, k_5]

for k in ks:
    k["omega_bar"] = k["k"] * omega_n_exp  # TODO Il faut utiliser omega_n expérimental et pas le théorique

#####################
#### Definitions ####
#####################

def read_data(filename):
    data = pd.read_csv(filename, sep=';', header=0, dtype=float)
    data.rename(columns={"ax_20300039 (Fs:1024Hz Unit:m/s^2)": "x", "ay_20300039 (Fs:1024Hz Unit:m/s^2)": "y", "az_20300039 (Fs:1024Hz Unit:m/s^2)": "z"}, inplace=True)
    data['time'] = data['ticks'] * dt
    return data

def plot_raw_data(data):
    plt.figure()
    plt.plot(data["time"], data["x"], "-", label="acceleration")
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s^2]')
    plt.grid()
    plt.legend()
    plt.show()

def plot_acceleration(data, k_i, peaks):
    plt.figure()
    plt.plot(data["time"], data["x"], "-", label="Acceleration")
    plt.plot(data["time"].iloc[peaks[0]], peaks[1], "x", label="Peaks")
    plt.title("Acceleration response for k = {:.2f}".format(k_i["k"]))
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s^2]')
    plt.grid()
    plt.legend()
    plt.show()

def split_data(data, start, end):
    return data[(data['time'] >= start) & (data['time'] <= end)]

def find_peaks(data, k_i, T_guess):
    min_distance = 0.9 * T_guess / dt
    peaks, _ = sc.find_peaks(data["x"], distance=min_distance)
    peak_values = data["x"].iloc[peaks]
    return peaks, peak_values

###################
#### Main Code ####
###################

data = read_data("4 masses V2.csv")

if displayPlots:
    plot_raw_data(data)

data_k1 = split_data(data, k_1["start"], k_1["end"])
data_k2 = split_data(data, k_2["start"], k_2["end"])
data_k3 = split_data(data, k_3["start"], k_3["end"])
data_k4 = split_data(data, k_4["start"], k_4["end"])
data_k5 = split_data(data, k_5["start"], k_5["end"])

data_ks = [data_k1, data_k2, data_k3, data_k4, data_k5]

peaks_k1 = find_peaks(data_k1, k_1, 2 * np.pi / (k_1["omega_bar"]) / 6)  # OK
peaks_k2 = find_peaks(data_k2, k_2, 2 * np.pi / (k_2["omega_bar"]) / 2)  # OK
peaks_k3 = find_peaks(data_k3, k_3, 2 * np.pi / (k_3["omega_bar"]) / 8)  # OK
peaks_k4 = find_peaks(data_k4, k_4, 2 * np.pi / (k_4["omega_bar"]) / 2)  # OK
peaks_k5 = find_peaks(data_k5, k_5, 2 * np.pi / (k_5["omega_bar"]) / 8)

peaks_ks = [peaks_k1, peaks_k2, peaks_k3, peaks_k4, peaks_k5]

if displayPlots:
    plot_acceleration(data_k1, k_1, peaks_k1)
    plot_acceleration(data_k2, k_2, peaks_k2)
    plot_acceleration(data_k3, k_3, peaks_k3)
    plot_acceleration(data_k4, k_4, peaks_k4)
    plot_acceleration(data_k5, k_5, peaks_k5)

for i, k in enumerate(ks):
    mean_peak = np.mean(peaks_ks[i][1])
    print(f"k = {k['k']}: Mean peak amplitude = {mean_peak:.4f} m/s^2")

if displayPlots:
    k_values = [k["k"] for k in ks]
    mean_peaks = [np.mean(peaks_ks[i][1]) for i in range(len(ks))]

    plt.figure()
    plt.plot(k_values, mean_peaks, 'o-')
    plt.xlabel('k')
    plt.ylabel('Mean peak amplitude [m/s^2]')
    plt.title('Mean peak amplitude vs k')
    plt.grid()
    plt.show()