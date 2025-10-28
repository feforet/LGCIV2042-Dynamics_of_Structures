#Using the accelerometer record, extract the amplitude of the steady-state acceleration response of the structure, plot it as a function of the ratio k and explain the curve obtained.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#####################
####  Constants  ####
#####################

dt = 1 / 1024

# TODO find start and end times for each k
k_1 = {"k": 0.5, "start": 1, "end": 9}
k_2 = {"k": 0.75, "start": 11, "end": 19}
k_3 = {"k": 1.0, "start": 21, "end": 29}
k_4 = {"k": 1.5, "start": 31, "end": 39}
k_5 = {"k": 2.0, "start": 41, "end": 49}

ks = [k_1, k_2, k_3, k_4, k_5]


#####################
#### Definitions ####
#####################

def read_data(filename):
    data = pd.read_csv(filename, sep=';', header=0, dtype=float)
    data.rename(columns={"ax_20300039 (Fs:1024Hz Unit:m/s^2)": "x", "ay_20300039 (Fs:1024Hz Unit:m/s^2)": "y", "az_20300039 (Fs:1024Hz Unit:m/s^2)": "z"}, inplace=True)
    data['time'] = data['ticks'] * dt
    return data

def plot_all_ks(data):
    plt.figure()
    plt.plot(data["time"], data["x"], "-", label="acceleration")
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s^2]')
    plt.grid()
    plt.legend()
    plt.show()

def split_data(data, start, end):
    return data[(data['time'] >= start) & (data['time'] <= end)]


###################
#### Main Code ####
###################

data = read_data("4 masses V2.csv")

plot_all_ks(data)

data_k1 = split_data(data, k_1["start"], k_1["end"])
data_k2 = split_data(data, k_2["start"], k_2["end"])
data_k3 = split_data(data, k_3["start"], k_3["end"])
data_k4 = split_data(data, k_4["start"], k_4["end"])
data_k5 = split_data(data, k_5["start"], k_5["end"])

