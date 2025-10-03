import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dt = 1 / 1024     # [s]

def read_data(filename):
    data = pd.read_csv(filename, sep=';', header=0, dtype=float)
    return data

def plotter(data):
    for col in data.columns:
        if col.startswith('a'):
            plt.plot(data['time'], data[col], label=col[1])

    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s^2]')
    plt.grid()
    plt.legend()
    plt.show()


data = read_data("Free vibration_4 blocks.csv")
data['time'] = data['ticks'] * dt
plotter(data)
