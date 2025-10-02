import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dt = 1 / 1024     # [s]

def read_data(filename):
    data = pd.read_csv(filename, sep=';', header=0, dtype=float)
    return data

datas = read_data("C:\\Users\\felix\\OneDrive - UCL\\Ucl\\Master\\Q9\\LGCIV 2042 - Dynamics of Structures\\LGCIV2042-Dynamics_of_Structures\\Free vibration_4 blocks.csv")

