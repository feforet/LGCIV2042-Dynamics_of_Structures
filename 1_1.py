import numpy as np

rho = 7900  #kg/m^3
m_blocks = rho * 60e-3 * 50e-3 * 40e-3  # kg
m_vertical = rho * 70e-3 * 300e-3 * 1.42e-3  # kg
m_horizontal = 0*rho * 70e-3 * 30e-3 * 1.42e-3  # kg
print("m_blocks =", m_blocks, "kg")
print("m_vertical =", m_vertical, "kg")
print("m_horizontal =", m_horizontal, "kg")

E = 210000e6  # Pa
I = (30e-3 * 1.42e-3**3) / 12  # m^4

h_blocks = 275e-3  # m
h_vertical = 150e-3  # m
h_horizontal = 300.71e-3  # m

m = m_blocks + m_vertical + m_horizontal
L = (m_blocks * h_blocks + m_vertical * h_vertical + m_horizontal * h_horizontal) / m
print("m =", m, "kg")
print("L =", L, "m")
print("I =", I, "m^4")

k = 3 * E * I / (L**3)
print("k =", k, "N/m")

omega_n = np.sqrt(k / m)
print("omega_n =", omega_n, "rad/s")

f_n = omega_n / (2 * np.pi)
print("f_n =", f_n, "Hz")

T = 1 / f_n
print("T_n =", T, "s")

