import numpy as np

rho = lambda: 7900  #kg/m^3
m_blocks = lambda: rho() * 60e-3 * 50e-3 * 40e-3  # kg
m_vertical = lambda: rho() * 70e-3 * 300e-3 * 1.42e-3  # kg
m_horizontal = lambda: rho() * 70e-3 * 30e-3 * 1.42e-3  # kg

E = lambda: 210000e6  # Pa
I = lambda: (30e-3 * 1.42e-3**3) / 12  # m^4

h_blocks = lambda: 275e-3  # m
h_vertical = lambda: 150e-3  # m
h_horizontal = lambda: 300.71e-3  # m

m = lambda: m_blocks() + m_vertical() + m_horizontal()
L = lambda: (m_blocks() * h_blocks() + m_vertical() * h_vertical() + m_horizontal() * h_horizontal()) / m()

k = lambda: 3 * E() * I() / (L()**3)

omega_n = lambda: np.sqrt(k() / m())

f_n = lambda: omega_n() / (2 * np.pi)

T_n = lambda: 1 / f_n()

if __name__ == "__main__":
    print("m_blocks =", m_blocks(), "kg")
    print("m_vertical =", m_vertical(), "kg")
    print("m_horizontal =", m_horizontal(), "kg")
    print("m =", m(), "kg")
    print("L =", L(), "m")
    print("I =", I(), "m^4")
    print("k =", k(), "N/m")
    print("omega_n =", omega_n(), "rad/s")
    print("f_n =", f_n(), "Hz")
    print("T_n =", T_n(), "s")
