# For the ground motions described in Part 2 (¨ug (t) = 0.1g sin(ωnt)), program the Central Difference Method
# to compute the response U (t) and ¨U (t) of the structure during 40s. Use different time step: ∆t = Tn/50; Tn/25;
# Tn/5; Tn/2. Compare with the experimental data and your answer from (2.2). Discuss the influence of the
# choice of the time step.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import part_1_1 as p11

g = 9.81
end_time = 40.0

c = 0.001256421782425465

dts_tmp = [p11.T_n()/50, p11.T_n()/25, p11.T_n()/5, p11.T_n()/2]
dts = {str(dt): dt for dt in dts_tmp}

u_dot_dot_g = lambda t: 0.1 * g * np.sin(p11.omega_n() * t)
u_dot_g = lambda t: -0.1 * g / p11.omega_n() * (np.cos(p11.omega_n() * t))
u_g = lambda t: -0.1 * g / p11.omega_n()**2 * (np.sin(p11.omega_n() * t))
discretize = lambda dt: np.arange(0, end_time+dt, dt)

timestamps = {key: discretize(dt) for key, dt in dts.items()}

u_gs = {index: u_g(ts) for index, ts in timestamps.items()}
u_dot_gs = {index: u_dot_g(ts) for index, ts in timestamps.items()}
u_dot_dot_gs = {index: u_dot_dot_g(ts) for index, ts in timestamps.items()}

def plot_u_g_and_derivatives(timestamps, u, u_dot, u_dot_dot):
    for key in timestamps.keys():
        plt.plot(timestamps[key], u[key], label="{}: {}".format(key, dts[key]))
    plt.xlabel('Time [s]')
    plt.ylabel('Ground Displacement [m]')
    plt.grid()
    plt.legend()
    plt.title("Ground Displacement")
    plt.show()

    for key in timestamps.keys():
        plt.plot(timestamps[key], u_dot[key], label="{}: {}".format(key, dts[key]))
    plt.xlabel('Time [s]')
    plt.ylabel('Ground Velocity [m/s]')
    plt.grid()
    plt.legend()
    plt.title("Ground Velocity")
    plt.show()

    for key in timestamps.keys():
        plt.plot(timestamps[key], u_dot_dot[key], label="{}: {}".format(key, dts[key]))
    plt.xlabel('Time [s]')
    plt.ylabel('Ground Acceleration [m/s^2]')
    plt.grid()
    plt.legend()
    plt.title("Ground Acceleration")
    plt.show()


def central_difference_method(m, c, k, timestamps):
    n_steps = len(timestamps)

    u = np.zeros(n_steps)
    u_dot = np.zeros(n_steps)
    u_dot_dot = np.zeros(n_steps)

    # Initial conditions
    u[0] = 0.0
    u[-1] = u[0] - dt * u_dot[0] + (dt**2 / 2) * u_dot_dot[0]
    
    k_b = m / (dt**2) + c / (2 * dt)
    a = m / (dt**2) - c / (2 * dt)
    b = k - 2 * m / (dt**2)

    for n in range(n_steps-1):
        p_n = -m * u_dot_dot_g(timestamps[n])
        p_b = p_n - a * u[n-1] - b * u[n]

        u[n+1] = p_b / k_b
        
        u_dot[n] = (u[n+1] - u[n-1]) / (2 * dt)
        u_dot_dot[n] = (u[n+1] - 2 * u[n] + u[n-1]) / (dt**2)

    return u, u_dot, u_dot_dot

u_s, udot_s, udotdot_s = {}, {}, {}
for key, dt in dts.items():
    ts = discretize(dt)
    u_, udot_, udotdot_ = central_difference_method(p11.m(), c, p11.k(), timestamps[key])
    u_s[key] = u_
    udot_s[key] = udot_
    udotdot_s[key] = udotdot_

plot_u_g_and_derivatives(timestamps, u_s, udot_s, udotdot_s)

print(dts.keys())