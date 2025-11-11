# For the ground motions described in Part 2 (¨ug (t) = 0.1g sin(ωnt)), program the Central Difference Method
# to compute the response U (t) and ¨U (t) of the structure during 40s. Use different time step: ∆t = Tn/50; Tn/25;
# Tn/5; Tn/2. Compare with the experimental data and your answer from (2.2). Discuss the influence of the
# choice of the time step.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import part_1_1 as p11
import Part1_Free_vibration as p1_123456
import part_2_1 as p2_1
import Part_22_23 as p2_23

g = 9.81
end_time = 40.0

c = p1_123456.params_Part1['c']

divisors = [50, 25, 5, 2]
dts = {str(div): p11.T_n()/div for div in divisors}

u_dot_dot_g = lambda t: 0.1 * g * np.sin(p11.omega_n() * t)
u_dot_g = lambda t: -0.1 * g / p11.omega_n() * (np.cos(p11.omega_n() * t))
u_g = lambda t: -0.1 * g / (p11.omega_n()**2) * (np.sin(p11.omega_n() * t))
discretize = lambda dt: np.arange(0, end_time+dt, dt)

timestamps = {key: discretize(dt) for key, dt in dts.items()}

u_gs = {index: u_g(ts) for index, ts in timestamps.items()}
u_dot_gs = {index: u_dot_g(ts) for index, ts in timestamps.items()}
u_dot_dot_gs = {index: u_dot_dot_g(ts) for index, ts in timestamps.items()}

u_analytic, a_analytic = p2_23.analytic_steady_state(1, n=int(end_time*100), t_max=end_time)
experimental_response = p2_23.processing_data_exp([p2_1.data_ks[2]], [p2_1.ks[2]])

def plot_u_and_u_dot_dot(timestamps, u, u_dot_dot):
    plt.plot(u_analytic.t, u_analytic.u, label='Analytical steady-state')
    plt.plot(timestamps['50'], u['50'], label=f"dt = T_n/50")
    plt.plot(timestamps['25'], u['25'], label=f"dt = T_n/25")
    plt.plot(timestamps['5'], u['5'], label=f"dt = T_n/5")
    plt.ylim(-1.5, 1.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Displacement response [m]')
    plt.grid()
    plt.legend()
    plt.title("Displacement response U(t)")
    plt.savefig("Figures/Q3_Displacement_responses.png")
    plt.show()
    
    plt.plot(timestamps['2'], u['2'], label=f"dt = T_n/2")
    plt.yscale('log')
    plt.xlabel('Time [s]')
    plt.ylabel('Displacement response [m]')
    plt.grid()
    plt.legend()
    plt.title("Displacement response U(t) for dt = T_n/2")
    plt.savefig("Figures/Q3_Displacement_response_dt2.png")
    plt.show()

    plt.plot(a_analytic.t, a_analytic.u, label='Analytical steady-state')
    plt.plot(timestamps['50'], u_dot_dot['50'], label=f"dt = T_n/50")
    plt.plot(timestamps['25'], u_dot_dot['25'], label=f"dt = T_n/25")
    plt.plot(experimental_response[0][0].t[:40*1024], experimental_response[0][0].u[:40*1024], label='Experimental steady-state')
    plt.plot(timestamps['5'], u_dot_dot['5'], label=f"dt = T_n/5")
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration response [m/s^2]')
    plt.grid()
    plt.legend()
    plt.title("Acceleration response U''(t)")
    plt.savefig("Figures/Q3_Acceleration_responses.png")
    plt.show()
    
    plt.plot(timestamps['2'], u_dot_dot['2'], label=f"dt = T_n/2")
    plt.yscale('log')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration response [m/s^2]')
    plt.grid()
    plt.legend()
    plt.title("Acceleration response U''(t) for dt = T_n/2")
    plt.savefig("Figures/Q3_Acceleration_response_dt2.png")
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

plot_u_and_u_dot_dot(timestamps, u_s, udotdot_s)

print(dts.keys())