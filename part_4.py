import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Part1_Free_vibration import damped_response
import Part1_Free_vibration as p1_23456
import part_1_1 as p1_1
import part_2_1 as p2_1
import Part_22_23 as p2_23


omega_n = p1_23456.omega_n
c = p1_23456.params_Part1["c"]
u0 = p1_23456.params_Part1["omega"]
xi = p1_23456.params_Part1["xi"]
m = p1_1.m()


def compute_fft(signal):
    return np.fft.fft(signal)


def part_4_1():
    dt_val = 40     #[s], time of the computed theorical acceleration
    # Recorded acceleration signal from 1.2
    response_1_2 = p1_23456.compute_experimental_signals_on_dtVal(p1_23456.ax, p1_23456.vx, p1_23456.ux, dt_val, showPlot=False)[0].u
    # Analytical free undamped response from 1.3
    response_1_3 = p1_23456.free_undamped_response(omega_n, dt_val, u0, v0=0.0)[0].u


    fft_1_2 = compute_fft(response_1_2)
    fft_1_3 = compute_fft(response_1_3)

    plt.plot(np.abs(fft_1_2), label='FFT of Response 1.2')
    plt.plot(np.abs(fft_1_3), label='FFT of Response 1.3')
    plt.title('FFT of Responses')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.show()

part_4_1()

def part_4_2():
    dt_val = 60
    response_1_6 = damped_response(dt_val, xi, omega_n, u0)[0].u
    response_2pcent_damping = damped_response(dt_val, int(0.02/(2*m*omega_n)), omega_n, u0)[0].u
    response_5pcent_damping = damped_response(dt_val, int(0.05/(2*m*omega_n)), omega_n, u0)[0].u
    response_10pcent_damping = damped_response(dt_val, int(0.1/(2*m*omega_n)), omega_n, u0)[0].u

    fft_1_6 = compute_fft(response_1_6)
    fft_2pcent = compute_fft(response_2pcent_damping)
    fft_5pcent = compute_fft(response_5pcent_damping)
    fft_10pcent = compute_fft(response_10pcent_damping)
    
    plt.plot(np.abs(fft_1_6), label='FFT of Response 1.6')
    plt.plot(np.abs(fft_2pcent), label='FFT of Response 2% Damping')
    plt.plot(np.abs(fft_5pcent), label='FFT of Response 5% Damping')
    plt.plot(np.abs(fft_10pcent), label='FFT of Response 10% Damping')
    plt.title('FFT of Responses with Different Damping')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.show()

part_4_2()

def part_4_34(shorten=False):
    responses = p2_1.data_ks
    response_k_05 = responses[0]
    response_k_075 = responses[1]
    response_k_1 = responses[2]
    response_k_15 = responses[3]
    response_k_2 = responses[4]
    print(f"args : \t {0.5} \t {omega_n} \t {len(response_k_05)}")
    analitical_response_k_05 = p2_23.analytic_steady_state(0.5, len(response_k_05), len(response_k_05)/1024)[1].u
    analitical_response_k_075 = p2_23.analytic_steady_state(0.75, len(response_k_075), len(response_k_075)/1024)[1].u
    analitical_response_k_1 = p2_23.analytic_steady_state(1.0, len(response_k_1), len(response_k_1)/1024)[1].u
    analitical_response_k_15 = p2_23.analytic_steady_state(1.5, len(response_k_15), len(response_k_15)/1024)[1].u
    analitical_response_k_2 = p2_23.analytic_steady_state(2.0, len(response_k_2), len(response_k_2)/1024)[1].u

    if shorten:
        response_k_05 = response_k_05[:1024*20]
        response_k_075 = response_k_075[:1024*20]
        response_k_1 = response_k_1[:1024*20]
        response_k_15 = response_k_15[:1024*20]
        response_k_2 = response_k_2[:1024*20]
        analitical_response_k_05 = analitical_response_k_05[:1024*20]
        analitical_response_k_075 = analitical_response_k_075[:1024*20]
        analitical_response_k_1 = analitical_response_k_1[:1024*20]
        analitical_response_k_15 = analitical_response_k_15[:1024*20]
        analitical_response_k_2 = analitical_response_k_2[:1024*20]

    fft_k_05 = compute_fft(response_k_05)
    fft_k_075 = compute_fft(response_k_075)
    fft_k_1 = compute_fft(response_k_1)
    fft_k_15 = compute_fft(response_k_15)
    fft_k_2 = compute_fft(response_k_2)
    fft_anal_k_05 = compute_fft(analitical_response_k_05)
    fft_anal_k_075 = compute_fft(analitical_response_k_075)
    fft_anal_k_1 = compute_fft(analitical_response_k_1)
    fft_anal_k_15 = compute_fft(analitical_response_k_15)
    fft_anal_k_2 = compute_fft(analitical_response_k_2)

    plt.plot(np.abs(fft_k_05), label='FFT of k=0.5')
    plt.plot(np.abs(fft_k_075), label='FFT of k=0.75')
    plt.plot(np.abs(fft_k_1), label='FFT of k=1.0')
    plt.plot(np.abs(fft_k_15), label='FFT of k=1.5')
    plt.plot(np.abs(fft_k_2), label='FFT of k=2.0')
    plt.plot(np.abs(fft_anal_k_05), label='FFT of Anal k=0.5', linestyle='dashed')
    plt.plot(np.abs(fft_anal_k_075), label='FFT of Anal k=0.75', linestyle='dashed')
    plt.plot(np.abs(fft_anal_k_1), label='FFT of Anal k=1.0', linestyle='dashed')
    plt.plot(np.abs(fft_anal_k_15), label='FFT of Anal k=1.5', linestyle='dashed')
    plt.plot(np.abs(fft_anal_k_2), label='FFT of Anal k=2.0', linestyle='dashed')
    plt.title('FFT of Responses for Different Stiffness')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.show()

part_4_34()
part_4_34(shorten=True)