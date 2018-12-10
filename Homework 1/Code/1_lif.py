"""
LIF Neuron

Author: Shashwat Shukla
Date: 11th August 2018
"""
# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Runtime parameters
N = 10  # number of neurons
T = 500 * 1e-3  # total time to simulate (sec)
dt = 0.1 * 1e-3  # simulation time-step (sec)
M = np.int(T / dt)  # number of time-steps

# Neuron model parameters
C = 300e-12
gamma = 1.0 / C
g_L = 30e-9
V_T = 20e-3
E_L = -70e-3
I_C = g_L * (V_T - E_L)  # threshold current

V = E_L * np.ones((M, N))  # Array for membrane potentials
I = np.ones((M, N))  # Array for external injected currents
for i in range(N):
    I[:, i] = I_C * (1 + 0.1 * (i + 1))

# Simulate the dynamics of the N neurons for M timesteps
for t in range(M - 1):
    # 2nd order Runge Kutta
    dV1 = 0.5 * dt * gamma * (-g_L * (V[t] - E_L) + I[t]) 
    dV2 = dt  * gamma * (-g_L * (V[t] + dV1 - E_L) + I[t])
    V[t + 1] = V[t] + dV2 # compute voltage for next timestep
    V[t + 1][V[t + 1] >= V_T] = V_T # neurons spike
    V[t + 1][V[t] == V_T] = E_L # reset voltage to E_L

# Compute Interspike intervals
isi_avg = np.zeros(N)
for i in range(N):
	spike_times = np.where(V[:, i] == V_T)
	isi = dt * np.diff(spike_times)
	avg = np.mean(isi)
	isi_avg[i] = avg

# Plot membrane potentials for some of the neurons
plt.plot(1e3 * dt * np.arange(M), V[:, 1])
plt.title('Neuron 2', fontweight='bold')
plt.ylabel('Membrane Potential (V)', fontweight='bold')
plt.xlabel('Time (msec)', fontweight='bold')
plt.show()

plt.plot(1e3 * dt * np.arange(M), V[:, 3])
plt.title('Neuron 4', fontweight='bold')
plt.ylabel('Membrane Potential (V)', fontweight='bold')
plt.xlabel('Time (msec)', fontweight='bold')
plt.show()

plt.plot(1e3 * dt * np.arange(M), V[:, 5])
plt.title('Neuron 6', fontweight='bold')
plt.ylabel('Membrane Potential (V)', fontweight='bold')
plt.xlabel('Time (msec)', fontweight='bold')
plt.show()

plt.plot(1e3 * dt * np.arange(M), V[:, 7])
plt.title('Neuron 8', fontweight='bold')
plt.ylabel('Membrane Potential (V)', fontweight='bold')
plt.xlabel('Time (msec)', fontweight='bold')
plt.show()

# Plot Interspike Intervals
plt.plot(I[0], 1e3 * isi_avg)
plt.title('Average ISI vs Current', fontweight='bold')
plt.ylabel('Average Interspike Interval (msec)', fontweight='bold')
plt.xlabel('Injected Current (A)', fontweight='bold')
plt.show()