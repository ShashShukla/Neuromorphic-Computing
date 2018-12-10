"""
Izhikevich Neuron

Author: Shashwat Shukla
Date: 15th August 2018
"""
# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Runtime parameters
N = 3  # number of neurons
T = 500 * 1e-3  # total time to simulate (sec)
dt = 0.1 * 1e-3  # simulation time-step (sec)
M = np.int(T / dt)  # number of time-steps

# Neuron model parameters
#                C       kz       Er      Et       a      b      c       d      Vpeak
param_list = [[100e-12, 0.7e-6, -60e-3, -40e-3, 0.03e3, -2e-9, -50e-3, 100e-12, 35e-3],  # RS
              [150e-12, 1.2e-6, -75e-3, -45e-3, 0.01e3,  5e-9, -56e-3, 130e-12, 50e-3],  # IB
              [50e-12,  1.5e-6, -60e-3, -40e-3, 0.03e3,  1e-9, -40e-3, 150e-12, 25e-3]]  # CH
param_list = np.array(param_list)

V  = np.zeros((M, N))  # Array for membrane potentials
U  = np.zeros((M, N))  # Array for recovery variable
I  = np.zeros((M, N))  # Array for external injected currents

# Set parameters for the neurons
# Setting all neurons to be RS, change as needed
# RS = 0, IB = 1, CH = 2
C  = np.repeat(param_list[0][0], N)
kz = np.repeat(param_list[0][1], N)
Er = np.repeat(param_list[0][2], N)
Et = np.repeat(param_list[0][3], N)
a  = np.repeat(param_list[0][4], N) 
b  = np.repeat(param_list[0][5], N)
c  = np.repeat(param_list[0][6], N)
d  = np.repeat(param_list[0][7], N)
Vpeak  = np.repeat(param_list[0][8], N)

V[0] = Er # Initialise voltages, U is initialised to zero above
# Specify external currents
for i in range(N):
    I[:, i] = 400e-12 + i * 100e-12 # 400pA, 500pA, 600pA

def f(V, U, t):
   return np.divide(np.multiply(kz, np.multiply((V - Er), (V - Et))) - U + I[t], C)

def g(V, U, t):
    return np.multiply(a, np.multiply(b, V - Er) - U)

# Simulate the dynamics of the N neurons for M timesteps
for t in range(M - 1):
    # 4th order Runge Kutta
    k1 = f(V[t], U[t], t)
    l1 = g(V[t], U[t], t)
    k2 = f(V[t] + 0.5 * dt * k1, U[t] + 0.5 * dt * l1, t)
    l2 = g(V[t] + 0.5 * dt * k1, U[t] + 0.5 * dt * l1, t)
    k3 = f(V[t] + 0.5 * dt * k2, U[t] + 0.5 * dt * l2, t)
    l3 = g(V[t] + 0.5 * dt * k2, U[t] + 0.5 * dt * l2, t)
    k4 = f(V[t] + dt * k3, U[t] + dt * l3, t)
    l4 = g(V[t] + dt * k3, U[t] + dt * l3, t)
    dV = dt * (1.0 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    dU = dt * (1.0 / 6) * (l1 + 2 * l2 + 2 * l3 + l4)
    V[t + 1] = V[t] + dV # compute voltage for next timestep
    U[t + 1] = U[t] + dU # compute recovery variable for next timestep
    V[t + 1][(V[t + 1] - Vpeak) >= 0] = Vpeak[(V[t + 1] - Vpeak) >= 0] # neurons spike
    V[t + 1][(V[t] - Vpeak) == 0] = c[(V[t] - Vpeak) == 0] # voltage after spike
    U[t + 1][(V[t] - Vpeak) == 0] = U[t][(V[t] - Vpeak) == 0] + d[(V[t] - Vpeak) == 0] # recovery variable after spike

# Plot membrane potentials of the neurons
plt.plot(1e3 * dt * np.arange(M), V[:, 0])
plt.title('Neuron 1', fontweight='bold')
plt.ylabel('Membrane Potential (V)', fontweight='bold')
plt.xlabel('Time (msec)', fontweight='bold')
plt.show()

plt.plot(1e3 * dt * np.arange(M), V[:, 1])
plt.title('Neuron 2', fontweight='bold')
plt.ylabel('Membrane Potential (V)', fontweight='bold')
plt.xlabel('Time (msec)', fontweight='bold')
plt.show()

plt.plot(1e3 * dt * np.arange(M), V[:, 2])
plt.title('Neuron 3', fontweight='bold')
plt.ylabel('Membrane Potential (V)', fontweight='bold')
plt.xlabel('Time (msec)', fontweight='bold')
plt.show()
