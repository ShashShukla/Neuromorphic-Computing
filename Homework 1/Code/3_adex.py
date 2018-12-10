"""
Adaptive Exponential Integrate and Fire

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
#                C       gL       EL     VT    delT    a    tauW     b        Vr
param_list = [[200e-12, 10e-9, -70e-3, -50e-3, 2e-3, 2e-9,  30e-3,   0e-12, -58e-3],  # RS
              [130e-12, 18e-9, -58e-3, -50e-3, 2e-3, 4e-9, 150e-3, 120e-12, -50e-3],  # IB
              [200e-12, 10e-9, -58e-3, -50e-3, 2e-3, 2e-9, 200e-3, 100e-12, -46e-3]]  # CH
param_list = np.array(param_list)

V  = np.zeros((M, N))  # Array for membrane potentials
U  = np.zeros((M, N))  # Array for recovery variable
I  = np.zeros((M, N))  # Array for external injected currents

# Set parameters for the neurons
# Setting all neurons to be RS, change as needed
# RS = 0, IB = 1, CH = 2
C    = np.repeat(param_list[0][0], N)
gL   = np.repeat(param_list[0][1], N)
EL   = np.repeat(param_list[0][2], N)
VT   = np.repeat(param_list[0][3], N)
delT = np.repeat(param_list[0][4], N) 
a    = np.repeat(param_list[0][5], N)
tauW = np.repeat(param_list[0][6], N)
b    = np.repeat(param_list[0][7], N)
Vr   = np.repeat(param_list[0][8], N)

V[0] = Vr # Initialise voltages, U is initialised to zero above
# Specify external currents
for i in range(N):
    I[:, i] = 250e-12 + i * 100e-12 # 250pA, 350pA, 450pA

def f(V, U, t):
   return np.divide(np.multiply(-gL, V - EL) + np.multiply(np.multiply(gL, delT), np.exp(np.divide(V - VT, delT))) - U + I[t], C)

def g(V, U, t):
    return np.multiply(a, V - EL) - U

# Simulate the dynamics of the N neurons for M timesteps
for t in range(M - 1):
  # Forward Euler
  k1 = f(V[t], U[t], t)
  l1 = g(V[t], U[t], t)
  dV = dt * k1
  dU = dt * l1
  V[t + 1] = V[t] + dV # compute voltage for next timestep
  U[t + 1] = U[t] + dU # compute recovery variable for next timestep
  V[t + 1][V[t + 1] >= 0] = 0 # neurons spike
  V[t + 1][V[t] == 0] = Vr[V[t] == 0] # voltage after spike
  U[t + 1][V[t] == 0] = U[t][V[t] == 0] + b[V[t] == 0] # recovery variable after spike

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