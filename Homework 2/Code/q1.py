"""
Question 1, Homework 2
EE746-Neuromorphic Computing, IITB 

Author: Shashwat Shukla
Date: 6th September 2018
"""
# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Define runtime variables
N  = 1  # number of presynaptic neurons
T  = 500 * 1e-3  # total time to simulate (sec)
dt = 0.1 * 1e-3  # simulation time-step (sec)
r  = 10 # mean firing rate, lambda (spikes/sec)
M  = np.int(T / dt)  # number of time-steps

I_0   = 1 * 1e-12 # current scaling factor
tau   = 15 * 1e-3 # Time constant 1 for EPSP (sec)
tau_s = 0.25 * tau # Time constant 2 for EPSP (sec)
mu    = 50 # mean of gaussian, w_0
sigma = 5 # standard deviation of gaussian, sigma_w

# Generate N poisson spikes trains for M seconds each
def generate_poisson(N, r, dt, M):
	poisson = np.random.rand(N, M)
	poisson = 1.0 * (poisson < r * dt)
	return poisson


# Computes the synaptic current due to one spike
def synapticTrace(t):
    return I_0 * ((np.exp(-t / tau)) - (np.exp(-t / tau_s)))


# Computes postsynaptic currents for all presynaptic neurons given their spiking patterns
def generate_current(N, x, epsp):
	current = []
	for i in range(N):
		conv = np.convolve(x[i], epsp, 'full')[:M] # truncated convolution 
		current.append(conv)
	current = np.array(current)
	return current


# Compute and store EPSP trace
n = int(tau * 5 / dt) # the trace will die out in 5 of the slower time constants
epsp = np.arange(0, n) * dt
epsp = synapticTrace(epsp)

# Compute spiketrains and currents
x       = generate_poisson(N, r, dt, M) # generate the spiking patterns of presynaptic neurons
current = generate_current(N, x, epsp) # compute postsynaptic currents
w       = 500 # syaptic weight
I_post  = np.transpose(np.multiply(np.transpose(current), w)) # compute weighted postsynaptic currents

# Neuron model parameters
#                C       gL       EL     VT    delT    a    tauW     b        Vr
param_list = [[200e-12, 10e-9, -70e-3, -50e-3, 2e-3, 2e-9,  30e-3,   0e-12, -58e-3],  # RS
              [130e-12, 18e-9, -58e-3, -50e-3, 2e-3, 4e-9, 150e-3, 120e-12, -50e-3],  # IB
              [200e-12, 10e-9, -58e-3, -50e-3, 2e-3, 2e-9, 200e-3, 100e-12, -46e-3]]  # CH
param_list = np.array(param_list)

V  = np.zeros((M,1))  # Array for membrane potentials
U  = np.zeros((M,1))  # Array for recovery variable
I  = np.sum(I_post, axis = 0) # compute total postsynaptic current

# Set parameters for the neurons
# RS = 0, IB = 1, CH = 2
C    = np.repeat(param_list[0][0], 1)
gL   = np.repeat(param_list[0][1], 1)
EL   = np.repeat(param_list[0][2], 1)
VT   = np.repeat(param_list[0][3], 1)
delT = np.repeat(param_list[0][4], 1) 
a    = np.repeat(param_list[0][5], 1)
tauW = np.repeat(param_list[0][6], 1)
b    = np.repeat(param_list[0][7], 1)
Vr   = np.repeat(param_list[0][8], 1)

V[0] = Vr # Initialise voltages, U is initialised to zero above

# AEF neuron equation for V
def f(V, U, t):
   return np.divide(np.multiply(-gL, V - EL) + np.multiply(np.multiply(gL, delT), np.exp(np.divide(V - VT, delT))) - U + I[t], C)


# AEF neuron equation for U
def g(V, U, t):
    return np.multiply(a, V - EL) - U


# Simulate the dynamics of the AEF neuron for M timesteps
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


# Plot current waveform of an EPSP due to one spike
plt.plot(1e3 * np.arange(0, n) * dt, epsp)
plt.title('EPSP', fontweight='bold')
plt.ylabel('Current (A)', fontweight='bold')
plt.xlabel('Time (msec)', fontweight='bold')
plt.show()

# Plot spiking of the presynaptic neuron
plt.plot(1e3 * dt * np.arange(M), x[0])
plt.title('Spiking of presynaptic neuron', fontweight='bold')
plt.ylabel('Spike pattern', fontweight='bold')
plt.xlabel('Time (msec)', fontweight='bold')
plt.show()

# Plot membrane potentials of the AEF neuron
plt.plot(1e3 * dt * np.arange(M), V)
plt.title('AEF RS Neuron', fontweight='bold')
plt.ylabel('Membrane Potential (V)', fontweight='bold')
plt.xlabel('Time (msec)', fontweight='bold')
plt.show()
