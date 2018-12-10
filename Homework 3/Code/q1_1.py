"""
Question 1, Homework 3
EE746-Neuromorphic Computing, IITB 

Author: Shashwat Shukla
Date: 28th September 2018
"""
# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Define runtime variables
N  = 5 # number of neurons in the network
T  = 100 * 1e-3 # total time to simulate (sec)
dt = 0.1 * 1e-3 # simulation time-step (sec)
r  = 100 # mean firing rate, lambda (spikes/sec)
M  = np.int(T / dt) # number of time-steps

I_0   = 1 * 1e-12 # current scaling factor
tau   = 15 * 1e-3 # Time constant 1 for EPSP (sec)
tau_s = 0.25 * tau # Time constant 2 for EPSP (sec)

# Generate N poisson spikes trains for M seconds
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
    conv = np.convolve(x[i], epsp, 'full')[:M] # truncate convolution 
    current.append(conv)
  current = np.array(current)
  return current


# Compute and store EPSP trace
n = int(tau * 5 / dt) # the trace will die out in 5 of the slower time constants
epsp = np.arange(0, n) * dt
epsp = synapticTrace(epsp)

# Setup synpatic connections
w = [] # array to store synaptic connectivity, weights, delays
w.append([]) # Neuron a has no outgoing synapses
w.append([[0, 3000, int(1e-3 / dt)], [4, 3000, int(8e-3 / dt)]]) # Neuron b links to a and e
w.append([[0, 3000, int(5e-3 / dt)], [4, 3000, int(5e-3 / dt)]]) # Neuron c links to a and e
w.append([[0, 3000, int(9e-3 / dt)], [4, 3000, int(1e-3 / dt)]]) # Neuron d links to a and e
w.append([]) # Neuron e has no outgoing synapses

# Neuron model parameters
C = 300e-12
gamma = 1.0 / C
g_L = 30e-9
V_T = 20e-3
E_L = -70e-3
Rp  = 2 * 1e-3 # refractory time (sec)
ref = np.int(Rp / dt) # number of time-steps for refraction

V    = E_L * np.ones((N, M)) # Array for membrane potentials
I    = np.zeros((N, M)) # Array for input synaptic currents
refr = np.zeros(N) # Refraction states for the neurons

# Inject external currents
tp = int(1e-3/dt) # duration of the pulse
# Case 1
I[1][ : tp] += 50e-9
I[2][4 * tp : 5 * tp] += 50e-9
I[3][8 * tp : 9 * tp] += 50e-9

# # Case 2
# I[1][7 * tp : 8 * tp] += 50e-9
# I[2][3 * tp : 4 * tp] += 50e-9
# I[3][ : tp] += 50e-9

# Inject currents downstream given neuron index, spike time and weight matrix
def updatePostCurrent(i, t, w):
  weights = w[i]
  for syn in weights: # syn[0]: postID, syn[1]: weight, syn[2]: delay
    if(t + syn[2] < M):
      if(t + syn[2] + n > M):
        I[syn[0]][(t + syn[2]) : M] += syn[1] * epsp[:M - (t + syn[2])] 
      else:
        I[syn[0]][t + syn[2] : t + syn[2] + n] += syn[1] * epsp


# Simulate dynnamics of all neurons
for t in range(M-1):
  for i in range(N):
    if(refr[i] == 0): # neuron is not in refractory state
      # 2nd order Runge Kutta
      dV1 = 0.5 * dt * gamma * (-g_L * (V[i][t] - E_L) + I[i][t]) 
      dV2 = dt  * gamma * (-g_L * (V[i][t] + dV1 - E_L) + I[i][t])
      V[i][t + 1] = V[i][t] + dV2 # compute voltage for next timestep
      if(V[i][t + 1] >= V_T):
        V[i][t + 1] = V_T # neuron spikes
        refr[i] = ref # neuron enters absolute refractory state
        updatePostCurrent(i, t, w) # update postsynaptic currents
    else: # neuron is in refractory state
      V[i][t + 1] = E_L 
    
    if(refr[i] > 0): # decrement refractory count
      refr[i] = refr[i] - 1


# Plot synaptic currents
fig, ax = plt.subplots(nrows=5, ncols=1, sharex=True, sharey=True)
fig.text(0.5, 0.04, 'Time (msec)', fontweight='bold', ha='center')
fig.text(0.04, 0.5, 'Current (A)', fontweight='bold', va='center', rotation='vertical')
fig.subplots_adjust(hspace=0)

plt.subplot(5,1,1)
plt.title('Ionic currents', fontweight='bold')
plt.plot(1e3 * dt * np.arange(M), I[0])

plt.subplot(5,1,2)
plt.plot(1e3 * dt * np.arange(M), I[1])

plt.subplot(5,1,3)
plt.plot(1e3 * dt * np.arange(M), I[2])

plt.subplot(5,1,4)
plt.plot(1e3 * dt * np.arange(M), I[3])

plt.subplot(5,1,5)
plt.plot(1e3 * dt * np.arange(M), I[4])
plt.show()

# Plot membrane potentials of all neurons
fig, ax = plt.subplots(nrows=5, ncols=1, sharex=True, sharey=True)
fig.text(0.5, 0.04, 'Time (msec)', fontweight='bold', ha='center')
fig.text(0.04, 0.5, 'Membrane Potential (mV)', fontweight='bold', va='center', rotation='vertical')
fig.subplots_adjust(hspace=0)

plt.subplot(5,1,1)
plt.title('Membrane potentials', fontweight='bold')
plt.plot(1e3 * dt * np.arange(M), 1e3 * V[0])

plt.subplot(5,1,2)
plt.plot(1e3 * dt * np.arange(M), 1e3 * V[1])

plt.subplot(5,1,3)
plt.plot(1e3 * dt * np.arange(M), 1e3 * V[2])

plt.subplot(5,1,4)
plt.plot(1e3 * dt * np.arange(M), 1e3 * V[3])

plt.subplot(5,1,5)
plt.plot(1e3 * dt * np.arange(M), 1e3 * V[4])
plt.show()
