"""
Question 4, Homework 2
(Alternative interpretation)
EE746-Neuromorphic Computing, IITB 

Author: Shashwat Shukla
Date: 6th September 2018
"""
# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Define runtime variables
N  = 100  # number of presynaptic neurons
T  = 500 * 1e-3  # total time to simulate (sec)
dt = 0.1 * 1e-3  # simulation time-step (sec)
r  = 1 # mean firing rate, lambda (spikes/sec)
M  = np.int(T / dt)  # number of time-steps

I_0   = 1 * 1e-12 # current scaling factor
tau   = 15 * 1e-3 # Time constant 1 for EPSP (sec)
tau_s = 0.25 * tau # Time constant 2 for EPSP (sec)
mu    = 250 # mean of gaussian, w_0
sigma = 25 # standard deviation of gaussian, sigma_w
gamma = 1 # Learning rate

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

# Compute spiketrains and currents
x       = generate_poisson(N, r, dt, M) # generate the spiking patterns of presynaptic neurons
current = generate_current(N, x, epsp) # compute postsynaptic currents
w       = np.random.normal(mu, sigma, N) # generate syaptic weights
I_post  = np.transpose(np.multiply(np.transpose(current), w)) # compute weighted postsynaptic currents
I  = np.sum(I_post, axis = 0) # compute total postsynaptic current

# Neuron model parameters
#                C       gL       EL     VT    delT    a    tauW     b        Vr
param_list = [[200e-12, 10e-9, -70e-3, -50e-3, 2e-3, 2e-9,  30e-3,   0e-12, -58e-3],  # RS
              [130e-12, 18e-9, -58e-3, -50e-3, 2e-3, 4e-9, 150e-3, 120e-12, -50e-3],  # IB
              [200e-12, 10e-9, -58e-3, -50e-3, 2e-3, 2e-9, 200e-3, 100e-12, -46e-3]]  # CH
param_list = np.array(param_list)

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

# AEF neuron equation for V
def f(V, U, I, t):
   return np.divide(np.multiply(-gL, V - EL) + np.multiply(np.multiply(gL, delT), np.exp(np.divide(V - VT, delT))) - U + I[t], C)

# AEF neuron equation for U
def g(V, U, I, t):
    return np.multiply(a, V - EL) - U

# Compute the postsynaptic voltage given input current and number of timesteps
def compute_postV(I, M, dt):
  V  = np.zeros((M,1))  # Array for membrane potentials
  U  = np.zeros((M,1))  # Array for recovery variable
  V[0] = EL # Initialise voltages, U is initialised to zero above
  # Simulate the dynamics of the AEF neuron for M timesteps
  for t in range(M - 1):
    # Forward Euler
    k1 = f(V[t], U[t], I, t)
    l1 = g(V[t], U[t], I, t)
    dV = dt * k1
    dU = dt * l1
    V[t + 1] = V[t] + dV # compute voltage for next timestep
    U[t + 1] = U[t] + dU # compute recovery variable for next timestep
    V[t + 1][V[t + 1] >= 0] = 0 # neurons spike
    V[t + 1][V[t] == 0] = Vr[V[t] == 0] # voltage after spike
    U[t + 1][V[t] == 0] = U[t][V[t] == 0] + b[V[t] == 0] # recovery variable after spike
  return V

def update_synapticWeights(x, w, N, maxIndex):
  last = []
  for i in range(N):
    spikes = np.where(x[i] > 0)[0] # indices where neuron spiked
    if(spikes.size == 0):
      last.append(-1)
      continue
    preSpikes = spikes[spikes < maxIndex] # spikes before the max event
    if(preSpikes.size == 0):
      last.append(-1)
      continue
    lastSpike = preSpikes[-1] # pick spike just before the max event
    last.append(lastSpike)
  last = np.array(last)
  i = np.argmax(last)
  deltaT = (maxIndex - last[i]) * dt
  deltaW = w[i] * gamma * (np.exp(-deltaT / tau) - np.exp(-deltaT / tau_s))
  w[i] = w[i] - deltaW # update synaptic weight
  if(w[i] < 10):
    w[i] = 10
  return w

V = compute_postV(I, M, dt) # compute postsynaptic voltage
V_init = V # voltage before weight updates
max_iter = 500 # maximum number of iterations
# Iterate until the neuron spikes
for i in range(max_iter):
  maxV = np.max(V)
  # Check if neuron has spiked
  if(maxV < 0):
    print "Neuron did not spike"
    print "Number of iterations required: ", i
    break
  maxIndex = np.where(V == 0)[0] # indices of when the neuron spiked  
  # Update synaptic weights
  for j in maxIndex: # iterate over all spikes
    w = update_synapticWeights(x, w, N, j)
  # Recompute postsynaptic current and voltage  
  I_post = np.transpose(np.multiply(np.transpose(current), w)) # compute weighted postsynaptic currents
  I = np.sum(I_post, axis = 0) # compute total postsynaptic current
  V = compute_postV(I, M, dt)


# Plot membrane potentials of the AEF neuron before weight updates
plt.plot(1e3 * dt * np.arange(M), V_init[:, 0])
plt.title('AEF RS Neuron, before updates', fontweight='bold')
plt.ylabel('Membrane Potential (V)', fontweight='bold')
plt.xlabel('Time (msec)', fontweight='bold')
plt.show()

# Plot membrane potentials of the AEF neuron after weight updates
plt.plot(1e3 * dt * np.arange(M), V[:, 0])
plt.title('AEF RS Neuron, after updates', fontweight='bold')
plt.ylabel('Membrane Potential (V)', fontweight='bold')
plt.xlabel('Time (msec)', fontweight='bold')
plt.show()
