"""
Question 4, Homework 3
EE746-Neuromorphic Computing, IITB 

Author: Shashwat Shukla
Date: 30th September 2018
"""
# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Define runtime variables
N  = 200 # number of neurons in the network
T  = 1000 * 1e-3 # total time to simulate (sec)
dt = 0.1 * 1e-3 # simulation time-step (sec)
r  = 100 # mean firing rate, lambda (spikes/sec)
M  = np.int(T / dt) # number of time-steps

I_0   = 1 * 1e-12 # current scaling factor
tau   = 15 * 1e-3 # Time constant 1 for EPSP (sec)
tau_s = 0.25 * tau # Time constant 2 for EPSP (sec)
w_s   = 3000.0 # synaptic weights for external stimulus
w_i   = -3000.0 # synaptic weights for inhibitory neurons
gam_i = 1.0 # w_e = -gam_i * w_i
N_ext = 25 # number of excitatory neurons that receive external stimulus
frac  = 0.8 # 80-20 split between excitatory and inhibitory
N_exc = int(N * frac) # number of excitatory neurons
fan   = int(N / 10) # fanout for each neuron

A_up   = 0.01
A_down = -0.02
tau_l  = 20 * 1e-3 # STDP time consant (sec)

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

# Compute spike-trains and currents
x        = generate_poisson(N_ext, r, dt, M) # generate poisson stimulus for external sources
current  = generate_current(N_ext, x, epsp) # compute external currents
I_ext    = w_s * current # weight the external currents by synaptic weights

# Setup synpatic connections
w_u = [[] for i in range(N)] # array to store upstream synaptic connectivity, weights, delays
w_d = [[] for i in range(N)] # array to store downstream synaptic connectivity, weights, delays
for i in range(N):
  count = 0
  arr_d = []
  if(i < N_exc):
    post  = np.arange(N)
  else:
    post  = np.arange(N_exc) # inhibitory neurons only connect to excitatory neurons
  while(count < fan):
    # Compute ID of postsynaptic neuron
    if(i < N_exc):
      tid = np.random.randint(N - count)
    else:
      tid = np.random.randint(N_exc - count)
    postID = post[tid]
    post = np.delete(post, tid) # remove this postneuron from list
    # Compute synaptic weight and delay
    if(i < N_exc):
      weight = -w_i * gam_i
      delay  = 1e-3 + 19e-3 * np.random.rand()
      delay  = np.int(delay / dt)
    else:
      weight = w_i
      delay  = 1e-3
      delay  = np.int(delay / dt)
    arr_d.append([postID, weight, delay])
    w_u[postID].append([i, len(arr_d) - 1]) # update the upstream array 
    count += 1
  w_d[i] = arr_d # update the downstream array

# Neuron model parameters
C = 300e-12
gamma = 1.0 / C
g_L = 30e-9
V_T = 20e-3
E_L = -70e-3
Rp  = 2 * 1e-3 # refractory time (sec)
ref = np.int(Rp / dt)  # number of time-steps for refraction

V      = E_L * np.ones((N, M))  # Array for membrane potentials
I      = np.zeros((N, M))  # Array for input synaptic currents
refr   = np.zeros(N) # Refraction states for the neurons
spikes = np.zeros((N, M)) # Array to denote when neurons spiked
spikeT = [[] for i in range(N)] # Array to store spiking times
raster = [[] for i in range(N)] # Array to store spiking times for the raster plot (in sec)

# Inject external currents
for i in range(N_ext):
  I[i] = I[i] + I_ext[i]

# Inject currents downstream given neuron index, spike time and weight matrix
def updatePostCurrent(i, t):
  weights = w_d[i]
  for syn in weights: # syn[0]: postID, syn[1]: weight, syn[2]: delay
    if(t + syn[2] < M):
      if(t + syn[2] + n > M):
        I[syn[0]][(t + syn[2]) : M] += syn[1] * epsp[:M - (t + syn[2])] 
      else:
        I[syn[0]][t + syn[2] : t + syn[2] + n] += syn[1] * epsp


def updateSynapticWeights(i, t):
  # Updates for upstream neurons
  for syn in w_u[i]:
    pre_id = syn[0]
    pos    = syn[1]
    weight = w_d[pre_id][pos][1]
    delay  = w_d[pre_id][pos][2]
    t_pre  = np.array(spikeT[pre_id])
    t_pre  = t_pre[t_pre < (t - delay)]
    if(len(t_pre) == 0):
      continue
    t_pre = t_pre[-1] # select the last causal spike time
    dw = weight * A_up * np.exp(-(t - t_pre - delay) * dt / tau_l) # Compute change in weight
    w_d[pre_id][pos][1] += dw
  
  # Updates for downstream neurons
  if(i < N_exc): # check if the neuron is excitatory
    for j, syn in enumerate(w_d[i]):
      post_id = syn[0]
      weight  = syn[1]
      delay   = syn[2]
      t_post  = np.array(spikeT[post_id])
      t_post  = t_post[t_post < (t + delay)]
      if(len(t_post) == 0):
        continue
      t_post = t_post[-1] # select the last causal spike time
      dw = weight * A_down * np.exp(-(t - t_post + delay) * dt / tau_l) # Compute change in weight
      w_d[i][j][1] += dw

we_avg = np.zeros(M)
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
        spikes[i][t] = 1
        spikeT[i].append(t) # time when the neuron fired
        raster[i].append(t * dt * 1e3) # append to the raster plot 
        updatePostCurrent(i, t) # update postsynaptic currents
        updateSynapticWeights(i, t) # STDP updates
    else: # neuron is in refractory state
      V[i][t + 1] = E_L 
    
    if(refr[i] > 0): # decrement refractory count
      refr[i] = refr[i] - 1
  for k in range(N_exc):
    we = w_d[k]
    for r in we:
      we_avg[t] += r[1] / (N_exc * fan)   
  

# Compute Re and Ri
avgExc = np.sum(spikes[ : N_exc], axis = 0) # Sum spikes over excitatory neurons
avgInh = np.sum(spikes[N_exc : ], axis = 0) # Sum spikes over inhibitory neurons
tWin = int(10e-3 / dt) # number of timesteps for 10ms window 
win = np.ones(tWin) / tWin
Re  = np.convolve(avgExc, win)[2 * tWin : M]
Ri  = np.convolve(avgInh, win)[2 * tWin : M]

# Display raster plot
plt.eventplot(raster)
plt.title('Raster plot', fontweight='bold')
plt.ylabel('Neuron index', fontweight='bold')
plt.xlabel('Time (msec)', fontweight='bold')
plt.show()

# Plot Re
plt.plot(1e3 * dt * np.arange(len(Re)), Re)
plt.title('Re(t), Average number of excitatory spikes', fontweight='bold')
plt.ylabel('Number of spikes', fontweight='bold')
plt.xlabel('Time (msec)', fontweight='bold')
plt.show()
# Plot Ri
plt.plot(1e3 * dt * np.arange(len(Ri)), Ri)
plt.title('Ri(t), Average number of inhibitory spikes', fontweight='bold')
plt.ylabel('Number of spikes', fontweight='bold')
plt.xlabel('Time (msec)', fontweight='bold')
plt.show()

# Plot average excitatory weight
plt.plot(1e3 * dt * np.arange(len(we_avg) - 1), we_avg[:M-1])
plt.title('Average excitatory weight', fontweight='bold')
plt.ylabel('Weight', fontweight='bold')
plt.xlabel('Time (msec)', fontweight='bold')
plt.show()