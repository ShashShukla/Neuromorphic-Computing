"""
Hodgkin-Huxley Neuron

Author: Shashwat Shukla
Date: 17th August 2018
"""
# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Runtime parameters
N = 1  # number of neurons
T = 12000 * 1e-3  # total time to simulate (sec)
dt = 0.1 * 1e-3  # simulation time-step (sec)
M = np.int(T / dt)  # number of time-steps
h1 =  np.int(900e-3 / dt) # index when current is started
h2 =  np.int(1500e-3 / dt) # index when current is stopped

# Neuron model parameters
#               C    ENa     EK      El     gNa     gK     gl      I0      Ei
param_list = [1e-6, 50e-3, -77e-3, -55e-3, 120e-3, 36e-3, 0.3e-3, 15e-6, -60e-3]
param_list = np.array(param_list)

V  = np.zeros((M, N))  # Array for membrane potentials
m  = np.zeros((M, N))  # Array for m channel
n  = np.zeros((M, N))  # Array for n channel
h  = np.zeros((M, N))  # Array for h channel
I  = np.zeros((M, N))  # Array for external injected currents
PNa = np.zeros((M, N))  # Array for power of Sodium ion channel currents
PK  = np.zeros((M, N))  # Array for power of Potassium ion channel currents
Pl  = np.zeros((M, N))  # Array for power of leaking currents
PC  = np.zeros((M, N))  # Array for power of membrane capacitor

# Set parameters for the neurons
C   = np.repeat(param_list[0], N)
ENa = np.repeat(param_list[1], N)
EK  = np.repeat(param_list[2], N)
El  = np.repeat(param_list[3], N)
gNa = np.repeat(param_list[4], N) 
gK  = np.repeat(param_list[5], N)
gl  = np.repeat(param_list[6], N)
I0  = np.repeat(param_list[7], N)
Ei  = np.repeat(param_list[8], N)

V[0] = Ei # Initialise voltages close to resting potential
# Specify external currents
for i in range(N):
    I[:, i] = np.ones(M)
    I[:, i][np.arange(M) < h1] = 0
    I[:, i][np.arange(M) > h2] = 0
    I[:, i] = np.multiply(I0[i], I[:, i]) # I(t) = I0 * [H(t - 2T0) - H(t - 3T0)] 

def iNa(V, m, h):
    return np.multiply(np.multiply(gNa, np.power(m, 3)), np.multiply(h, V - ENa))

def iK(V, n):
    return np.multiply(np.multiply(gK, np.power(n, 4)), V - EK)

def il(V):
    return np.multiply(gl, V - El)

def alpham(V):
    return np.divide(np.multiply(0.1, V * 1e3 + 40), 1 - np.exp(np.divide(V * 1e3 + 40, -10)))

def betam(V):
    return np.multiply(4, np.exp(np.multiply(-0.0556, V * 1e3 + 65)))

def alphan(V):
    return np.divide(np.multiply(0.01, V * 1e3 + 55), 1 - np.exp(np.divide(V * 1e3 + 55, -10)))

def betan(V):
    return np.multiply(0.125, np.exp(np.divide(V * 1e3 + 65, -80)))

def alphah(V):
    return np.multiply(0.07, np.exp(np.multiply(-0.05, V * 1e3 + 65)))

def betah(V):
    return np.divide(1, 1 + np.exp(np.multiply(-0.1, V * 1e3 + 35)))

def f1(V, m, n, h, t):
   return np.divide(-iNa(V, m, h) -iK(V, n) -il(V) + I[t], C)

def f2(V, m, t):
    return np.multiply(alpham(V), 1 - m) - np.multiply(betam(V), m)

def f3(V, n, t):
    return np.multiply(alphan(V), 1 - n) - np.multiply(betan(V), n)

def f4(V, h, t):
    return np.multiply(alphah(V), 1 - h) - np.multiply(betah(V), h)


# Simulate the dynamics of the N neurons for M timesteps
for t in range(M - 1):
    # Compute instantaneuous power
    PNa[t] = np.multiply(iNa(V[t], m[t], h[t]), V[t] - ENa)
    PK[t]  = np.multiply(iK(V[t], n[t]), V[t] - EK)
    Pl[t]  = np.multiply(il(V[t]), V[t] - El)
    PC[t]  = np.multiply(-iNa(V[t], m[t], h[t]) -iK(V[t], n[t]) -il(V[t]) + I[t], V[t])

    # Compute values for next time-step using 4th order Runge-Kutta
    k1 = f1(V[t], m[t], n[t], h[t], t)
    l1 = f2(V[t], m[t], t)
    m1 = f3(V[t], n[t], t)
    n1 = f4(V[t], h[t], t)
    k2 = f1(V[t] + 0.5 * dt * k1, m[t] + 0.5 * dt * l1, n[t] + 0.5 * dt * m1, h[t] + 0.5 * dt * n1, t)
    l2 = f2(V[t] + 0.5 * dt * k1, m[t] + 0.5 * dt * l1, t)
    m2 = f3(V[t] + 0.5 * dt * k1, n[t] + 0.5 * dt * m1, t)
    n2 = f4(V[t] + 0.5 * dt * k1, h[t] + 0.5 * dt * n1, t)
    k3 = f1(V[t] + 0.5 * dt * k2, m[t] + 0.5 * dt * l2, n[t] + 0.5 * dt * m2, h[t] + 0.5 * dt * n2, t)
    l3 = f2(V[t] + 0.5 * dt * k2, m[t] + 0.5 * dt * l2, t)
    m3 = f3(V[t] + 0.5 * dt * k2, n[t] + 0.5 * dt * m2, t)
    n3 = f4(V[t] + 0.5 * dt * k2, h[t] + 0.5 * dt * n2, t)
    k4 = f1(V[t] + dt * k3, m[t] + dt * l3, n[t] + dt * m3, h[t] + dt * n3, t)
    l4 = f2(V[t] + dt * k3, m[t] + dt * l3, t)
    m4 = f3(V[t] + dt * k3, n[t] + dt * m3, t)
    n4 = f4(V[t] + dt * k3, h[t] + dt * n3, t)
    dV = dt * (1.0 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    dm = dt * (1.0 / 6) * (l1 + 2 * l2 + 2 * l3 + l4)
    dn = dt * (1.0 / 6) * (m1 + 2 * m2 + 2 * m3 + m4)
    dh = dt * (1.0 / 6) * (n1 + 2 * n2 + 2 * n3 + n4)
    V[t + 1] = V[t] + dV
    m[t + 1] = m[t] + dm
    n[t + 1] = n[t] + dn
    h[t + 1] = h[t] + dh 

# Compute energy dissipated over one action potential
totENa = dt * np.sum(PNa, axis=0)
totEK  = dt * np.sum(PK, axis=0)
totEl  = dt * np.sum(Pl, axis=0)
totEC  = dt * np.sum(PC, axis=0)

print "Total energy of Na channel: ", totENa
print "Total energy of K channel: ", totEK
print "Total energy of leaking channel: ", totEl
print "Total energy of membrane capacitor: ", totEC

# Plot membrane potentials of the neurons
plt.plot(1e3 * dt * np.arange(M), 1000 * V[:, 0])
plt.title('Neuron 1', fontweight='bold')
plt.ylabel('Membrane Potential (mV)', fontweight='bold')
plt.xlabel('Time (msec)', fontweight='bold')
plt.show()

# Plot instantaneous power of the ion channels
plt.plot(1e3 * dt * np.arange(M), PNa[:, 0])
plt.title('Neuron 1', fontweight='bold')
plt.ylabel('Power of Na channel (W)', fontweight='bold')
plt.xlabel('Time (msec)', fontweight='bold')
plt.show()

plt.plot(1e3 * dt * np.arange(M), PK[:, 0])
plt.title('Neuron 1', fontweight='bold')
plt.ylabel('Power of K channel (W)', fontweight='bold')
plt.xlabel('Time (msec)', fontweight='bold')
plt.show()

plt.plot(1e3 * dt * np.arange(M), Pl[:, 0])
plt.title('Neuron 1', fontweight='bold')
plt.ylabel('Power of leaking channel (W)', fontweight='bold')
plt.xlabel('Time (msec)', fontweight='bold')
plt.show()

plt.plot(1e3 * dt * np.arange(M), PC[:, 0])
plt.title('Neuron 1', fontweight='bold')
plt.ylabel('Power of mebrane capacitor (W)', fontweight='bold')
plt.xlabel('Time (msec)', fontweight='bold')
plt.show()
