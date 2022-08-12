"""
Use the Crank-Nicholson method and finite element discretization in
space to numerically integrate the Shrodinger equation in 1D
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
import matplotlib.animation as animation


HBAR = 1.0
M = 1.0
N = 256
DT = 0.0001

s0 = np.linspace(-0.75, 0.75, N+2)
k = 2.0
# X = np.sin(k*s0)/k
X = np.arctanh(s0)
L = X[-1] - X[0]
# print(X)
# plt.plot(s0, X); plt.show(); plt.close()
# plt.scatter(X, np.zeros(len(X)), s=0.1); plt.show(); plt.close()


def curr2_int(x0, x1, x2):
    return ((x1**3/3 - x0**3/3 - x0*x1**2 + x0**3 + x0**2*dx01)/dx01**2 + 
            (x2**3/3 - x1**3/3 - x2**3 + x2*x1**2 + x2**2*dx12)/dx12**2)


def curr3_int(x0, x1, x2):
    return (-(-x1**4/4 + x1**3*x2
              - 3*x1**2*x2**2/2 + x1*x2**3 - x2**4/4)/(x1 - x2)**2
            + (x0**4/4 - x0**3*x1 +
               3*x0**2*x1**2/2 - x0*x1**3 + x1**4/4)/(x0 - x1)**2)


def curr2_next1_int(x1, x2):
    return -(-x1**4/4 - x1**3*(-x1/3 - 2*x2/3) + x1**2*x2**2
             - x1**2*(x1*x2 + x2**2/2) - x1*x2**3 + x2**4/4
             + x2**3*(-x1/3 - 2*x2/3)
             + x2**2*(x1*x2 + x2**2/2))/(x1 - x2)**3


def curr1_next2_int(x0, x1):
    return -(x0**4/4 - x0**3*x1 - x0**3*(2*x0/3 + x1/3)
             + x0**2*x1**2 - x0**2*(-x0**2/2 - x0*x1) - x1**4/4
             + x1**3*(2*x0/3 + x1/3)
             + x1**2*(-x0**2/2 - x0*x1))/(x0 - x1)**3


S = np.zeros([N, N]) # Overlap Matrix
T = np.zeros([N, N]) # Kinetic Term
U = np.zeros([N, N]) # Potential Term
V = 3000.0*X**2 # Harmonic Oscillator
# V = 20000.0*(X**2 - 1.0/9.0)**2 # Double well
# V = 2000.0*abs(X) # abs(x) well
for i in range(1, len(X)-1):
    j = i - 1
    x0, x1, x2 = X[i-1], X[i], X[i+1]
    dx01, dx12 = x1 - x0, x2 - x1
    S[j, j] += curr2_int(x0, x1, x2)
    T[j, j] += (HBAR**2/(2.0*M))*(1.0/dx01 + 1.0/dx12)
    U[j, j] += (V[i]*curr3_int(x0, x1, x2)
                + V[i-1]*curr1_next2_int(x0, x1)
                + V[i+1]*curr2_next1_int(x1, x2))
    if j < N-1:
        U[j, j+1] = (V[i]*curr2_next1_int(x1, x2)
                     + V[i+1]*curr1_next2_int(x1, x2))
        U[j+1, j] = U[j, j+1]
        S[j, j+1] = (x2**3/6.0 - x1**3/6.0
                     + 0.5*(x1**2*x2 - x2**2*x1))/dx12**2
        S[j+1, j] = S[j, j+1]
        T[j, j+1] = -(HBAR**2/(2.0*M))/dx12
        T[j+1, j] = T[j, j+1]

H = T + U
A = S + 1.0j*H*(DT/(2.0*HBAR))
B = S - 1.0j*H*(DT/(2.0*HBAR))
U = np.linalg.inv(A) @ B
# plt.imshow(H); plt.show(); plt.close()
# print(U)
# plt.imshow(np.real(U)); plt.show(); plt.close()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(X[0], X[-1])
pot_range = np.amax(V) - np.amin(V)
ax.set_ylim(-np.amax(V) - pot_range*0.1, 
            np.amax(V) + pot_range*0.1)
ax.set_xlabel('Position')
ax.set_ylabel('Potential Energy')
psi = np.exp(-(X[1: len(X) - 1]/L-0.25)**2/0.1**2)
data = {'psi': psi, 't': 0.0}
psi_scaled = np.zeros([N + 2])
psi_scaled[1: len(X)-1] = pot_range*psi/np.amax(np.abs(psi))
real_plot, = ax.plot(X, np.real(psi_scaled), label=r'Re|$\psi(x, t)$|')
imag_plot, = ax.plot(X, np.imag(psi_scaled), label=r'Im|$\psi(x, t)$|')
abs_plot, = ax.plot(X, np.abs(psi_scaled), color='black', linewidth=1.0, 
                    label=r'|$\psi(x, t)$|')
potential_plot, = ax.plot(X, V, color='gray', 
                          label='Potential V(x)', linewidth=2.0)


def animation_func(*args):
    for _ in range(3):
        psi = U @ data['psi']
        psi_scaled = np.zeros([N + 2], dtype=np.complex128)
        psi_scaled[1: len(X)-1] = pot_range*psi/np.amax(np.abs(psi))
        data['psi'] = psi
        data['t'] += DT
    real_plot.set_ydata(np.real(psi_scaled))
    imag_plot.set_ydata(np.imag(psi_scaled))
    abs_plot.set_ydata(np.abs(psi_scaled))
    return real_plot, imag_plot, abs_plot


data['anim'] = animation.FuncAnimation(fig, animation_func, 
                                       blit=True, interval=1.0)
plt.grid(linestyle='--', linewidth=0.5)
plt.legend()
plt.show()
plt.close()
print(f"Total time elapsed: {data['t']}s")

