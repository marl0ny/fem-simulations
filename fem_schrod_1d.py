"""
Numerically integrate the Schrodinger equation in 1D using
a finite element discretization
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh


HBAR = 1.0
M = 1.0
N = 50


s0 = np.linspace(-0.75, 0.75, N+2)
k = 2.0
# X = np.sin(k*s0)/k
X = np.arctanh(s0)
# print(X)
# plt.plot(s0, X); plt.show(); plt.close()
# plt.scatter(X, np.zeros(len(X))); plt.show(); plt.close()

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
# V = 1500.0*X**2 # Harmonic Oscillator
V = 20000.0*(X**2 - 1.0/9.0)**2 # Double well
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


n_states = 5
eigvals, eigvects = eigsh(T + U, M=S, k=n_states,
                          which='LM', sigma=0.0)
eigvect = np.zeros([N + 2])
for n in range(3):
    print(eigvals[n])
    eigvect[1: len(X)-1] = eigvects.T[n]
    plt.plot(X, eigvect, alpha=0.1 + 0.9/(n + 1.0), label=f'n = {n}')
plt.plot(X, 50*V/np.amax(V), label='V(x)', color='gray', linestyle='--')
plt.legend()
plt.ylim(np.amin(eigvects)-0.1, np.amax(eigvects)+0.1)
plt.xlim(X[0], X[-1])
plt.xlabel('x')
plt.title('Energy Eigenstates')
plt.grid()
plt.show()
plt.close()

