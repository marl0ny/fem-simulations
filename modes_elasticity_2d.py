import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import linalg
from scipy.sparse import csr_matrix, dok_matrix
import matplotlib.animation as animation


# Lame constants
LAMBDA = 1.0
MU = 1.0
# Young modulus
E = MU*(3.0*LAMBDA + 2.0*MU)/(LAMBDA + MU)
# Poisson ratio
NU = 0.5*LAMBDA/(LAMBDA + MU)
# Elastic Matrix
C = np.array([[1.0, NU, 0.0],
              [NU, 1.0, 0.0],
              [0.0, 0.0, 0.5*(1.0 - NU)]])
# Time step
DT = 0.5
# Density (Assume each element has the same mass)
RHO = 1.0
# Dissipation
# GAMMA1 = 0.0
# GAMMA1 = 0.01
# GAMMA1 = 0.0012
# Rayleigh Damping
GAMMA1 = 0.0007
# GAMMA1 = 0.0004
GAMMA2 = 1.0*GAMMA1


def get_area_of_element(element_vertices):
    x, y = 1, 2
    x10 = element_vertices[0][x] - element_vertices[1][x]
    x20 = element_vertices[0][x] - element_vertices[2][x]
    y20 = element_vertices[0][y] - element_vertices[2][y]
    y10 = element_vertices[0][y] - element_vertices[1][y]
    return (x10*y20 - x20*y10)/2.0


def get_deformation_matrix(element_vertices):
    """Get the deformation matrix for a single element
    """
    x, y = 1, 2
    x21 = -element_vertices[2][x] + element_vertices[1][x]
    x02 = -element_vertices[0][x] + element_vertices[2][x]
    x10 = -element_vertices[1][x] + element_vertices[0][x]
    y12 = -element_vertices[1][y] + element_vertices[2][y]
    y20 = -element_vertices[2][y] + element_vertices[0][y]
    y01 = -element_vertices[0][y] + element_vertices[1][y]
    return np.array([[y12, y20, y01, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, x21, x02, x10],
                     [x21, x02, x10, y12, y20, y01]])


def get_stiffness(phi1, phi2, element_vertices):
    D = get_deformation_matrix(element_vertices)
    area = get_area_of_element(element_vertices)
    return area*phi1.T @ D.T @ C @ D @ phi2


def get_force(f, phi, element_vertices):
    G = np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    area = get_area_of_element(element_vertices)
    return area*f.T @ G @ phi/3.0


vertices_array = np.loadtxt('./data/rectangle.node', skiprows=1)
elements_array = np.loadtxt('./data/rectangle.1.ele', skiprows=1)
boundary_vertices = np.array([v for v in vertices_array if v[3] > 0.0])
interior_vertices = np.array([v for v in vertices_array if v[3] < 1.0])
to_all_indices = {i: int(v[0]) for i, v in enumerate(interior_vertices)}
to_interiors_indices = {int(v[0]): i for i, v in enumerate(interior_vertices)}
N = len(interior_vertices)

K = dok_matrix((2*N, 2*N)) # Stiffness Matrix
M = dok_matrix((2*N, 2*N)) # Mass Matrix
f = np.zeros([2*N])
Fg = -0.001

for k in elements_array.T[0]:
    element = elements_array[int(k)-1]
    element_vertices = (vertices_array[int(element[1])-1],
                        vertices_array[int(element[2])-1],
                        vertices_array[int(element[3])-1])
    area = get_area_of_element(element_vertices)
    mass = RHO*area/12.0
    vertices_per_element = 3
    for i in range(vertices_per_element):
        v_i = element_vertices[i]
        if v_i[3] < 1.0:
            m = to_interiors_indices[v_i[0]]
            phi1_x = np.array([1.0 if i == s else 0.0 for s in range(6)])
            phi1_y = np.array([1.0 if 3 + i == s else 0.0 for s in range(6)])
            f[m] += get_force(np.array([0.0, Fg]), phi1_x, element_vertices)
            f[m + N] += get_force(np.array([0.0, Fg]), phi1_y,
                                  element_vertices)
            for j in range(0, i+1):
                v_j = element_vertices[j]
                phi2_x = np.array([1.0 if j == s else 0.0 for s in range(6)])
                phi2_y = np.array([1.0 if 3 + j == s 
                                   else 0.0 for s in range(6)])
                if v_j[3] < 1.0:
                    stiffness_xx = get_stiffness(phi1_x, phi2_x,
                                                 element_vertices)
                    stiffness_xy = get_stiffness(phi1_x, phi2_y,
                                                element_vertices)
                    stiffness_yx = get_stiffness(phi1_y, phi2_x,
                                                 element_vertices)
                    stiffness_yy = get_stiffness(phi1_y, phi2_y,
                                                 element_vertices)
                    n = to_interiors_indices[v_j[0]]
                    M[m, n] += mass
                    M[N + m, N + n] += mass
                    K[m, n] += stiffness_xx
                    K[m, N + n] += stiffness_xy
                    K[N + m, n] += stiffness_yx
                    K[N + m, N + n] += stiffness_yy
                    if m != n:
                        M[n, m] += mass
                        M[N + n, N + m] += mass
                        K[n, m] += stiffness_xx
                        K[N + n, m] += stiffness_xy
                        K[n, N + m] += stiffness_yx
                        K[N + n, N + m] += stiffness_yy


M = csr_matrix(M)
K = csr_matrix(K)
# plt.imshow(M.toarray())
# plt.show()
# plt.close()
x0 = interior_vertices.T[1]
y0 = interior_vertices.T[2]
xb = boundary_vertices.T[1]
yb = boundary_vertices.T[2]
zeros = np.zeros([2*N])
ones = np.zeros([2*N])
frames = [zeros, zeros]
# frames = [zeros, np.array(list(0.03*x0)
#                           + list(0.03*y0))]
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
ax.set_title('Linear Elasticity Using FEM in 2D')
ax.scatter(xb, yb, color='black')
plot_data = ax.scatter(x0, y0, label='Time discretization')
plot_data2 = ax.scatter(x0, y0, label='Eigenvalue method')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_ylim(-15.0, 15.0)
ax.set_xlim(-1.0, 35.0)
n_eigs = 6
eigvals, eigvects = linalg.eigsh(K, M=M, which='LM', sigma=0.0, k=n_eigs)
amplitudes = np.array([20.0, 0.0, 5.0, -5.0, 5.0, -20.0])
frequencies = np.zeros([n_eigs], dtype=np.complex128)
for i in range(n_eigs):
    frequency = 0.5*(-(GAMMA1 + GAMMA2*eigvals[i])
                     + (0.0j + (GAMMA1 + GAMMA2*eigvals[i])**2
                        - 4.0*eigvals[i])**0.5)
    frequencies[i] = frequency
start_phases = np.zeros([n_eigs], dtype=np.complex128)
n_eig = -1
ux, uy = eigvects.T[n_eig, 0:N].copy(), eigvects.T[n_eig, N:2*N].copy()
ux2, uy2 = eigvects.T[n_eig, 0:N].copy(), eigvects.T[n_eig, N:2*N].copy()
fm = linalg.cg(K, f)[0]
if n_eig >= 0:
    u = 20.0*eigvects.T[n_eig, 0:2*N]
    omega =  0.5*(-(GAMMA1 + GAMMA2*eigvals[n_eig])
                     + (0.0j + (GAMMA1 + GAMMA2*eigvals[n_eig])**2
                        - 4.0*eigvals[i])**0.5)
    omega_r, omega_i = np.real(omega), np.imag(omega)
else:
    u = np.zeros([2*N])
    i = 0
    for a, phi in zip(amplitudes, start_phases):
        eigvect = eigvects.T[i]
        u += a*eigvect*np.cos(np.imag(phi))*np.exp(np.real(phi))
        i += 1
frames = [u.copy() + fm, u.copy() + fm]

data = {'t': 0.0}
# print(f)

def animation_func(*args):
    for _ in range(1):
        data['t'] += DT
        t = data['t']
        u0, u1 = frames[0], frames[1]
        f2 = (DT**2*(-0.5*K @ u0) + 2.0*M @ u1 - M @ u0
               + DT**2*f
               + 0.5*DT*(GAMMA1*M + GAMMA2*K) @ u0
              )
        uxy = linalg.cg(M + DT**2*0.5*K + 0.5*DT*(M*GAMMA1 + K*GAMMA2), f2)[0]
        if n_eig >= 0:
            uxy2 = u*np.exp(omega_r*t)*np.cos(omega_i*t) + fm
        else:
            uxy2 = np.zeros([2*N])
            for i, a in enumerate(amplitudes):
                w_r, w_i = np.real(frequencies[i]), np.imag(frequencies[i])
                p_r, p_i = np.real(start_phases[i]), np.imag(start_phases[i])
                uxy2 += (a*eigvects.T[i]*
                         np.exp(w_r*t + p_r)*np.cos(w_i*t + p_i))
            uxy2 += fm
        ux[0: N], uy[0: N] = uxy[0: N], uxy[N: 2*N]
        ux2[0: N], uy2[0: N] = uxy2[0: N], uxy2[N: 2*N]
        frames[0], frames[1] = u1, uxy
    plot_data.set(offsets=np.array([x0 + ux, y0 + uy]).T)
    plot_data2.set(offsets=np.array([x0 + ux2, y0 + uy2]).T)
    return plot_data, plot_data2


anim = animation.FuncAnimation(fig, animation_func,
                               blit=True, interval=1.0)
# plot_data.set(offsets=np.array([x0 + 20.0*ux, y0 + 20.0*uy]).T)
plt.legend()
plt.show()
plt.close()
