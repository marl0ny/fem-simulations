"""
Linear elasticity using the finite element method in 3D.
The vibrational modes are computed using an eigensolver.

Plenty of bookkeeping is involved in writing this script; it is
possible some errors still remain.


Linear Elasticity references:
 - https://people.duke.edu/~hpgavin/StructuralDynamics/StructuralElements.pdf
 - https://www.mit.edu/~nnakata/page/Teaching_files/GEOPHYS130/
   GEOPHYS130_notes_all.pdf
 - https://www.math.uci.edu/~chenlong/226/elasticity.pdf

Linear basis functions on tetrahedral elements using natural coordinates
are used. The polynomial integration formula is found here:
 - T.J. Chung, "Finite Element Interpolation Functions", 
   in Computational Fluid Dynamics, 2nd ed, CUP, 2010, 
   ch 9, pp. 262-308.

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import linalg
from scipy.sparse import csr_matrix, dok_matrix
import matplotlib.animation as animation
from read_m_mesh import get_vertices_and_edges


# Lame constants
LAMBDA = 1.0
MU = 3.0
# Young modulus
E = MU*(3.0*LAMBDA + 2.0*MU)/(LAMBDA + MU)
# Poisson ratio
NU = 0.5*LAMBDA/(LAMBDA + MU)
# print(NU)
# Elastic Matrix
C = np.array([[1.0-NU, NU, NU, 0.0, 0.0, 0.0],
              [NU, 1.0-NU, NU, 0.0, 0.0, 0.0],
              [NU, NU, 1.0-NU, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, (1.0 - 2.0*NU)/2.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, (1.0 - 2.0*NU)/2.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, (1.0 - 2.0*NU)/2.0]]
             )*E/((1.0 + NU)*(1.0 - 2.0*NU))

DT = 0.5
# Density
RHO = 1.0
# Dissipation
# GAMMA1 = 0.0
# GAMMA1 = 0.01
# GAMMA1 = 0.0012
# Rayleigh Damping
GAMMA1 = 0.007
# GAMMA1 = 0.0004
GAMMA2 = 1.0*GAMMA1


def get_volume_of_element(element_vertices):
    x, y, z = 0, 1, 2
    x0, y0, z0 = element_vertices[0][x], element_vertices[0][y], \
        element_vertices[0][z]
    x1, y1, z1 = element_vertices[1][x], element_vertices[1][y], \
        element_vertices[1][z]
    x2, y2, z2 = element_vertices[2][x], element_vertices[2][y], \
        element_vertices[2][z]
    x3, y3, z3 = element_vertices[3][x], element_vertices[3][y], \
        element_vertices[3][z]
    return -(x0*y1*z2 - x0*y1*z3 - x0*y2*z1 + x0*y2*z3 + x0*y3*z1
             - x0*y3*z2 - x1*y0*z2 + x1*y0*z3 + x1*y2*z0 - x1*y2*z3
             - x1*y3*z0 + x1*y3*z2 + x2*y0*z1 - x2*y0*z3 - x2*y1*z0
             + x2*y1*z3 + x2*y3*z0 - x2*y3*z1 - x3*y0*z1 + x3*y0*z2
             + x3*y1*z0 - x3*y1*z2 - x3*y2*z0 + x3*y2*z1)/6.0


def get_to_tetrahedral_coord_matrix(element_vertices):
    x, y, z = 0, 1, 2
    x0, y0, z0 = element_vertices[0][x], element_vertices[0][y], \
        element_vertices[0][z]
    x1, y1, z1 = element_vertices[1][x], element_vertices[1][y], \
        element_vertices[1][z]
    x2, y2, z2 = element_vertices[2][x], element_vertices[2][y], \
        element_vertices[2][z]
    x3, y3, z3 = element_vertices[3][x], element_vertices[3][y], \
        element_vertices[3][z]
    v_6 = -(x0*y1*z2 - x0*y1*z3 - x0*y2*z1 + x0*y2*z3 + x0*y3*z1
            - x0*y3*z2 - x1*y0*z2 + x1*y0*z3 + x1*y2*z0 - x1*y2*z3
            - x1*y3*z0 + x1*y3*z2 + x2*y0*z1 - x2*y0*z3 - x2*y1*z0
            + x2*y1*z3 + x2*y3*z0 - x2*y3*z1 - x3*y0*z1 + x3*y0*z2
            + x3*y1*z0 - x3*y1*z2 - x3*y2*z0 + x3*y2*z1)
    return np.array([[(x1*y2*z3 - x1*y3*z2 - x2*y1*z3
                       + x2*y3*z1 + x3*y1*z2 - x3*y2*z1)/v_6,
                      (-y1*z2 + y1*z3 + y2*z1 - y2*z3 - y3*z1 + y3*z2)/v_6,
                      (x1*z2 - x1*z3 - x2*z1 + x2*z3 + x3*z1 - x3*z2)/v_6,
                      (-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)/v_6],
                      [(-x0*y2*z3 + x0*y3*z2 + x2*y0*z3
                        - x2*y3*z0 - x3*y0*z2 + x3*y2*z0)/v_6,
                       (y0*z2 - y0*z3 - y2*z0 + y2*z3 + y3*z0 - y3*z2)/v_6,
                       (-x0*z2 + x0*z3 + x2*z0 - x2*z3 - x3*z0 + x3*z2)/v_6,
                       (x0*y2 - x0*y3 - x2*y0 + x2*y3 + x3*y0 - x3*y2)/v_6],
                      [(x0*y1*z3 - x0*y3*z1 - x1*y0*z3
                        + x1*y3*z0 + x3*y0*z1 - x3*y1*z0)/v_6,
                       (-y0*z1 + y0*z3 + y1*z0 - y1*z3 - y3*z0 + y3*z1)/v_6,
                       (x0*z1 - x0*z3 - x1*z0 + x1*z3 + x3*z0 - x3*z1)/v_6,
                       (-x0*y1 + x0*y3 + x1*y0 - x1*y3 - x3*y0 + x3*y1)/v_6],
                      [(-x0*y1*z2 + x0*y2*z1 + x1*y0*z2
                        - x1*y2*z0 - x2*y0*z1 + x2*y1*z0)/v_6,
                       (y0*z1 - y0*z2 - y1*z0 + y1*z2 + y2*z0 - y2*z1)/v_6,
                       (-x0*z1 + x0*z2 + x1*z0 - x1*z2 - x2*z0 + x2*z1)/v_6,
                       (x0*y1 - x0*y2 - x1*y0 + x1*y2 + x2*y0 - x2*y1)/v_6]])



def get_deformation_matrix(element_vertices):
    diffs = get_to_tetrahedral_coord_matrix(element_vertices).T
    x_diffs, y_diffs, z_diffs  = diffs[1], diffs[2], diffs[3]
    D = np.zeros([6, 12])
    D[0, 0:4] = x_diffs
    D[1, 4:8] = y_diffs
    D[2, 8:12] = z_diffs
    D[3, 0:4], D[3, 4:8] = y_diffs, x_diffs
    D[4, 0:4], D[4, 8:12] = z_diffs, x_diffs
    D[5, 4:8], D[5, 8:12] = z_diffs, y_diffs
    # tmp = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
    # D = tmp @ D
    return D


def get_stiffness_matrix(element_vertices):
    D = get_deformation_matrix(element_vertices)
    vol = get_volume_of_element(element_vertices)
    k = 6.0*vol**2*(D.T @ C @ D)
    return k


def get_force_on_element(f, element_vertices):
    G = np.zeros([3, 12])
    G[0, 0:4], G[1, 4:8], G[2, 8:12] = np.ones([4]), np.ones([4]), np.ones([4])
    vol = get_volume_of_element(element_vertices)
    return vol*(f.T @ G)/4.0


vertices, elements = get_vertices_and_edges('./data/prism.txt')
index_set = set()


def in_boundary(vertex):
    return vertex[0] < 0.0 + 1e-6


vertices_index = np.arange(1, vertices.shape[0] + 1)
dist = np.zeros([vertices.shape[0]])
interior_vertices = np.array([vertices[i] for i in range(vertices.shape[0])
                              if not in_boundary(vertices[i])])
boundary_vertices = np.array([vertices[i] for i in range(vertices.shape[0])
                              if in_boundary(vertices[i])])
interior_vertices_index = np.array([vertices_index[i]
                                    for i in range(vertices.shape[0])
                                    if not in_boundary(vertices[i])])
to_interior_indices = {interior_vertices_index[i]: i
                       for i in range(interior_vertices.shape[0])}
to_all_indices = {i: interior_vertices_index[i]
                  for i in range(interior_vertices.shape[0])}
to_interior_indices_keys = to_interior_indices.keys()


def in_interior(index):
    return index in to_interior_indices_keys


N = interior_vertices.shape[0]
K = dok_matrix((3*N, 3*N))
M = dok_matrix((3*N, 3*N))
Fg = 0.001
f = np.zeros([3*N])

for k, element in enumerate(elements):
    element_vertices = [vertices[int(element[0])-1],
                        vertices[int(element[1])-1],
                        vertices[int(element[2])-1],
                        vertices[int(element[3])-1]]
    stiff_mat = get_stiffness_matrix(element_vertices)
    volume = get_volume_of_element(element_vertices)
    fg_vec = Fg*np.array([0.0, -1.0, 0.0]) # /np.sqrt(2.0)
    fe = get_force_on_element(fg_vec, element_vertices)
    for i in range(len(element_vertices)):
        if in_interior(element[i]):
            m = to_interior_indices[element[i]]
            ix, iy, iz = i, i + 4, i + 8
            mx, my, mz = m, m + N, m + 2*N
            f[mx] += fe[ix]
            f[my] += fe[iy]
            f[mz] += fe[iz]
            for j in range(0, i+1):
                if in_interior(element[j]):
                    n = to_interior_indices[element[j]]
                    jx, jy, jz = j, j + 4, j + 8
                    nx, ny, nz = n, n + N, n + 2*N
                    stiff_mxnx = stiff_mat[ix, jx]
                    stiff_myny = stiff_mat[iy, jy]
                    stiff_mznz = stiff_mat[iz, jz]
                    stiff_mxny = stiff_mat[ix, jy]
                    stiff_mynx = stiff_mat[iy, jx]
                    stiff_mxnz = stiff_mat[ix, jz]
                    stiff_mznx = stiff_mat[iz, jx]
                    stiff_mynz = stiff_mat[iy, jz]
                    stiff_mzny = stiff_mat[iz, jy]
                    # Stiffness
                    K[mx, nx] += stiff_mxnx
                    K[my, ny] += stiff_myny
                    K[mz, nz] += stiff_mznz
                    K[mx, ny] += stiff_mxny
                    K[my, nx] += stiff_mynx
                    K[mx, nz] += stiff_mxnz
                    K[mz, nx] += stiff_mznx
                    K[my, nz] += stiff_mynz
                    K[mz, ny] += stiff_mzny
                    # Mass
                    M[mx, nx] += volume/20.0
                    M[my, ny] += volume/20.0
                    M[mz, nz] += volume/20.0
                    M[nx, mx] += volume/20.0
                    M[ny, my] += volume/20.0
                    M[nz, mz] += volume/20.0
                    if m != n:
                        K[nx, mx] += stiff_mxnx
                        K[ny, my] += stiff_myny
                        K[nz, mz] += stiff_mznz
                        K[ny, mx] += stiff_mxny
                        K[nx, my] += stiff_mynx
                        K[nz, mx] += stiff_mxnz
                        K[nx, mz] += stiff_mznx
                        K[nz, my] += stiff_mynz
                        K[ny, mz] += stiff_mzny


# tmp = K.toarray()
# print(np.amax(np.abs(tmp - tmp.T)))
M = csr_matrix(M)
K = csr_matrix(K)
xb = boundary_vertices

n_eigs = 6
eigvals, eigvects = linalg.eigsh(K, M=M, which='LM', sigma=0.0, k=n_eigs)
fm = linalg.cg(K, f)[0]
amplitudes = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])/2.0
frequencies = np.zeros([n_eigs], dtype=np.complex128)
for i in range(n_eigs):
    frequency = 0.5*(-(GAMMA1 + GAMMA2*eigvals[i])
                     + (0.0j + (GAMMA1 + GAMMA2*eigvals[i])**2
                        - 4.0*eigvals[i])**0.5)
    frequencies[i] = frequency
start_phases = np.zeros([n_eigs], dtype=np.complex128)
u = np.zeros([3*N])
i = 0
for a, phi in zip(amplitudes, start_phases):
    eigvect = eigvects.T[i]
    u += a*eigvect*np.cos(np.imag(phi))*np.exp(np.real(phi))
    i += 1

x = interior_vertices.copy()
zeros = np.zeros([3*N])
frames = [u.copy() + fm, u.copy() + fm]
data = {'t': 0.0}

fig = plt.figure()
fig.suptitle('Linear Elasticity Using FEM in 3D')
axes = fig.subplots(1, 3, sharey=True)
axes[0].set_aspect('equal')
axes[1].set_aspect('equal')
axes[2].set_aspect('equal')
axes[0].scatter(xb.T[1], xb.T[2], alpha=0.5, color='black')
yz_data = axes[0].scatter(x.T[1], x.T[2], alpha=0.5)
axes[0].set_xlabel('yz-plane')
axes[1].scatter(xb.T[0], xb.T[2], alpha=0.5, color='black')
xz_data = axes[1].scatter(x.T[0], x.T[2], alpha=0.5)
axes[1].set_xlabel('xz-plane')
axes[2].scatter(xb.T[0], xb.T[1], alpha=0.5, color='black')
xy_data = axes[2].scatter(x.T[0], x.T[1], alpha=0.5)
axes[2].set_xlabel('xy-plane')
for i in range(3):
    if i == 0:
        axes[i].set_xlim(-4.0, 1.0)
        axes[i].set_ylim(-4.0, 4.0)
    else:
        axes[i].set_xlim(-1.0, 4.0)
        axes[i].set_ylim(-4.0, 4.0)


def animation_func(*arg):
    ux, uy, uz = np.zeros([N]), np.zeros([N]), np.zeros([N])
    x = np.zeros([3, N])
    for _ in range(1):
        data['t'] += DT
        t = data['t']
        x0 = interior_vertices.copy()
        u0, u1 = frames[0], frames[1]
        uxyz = np.zeros([3*N])
        for i, a in enumerate(amplitudes):
            w_r, w_i = np.real(frequencies[i]), np.imag(frequencies[i])
            p_r, p_i = np.real(start_phases[i]), np.imag(start_phases[i])
            uxyz += a*eigvects.T[i]*np.exp(w_r*t + p_r)*np.cos(w_i*t + p_i)
        uxyz += fm
        # f2 = (DT**2*(-0.5*K @ u0) + 2.0*M @ u1 - M @ u0
        #        + DT**2*f
        #        + 0.5*DT*(GAMMA1*M + GAMMA2*K) @ u0
        #       )
        # uxyz = linalg.cg(M + DT**2*0.5*K + 0.5*DT*(M*GAMMA1 + K*GAMMA2), f2)[0]
        ux, uy, uz = uxyz[0:N], uxyz[N: 2*N], uxyz[2*N: 3*N]
        frames[0], frames[1] = u1, uxyz
        x[0:N], x[1:N], x[2:N] = x0.T[0] + ux, x0.T[1] + uy, x0.T[2] + uz
    yz_data.set(offsets=np.array([x[1], x[2]]).T)
    xz_data.set(offsets=np.array([x[0], x[2]]).T)
    xy_data.set(offsets=np.array([x[0], x[1]]).T)
    return yz_data, xz_data, xy_data



anim = animation.FuncAnimation(fig, animation_func,
                               blit=True, interval=1.0)
plt.show()
plt.close()

