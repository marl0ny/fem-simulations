"""
Static linear elasticity using the finite element method in 3D.

Plenty of bookkeeping is involved in writing this script; it is
possible some errors still remain.

Notes on Linear Elasticity:
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
MU = 1.0
# Young modulus
E = MU*(3.0*LAMBDA + 2.0*MU)/(LAMBDA + MU)
# Poisson ratio
NU = 0.5*LAMBDA/(LAMBDA + MU)
# Elastic Matrix
C = np.array([[1.0-NU, NU, NU, 0.0, 0.0, 0.0],
              [NU, 1.0-NU, NU, 0.0, 0.0, 0.0],
              [NU, NU, 1.0-NU, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, (1.0 - 2.0*NU)/2.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, (1.0 - 2.0*NU)/2.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, (1.0 - 2.0*NU)/2.0]]
             )*E/((1.0 + NU)*(1.0 - 2.0*NU))


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


vertices, elements = get_vertices_and_edges('./data/cylinder.txt')

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
print(N)
K = dok_matrix((3*N, 3*N))
Fg = 0.0005
f = np.zeros([3*N])

for k, element in enumerate(elements):
    element_vertices = [vertices[int(element[0])-1],
                        vertices[int(element[1])-1],
                        vertices[int(element[2])-1],
                        vertices[int(element[3])-1]]
    stiff_mat = get_stiffness_matrix(element_vertices)
    fg_vec = Fg*np.array([0.0, -3.0, 0.0])
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
                    K[mx, nx] += stiff_mxnx
                    K[my, ny] += stiff_myny
                    K[mz, nz] += stiff_mznz
                    K[mx, ny] += stiff_mxny
                    K[my, nx] += stiff_mynx
                    K[mx, nz] += stiff_mxnz
                    K[mz, nx] += stiff_mznx
                    K[my, nz] += stiff_mynz
                    K[mz, ny] += stiff_mzny
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


uxyz = linalg.cg(csr_matrix(K), f, tol=1e-8)[0]
ux, uy, uz = uxyz[0:N], uxyz[N: 2*N], uxyz[2*N: 3*N]
u = np.array([ux, uy, uz]).T
xb = boundary_vertices
x = interior_vertices + u

fig = plt.figure()
fig.suptitle('Linear Elasticity Using FEM in 3D')
axes = fig.subplots(1, 3, sharey=True)
axes[0].set_aspect('equal')
axes[1].set_aspect('equal')
axes[2].set_aspect('equal')
axes[0].scatter(xb.T[1], xb.T[2], alpha=0.5, color='black')
axes[0].scatter(x.T[1], x.T[2], alpha=0.5)
axes[0].set_xlabel('yz-plane')
axes[1].scatter(xb.T[0], xb.T[2], alpha=0.5, color='black')
axes[1].scatter(x.T[0], x.T[2], alpha=0.5)
axes[1].set_xlabel('xz-plane')
axes[2].scatter(xb.T[0], xb.T[1], alpha=0.5, color='black')
axes[2].scatter(x.T[0], x.T[1], alpha=0.5)
axes[2].set_xlabel('xy-plane')
plt.show()
plt.close()

fig = plt.figure()
fig.suptitle('Linear Elasticity Using FEM in 3D')
ax = fig.add_subplot(111, projection='3d')
# ax.set_aspect('equal')
ax.scatter(x.T[0], x.T[1], x.T[2], alpha=1.0)
ax.scatter(xb.T[0], xb.T[1], xb.T[2], alpha=1.0, color='black')
plt.show()
plt.close()


