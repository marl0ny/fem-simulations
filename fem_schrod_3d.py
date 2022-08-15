"""
Numerically solve the time independent Schrodinger equation in 3D using
a finite element discretization.

Linear basis functions on tetrahedral elements using natural coordinates
are used. The polynomial integration formula is found here:
 - T.J. Chung, "Finite Element Interpolation Functions", 
   in Computational Fluid Dynamics, 2nd ed, CUP, 2010, 
   ch 9, pp. 262-308.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix, dok_matrix
from read_m_mesh import get_vertices_and_edges
import matplotlib.pyplot as plt

HBAR = 1.0
M_E = 1.0

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


def grad_basis_func_product(phi1, phi2, element_vertices):
    x, y, z = 0, 1, 2
    x0, y0, z0 = element_vertices[0][x], element_vertices[0][y], \
        element_vertices[0][z]
    x1, y1, z1 = element_vertices[1][x], element_vertices[1][y], \
        element_vertices[1][z]
    x2, y2, z2 = element_vertices[2][x], element_vertices[2][y], \
        element_vertices[2][z]
    x3, y3, z3 = element_vertices[3][x], element_vertices[3][y], \
        element_vertices[3][z]
    vol_6 = -(x0*y1*z2 - x0*y1*z3 - x0*y2*z1 + x0*y2*z3 + x0*y3*z1
             - x0*y3*z2 - x1*y0*z2 + x1*y0*z3 + x1*y2*z0 - x1*y2*z3
             - x1*y3*z0 + x1*y3*z2 + x2*y0*z1 - x2*y0*z3 - x2*y1*z0
             + x2*y1*z3 + x2*y3*z0 - x2*y3*z1 - x3*y0*z1 + x3*y0*z2
             + x3*y1*z0 - x3*y1*z2 - x3*y2*z0 + x3*y2*z1)
    a = -np.array([(y1*z2 - y1*z3 - y2*z1 + y2*z3 + y3*z1 - y3*z2)/vol_6,
                  (-y0*z2 + y0*z3 + y2*z0 - y2*z3 - y3*z0 + y3*z2)/vol_6,
                  (y0*z1 - y0*z3 - y1*z0 + y1*z3 + y3*z0 - y3*z1)/vol_6,
                  (-y0*z1 + y0*z2 + y1*z0 - y1*z2 - y2*z0 + y2*z1)/vol_6])
    b = -np.array([(-x1*z2 + x1*z3 + x2*z1 - x2*z3 - x3*z1 + x3*z2)/vol_6,
                  (x0*z2 - x0*z3 - x2*z0 + x2*z3 + x3*z0 - x3*z2)/vol_6,
                  (-x0*z1 + x0*z3 + x1*z0 - x1*z3 - x3*z0 + x3*z1)/vol_6,
                  (x0*z1 - x0*z2 - x1*z0 + x1*z2 + x2*z0 - x2*z1)/vol_6])
    c = -np.array([(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2)/vol_6,
                  (-x0*y2 + x0*y3 + x2*y0 - x2*y3 - x3*y0 + x3*y2)/vol_6,
                  (x0*y1 - x0*y3 - x1*y0 + x1*y3 + x3*y0 - x3*y1)/vol_6,
                  (-x0*y1 + x0*y2 + x1*y0 - x1*y2 - x2*y0 + x2*y1)/vol_6])
    stiffness = (vol_6/6.0)*(np.outer(a, a) + np.outer(b, b) + np.outer(c, c))
    return phi1 @ stiffness @ phi2


def get_potential_matrix(potential_vals):
    mat = np.zeros([4, 4])
    for i in range(4):
        v_i = potential_vals[i]
        for j in range(0, i + 1):
            if i != j:
                v_j = potential_vals[j]
                k_neq_ij = [p for p in range(4) if p != i and p != j]
                mat[i, j] = 2.0*v_i + 2.0*v_j + \
                    potential_vals[k_neq_ij[0]] + potential_vals[k_neq_ij[1]]
                mat[j, i] += mat[i, j]
            else:
                k_neq_i = [p for p in range(4) if p != i]
                mat[i, i] = 2.0*(potential_vals[k_neq_i[0]] +
                                 potential_vals[k_neq_i[1]] +
                                 potential_vals[k_neq_i[2]] + 3.0*v_i)

    return mat



vertices, elements = get_vertices_and_edges('./data/sphere.txt')
# elements_set = set()
# for e in elements:
#     if tuple(e) in elements_set:
#         print(e)
#     elements_set.add(tuple(e))
# vertices = vertices[1::]
# index_set = set()
# for e in elements:
#     for i in e:
#         index_set.add(i)
# for i in range(1, len(vertices)+1):
#     if not i in index_set:
#         print(i)
# import sys; sys.exit();
vertices_index = np.arange(1, vertices.shape[0]+1)
dist = np.zeros([vertices.shape[0]])
for i in range(vertices.shape[0]):
    dist[i] = np.linalg.norm(vertices[i])
interior_vertices = np.array([vertices[i] for i in range(1, vertices.shape[0])
                              if dist[i] < 1.0 - 1e-10])
interior_vertices_index = np.array([vertices_index[i]
                                    for i in range(1, len(vertices))
                                    if dist[i] < 1.0 - 1e-10])
to_interior_indices = {interior_vertices_index[i]: i
                       for i, v in enumerate(interior_vertices)}
to_all_indices = {i: interior_vertices_index[i]
                  for i, v in enumerate(interior_vertices)}
to_interior_indices_keys = to_interior_indices.keys()
OMEGA = 0.0
V = np.array([(OMEGA**2*M_E/2.0)*(vertices[i, 1]**2 + vertices[i, 2]**2
                                  + vertices[i, 0]**2)
              for i in range(vertices.shape[0])])


def in_interior(index):
    return index in to_interior_indices_keys


N = interior_vertices.shape[0]
print(N)
T = dok_matrix((N, N))
M = dok_matrix((N, N))
U = dok_matrix((N, N))

for k, element in enumerate(elements):
    element_vertices = [vertices[int(element[0])-1],
                        vertices[int(element[1])-1],
                        vertices[int(element[2])-1],
                        vertices[int(element[3])-1]]
    elem_int = get_volume_of_element(element_vertices)/120.0
    potential_values = (V[int(element[1])-1], V[int(element[2])-1],
                        V[int(element[3])-1], V[int(element[0])-1])
    potential_matrix = get_potential_matrix(potential_values)
    f = 1.0
    # if elem_int < 0.0:
    #     f = -1.0
    #     elem_int *= f
    # if elem_int < 0.0:
    #     elem_int *= -1.0
    #     element_vertices = [element_vertices[1], element_vertices[0],
    #                         element_vertices[2], element_vertices[3]]
    for i in range(len(element_vertices)):
        v_i = element_vertices[i]
        phi1 = np.array([1.0 if i == p else 0.0 for p in range(4)])
        for j in range(0, i+1):
            v_j = element_vertices[j]
            phi2 = np.array([1.0 if j == p else 0.0 for p in range(4)])
            if in_interior(element[i]) and in_interior(element[j]):
                potential_val = elem_int*(phi1 @ potential_matrix @ phi2)
                grad_elem_int = f*grad_basis_func_product(phi1, phi2,
                                                          element_vertices)
                m = to_interior_indices[element[i]]
                n = to_interior_indices[element[j]]
                M[m, n] += 6.0*elem_int
                T[m, n] += (0.5*HBAR**2/M_E)*grad_elem_int
                U[m, n] += potential_val
                M[n, m] += 6.0*elem_int
                if n != m:
                    T[n, m] += (0.5*HBAR**2/M_E)*grad_elem_int
                    U[n, m] += potential_val

n_states = 7
n_state = 6
# plt.imshow(csr_matrix(M)[0:100, 0:100].toarray()); plt.show(); plt.close()
# import sys; sys.exit()
# print(min(M), max(M))
eigvals, eigvects = eigsh(csr_matrix(T + U), M=csr_matrix(M), k=n_states,
                          which='LM', sigma=0.0)
print(eigvals)
eigvect = eigvects.T[n_state]
max_val = np.amax(np.abs(eigvect)**2)
colours = np.zeros([vertices.shape[0], 4])
for i in range(colours.shape[0]):
    if i+1 in to_interior_indices_keys:
        val = eigvect[to_interior_indices[i+1]]
        r = 1.0 if val > 0.0 else 0.0
        b = 1.0 if val <= 0.0 else 0.0
        a = val**2/max_val
        colours[i, 0: 4] = np.array([r, 0.0, b, a])


plt.style.use('dark_background')

fig = plt.figure()
fig.suptitle(f'Stationary State (n={n_state})')
ax = fig.add_subplot(111, projection='3d')
# ax.set_aspect('equal')
ax.scatter(vertices.T[0], vertices.T[1], vertices.T[2], 
           color=colours)
plt.show()
plt.close()

fig = plt.figure()
fig.suptitle(f'Stationary State (n={n_state})')
axes = fig.subplots(1, 3, sharey=True)
axes[0].set_aspect('equal')
axes[1].set_aspect('equal')
axes[2].set_aspect('equal')
axes[0].scatter(vertices.T[1], vertices.T[2],
                color=colours)
axes[0].set_xlabel('yz-plane')
axes[1].scatter(vertices.T[2], vertices.T[0],
                color=colours)
axes[1].set_xlabel('zx-plane')
axes[2].scatter(vertices.T[0], vertices.T[1],
                color=colours)
axes[2].set_xlabel('xy-plane')
plt.show()
plt.close()



