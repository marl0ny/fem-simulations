"""
Numerically solve the time-independent Schrodinger equation in 2D using
a finite element discretization

Reference:
 - T. Jos, "The Finite Element Method for Partial Differential Equations," 
   in Computational Physics, 2nd ed, CUP, 2013, ch 13, pp. 423 - 447. 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix, dok_matrix


HBAR = 1.0
M_E = 1.0


def get_area_of_element(element_vertices):
    x, y = 1, 2
    x10 = element_vertices[0][x] - element_vertices[1][x]
    x20 = element_vertices[0][x] - element_vertices[2][x]
    y20 = element_vertices[0][y] - element_vertices[2][y]
    y10 = element_vertices[0][y] - element_vertices[1][y]
    return (x10*y20 - x20*y10)/2.0


def get_stiffness_matrix(element_vertices):
    x, y = 1, 2
    area = get_area_of_element(element_vertices)
    x21 = -element_vertices[2][x] + element_vertices[1][x]
    x02 = -element_vertices[0][x] + element_vertices[2][x]
    x10 = -element_vertices[1][x] + element_vertices[0][x]
    y12 = -element_vertices[1][y] + element_vertices[2][y]
    y20 = -element_vertices[2][y] + element_vertices[0][y]
    y01 = -element_vertices[0][y] + element_vertices[1][y]
    coeffs1 = np.array([x21, x02, x10])
    coeffs2 = np.array([y12, y20, y01])
    stiffness = np.outer(coeffs1, coeffs1) + np.outer(coeffs2, coeffs2)
    return 0.25*stiffness/area


def get_potential_matrix(potential_vals, element_vertices):
    area = get_area_of_element(element_vertices)
    mat = np.zeros([3, 3])
    for i in range(3):
        v_i = potential_vals[i]
        for j in range(0, i+1):
            if i != j:
                v_j = potential_vals[j]
                k = [p for p in range(3) if p != i and p != j][0]
                v_k = potential_vals[k]
                mat[i, j] = 2.0*v_i + 2.0*v_j + v_k
                mat[j, i] = mat[i, j]
            else:
                km = [p for p in range(3) if p != i]
                v_k, v_m = potential_vals[km[0]], potential_vals[km[1]]
                mat[i, i] = 2.0*(v_k + v_m + 3.0*v_i)
    return mat*area/60.0


vertices_array = np.loadtxt('./data/circle2.1.node', skiprows=1)
elements_array = np.loadtxt('./data/circle2.1.ele', skiprows=1)
interior_vertices = np.array([v for v in vertices_array if v[3] < 1.0])
to_all_indices = {i: int(v[0]) for i, v in enumerate(interior_vertices)}
to_interiors_indices = {int(v[0]): i for i, v in enumerate(interior_vertices)}
N = len(interior_vertices)
print(N)
OMEGA = 0.0
V = np.array([(OMEGA**2*M_E/2.0)*
              (vertices_array[i, 1]**2 + vertices_array[i, 2]**2)
              for i in range(vertices_array.shape[0])])


T = dok_matrix((N, N))
U = dok_matrix((N, N))
M = dok_matrix((N, N))
for k in elements_array.T[0]:
    element = elements_array[int(k)-1]
    element_vertices = (vertices_array[int(element[1])-1],
                        vertices_array[int(element[2])-1],
                        vertices_array[int(element[3])-1])
    potential_values = (V[int(element[1])-1], V[int(element[2])-1],
                        V[int(element[3])-1])
    area = get_area_of_element(element_vertices)
    # if area == 0.0:
    #     print(element)
    potential_matrix = get_potential_matrix(potential_values, element_vertices)
    stiffness_matrix = get_stiffness_matrix(element_vertices)
    for i in range(len(element_vertices)):
        v_i = element_vertices[i]
        for j in range(0, i+1):
            v_j = element_vertices[j]
            if v_i[3] < 1.0 and v_j[3] < 1.0:
                potential_val = potential_matrix[i, j]
                stiffness_val = stiffness_matrix[i, j]
                k = to_interiors_indices[v_i[0]]
                l = to_interiors_indices[v_j[0]]
                M[k, l] += area/12.0
                T[k, l] += (0.5*HBAR**2/M_E)*stiffness_val
                U[k, l] += potential_val
                M[l, k] += area/12.0
                if k != l:
                    T[l, k] += (0.5*HBAR**2/M_E)*stiffness_val
                    U[l, k] += potential_val


n_states = 15
n_state = 3
# print(M.toarray().shape)
eigvals, eigvects = eigsh(csr_matrix(T + U), M=csr_matrix(M),
                          k=n_states, which='LM', sigma=0.0)
print(eigvals)
eigvect = eigvects.T[n_state]
x, y = vertices_array.T[1], vertices_array.T[2]
c = np.zeros([len(x)])
for i in range(len(c)):
    index = vertices_array[i, 0]
    if int(index) - 1 in to_interiors_indices.keys():
        val = eigvect[to_interiors_indices[int(index) - 1]]
        c[i] = val/np.amax(np.abs(val))
plt.style.use('dark_background')
for k in elements_array.T[0]:
    element = elements_array[int(k)-1]
    arr = np.array([vertices_array[int(element[1])-1],
                    vertices_array[int(element[2])-1],
                    vertices_array[int(element[3])-1]])
    phi = 0.0
    if int(element[1]) in to_interiors_indices.keys():
        phi += eigvect[to_interiors_indices[int(element[1])]]
    if int(element[2]) in to_interiors_indices.keys():
        phi += eigvect[to_interiors_indices[int(element[2])]]
    if int(element[3]) in to_interiors_indices.keys():
        phi += eigvect[to_interiors_indices[int(element[3])]]
    abs_phi = np.abs(phi)/3.0
    plt.plot(arr.T[1], arr.T[2], linewidth=0.25, alpha=0.5, color='grey')
    plt.fill(arr.T[1], arr.T[2],
             linewidth=0.25, alpha=1.0 if abs_phi > 1.0 else abs_phi,
             color='purple' if phi > 0.0 else 'yellow')

plt.scatter(interior_vertices.T[1], interior_vertices.T[2],
            alpha=np.abs(eigvect)/np.amax(np.abs(eigvect)),
            color='white', 
            s=50.0*np.abs(eigvect)/np.amax(np.abs(eigvect)))
plt.title(f'Stationary State of Circular Well Using FEM (n = {n_state})')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.close()

