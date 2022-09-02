"""
Numerically solve the time-independent Schrodinger equation in 2D using
a finite element discretization. Quadratic basis functions instead of linear
are used.

Reference:
 - T. Jos, "The Finite Element Method for Partial Differential Equations," 
   in Computational Physics, 2nd ed, CUP, 2013, ch 13, pp. 423 - 447. 
 - T.J. Chung, "Finite Element Interpolation Functions", 
   in Computational Fluid Dynamics, 2nd ed, CUP, 2010, 
   ch 9, pp. 262-308.
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


def get_mass_matrix(element_vertices):
    area = get_area_of_element(element_vertices)
    mat = np.zeros([6, 6])
    mat[0, 0] = 1/60
    mat[0, 1] = -1/360
    mat[0, 2] = -1/360
    mat[0, 3] = 0
    mat[0, 4] = -1/90
    mat[0, 5] = 0
    mat[1, 0] = mat[0, 1]
    mat[1, 1] = 1/60
    mat[1, 2] = -1/360
    mat[1, 3] = 0
    mat[1, 4] = 0
    mat[1, 5] = -1/90
    mat[2, 0] = mat[0, 2]
    mat[2, 1] = mat[1, 2]
    mat[2, 2] = 1/60
    mat[2, 3] = -1/90
    mat[2, 4] = 0
    mat[2, 5] = 0
    mat[3, 0] = mat[0, 3]
    mat[3, 1] = mat[1, 3]
    mat[3, 2] = mat[2, 3]
    mat[3, 3] = 4/45
    mat[3, 4] = 2/45
    mat[3, 5] = 2/45
    mat[4, 0] = mat[0, 4]
    mat[4, 1] = mat[1, 4]
    mat[4, 2] = mat[2, 4]
    mat[4, 3] = mat[3, 4]
    mat[4, 4] = 4/45
    mat[4, 5] = 2/45
    mat[5, 0] = mat[0, 5]
    mat[5, 1] = mat[1, 5]
    mat[5, 2] = mat[2, 5]
    mat[5, 3] = mat[3, 5]
    mat[5, 4] = mat[4, 5]
    mat[5, 5] = 4/45
    return 2.0*area*mat


def get_stiffness_matrix(element_vertices):
    x, y = 1, 2
    x21 = -element_vertices[2][x] + element_vertices[1][x]
    x02 = -element_vertices[0][x] + element_vertices[2][x]
    x10 = -element_vertices[1][x] + element_vertices[0][x]
    y12 = -element_vertices[1][y] + element_vertices[2][y]
    y20 = -element_vertices[2][y] + element_vertices[0][y]
    y01 = -element_vertices[0][y] + element_vertices[1][y]
    area = get_area_of_element(element_vertices)
    # area = (x10*y20 - x20*y10)/2.0
    dd0 = np.array([x21, y12])
    dd1 = np.array([x02, y20])
    dd2 = np.array([x10, y01])
    mat = np.zeros([6, 6])
    for d0, d1, d2 in zip(dd0, dd1, dd2):
        mat[0, 0] += d0**2/2
        mat[0, 1] += -d0*d1/6
        mat[0, 2] += -d0*d2/6
        mat[0, 3] += 2*d0*d1/3
        mat[0, 4] += 0
        mat[0, 5] += 2*d0*d2/3
        mat[1, 0] += -d0*d1/6
        mat[1, 1] += d1**2/2
        mat[1, 2] += -d1*d2/6
        mat[1, 3] += 2*d0*d1/3
        mat[1, 4] += 2*d1*d2/3
        mat[1, 5] += 0
        mat[2, 0] += -d0*d2/6
        mat[2, 1] += -d1*d2/6
        mat[2, 2] += d2**2/2
        mat[2, 3] += 0
        mat[2, 4] += 2*d1*d2/3
        mat[2, 5] += 2*d0*d2/3
        mat[3, 0] += 2*d0*d1/3
        mat[3, 1] += 2*d0*d1/3
        mat[3, 2] += 0
        mat[3, 3] += 4*d0**2/3 + 4*d0*d1/3 + 4*d1**2/3
        mat[3, 4] += 2*d0*d1/3 + 4*d0*d2/3 + 2*d1**2/3 + 2*d1*d2/3
        mat[3, 5] += 2*d0**2/3 + 2*d0*d1/3 + 2*d0*d2/3 + 4*d1*d2/3
        mat[4, 0] += 0
        mat[4, 1] += 2*d1*d2/3
        mat[4, 2] += 2*d1*d2/3
        mat[4, 3] += 2*d0*d1/3 + 4*d0*d2/3 + 2*d1**2/3 + 2*d1*d2/3
        mat[4, 4] += 4*d1**2/3 + 4*d1*d2/3 + 4*d2**2/3
        mat[4, 5] += 4*d0*d1/3 + 2*d0*d2/3 + 2*d1*d2/3 + 2*d2**2/3
        mat[5, 0] += 2*d0*d2/3
        mat[5, 1] += 0
        mat[5, 2] += 2*d0*d2/3
        mat[5, 3] += 2*d0**2/3 + 2*d0*d1/3 + 2*d0*d2/3 + 4*d1*d2/3
        mat[5, 4] += 4*d0*d1/3 + 2*d0*d2/3 + 2*d1*d2/3 + 2*d2**2/3
        mat[5, 5] += 4*d0**2/3 + 4*d0*d2/3 + 4*d2**2/3
    stiffness = 0.5*mat/area
    return stiffness


def get_potential_matrix(potential_vals: np.ndarray, 
                         element_vertices):
    area = get_area_of_element(element_vertices)
    k_coeffs = 2.0*area*potential_vals
    if len(k_coeffs) != 6:
        return np.zeros([len(k_coeffs), len(k_coeffs)])
    k0, k1, k2 ,k3, k4, k5 = list(k_coeffs)
    mat = np.zeros([6, 6])
    mat[0, 0] = (0.00714285714285717*k0 - 0.000793650793650791*k1 
        - 0.000793650793650791*k2 + 0.00476190476190475*k3 
        + 0.00158730158730159*k4 + 0.00476190476190475*k5)
    mat[1, 0] = (-0.000793650793650791*k0 - 0.000793650793650791*k1 
        + 0.000396825396825399*k2 - 0.00158730158730159*k3 
        + 1.73472347597681e-18*k4 + 1.73472347597681e-18*k5)
    mat[0, 1] = mat[1, 0]
    mat[1, 1] = (-0.000793650793650791*k0 + 0.00714285714285717*k1 
        - 0.000793650793650791*k2 + 0.00476190476190475*k3 
        + 0.00476190476190475*k4 + 0.00158730158730159*k5)
    mat[2, 0] = (-0.000793650793650791*k0 + 0.000396825396825399*k1 
        - 0.000793650793650791*k2 + 1.73472347597681e-18*k3 
        + 1.73472347597681e-18*k4 - 0.00158730158730159*k5)
    mat[0, 2] = mat[2, 0]
    mat[2, 1] = (0.000396825396825399*k0 - 0.000793650793650791*k1 
        - 0.000793650793650791*k2 + 1.73472347597681e-18*k3 
        - 0.00158730158730159*k4 + 1.73472347597681e-18*k5)
    mat[1, 2] = mat[2, 1]
    mat[2, 2] = (-0.000793650793650791*k0 - 0.000793650793650791*k1 
        + 0.00714285714285717*k2 + 0.00158730158730159*k3 
        + 0.00476190476190475*k4 + 0.00476190476190475*k5)
    mat[3, 0] = (0.00476190476190475*k0 - 0.00158730158730159*k1 
        + 1.73472347597681e-18*k2 - 0.00317460317460317*k4)
    mat[0, 3] = mat[3, 0]
    mat[3, 1] = (-0.00158730158730159*k0 + 0.00476190476190475*k1 
        + 1.73472347597681e-18*k2 - 0.00317460317460317*k5)
    mat[1, 3] = mat[3, 1]
    mat[3, 2] = (1.73472347597681e-18*k0 + 1.73472347597681e-18*k1 
        + 0.00158730158730159*k2 - 0.00634920634920635*k3 
        - 0.00317460317460317*k4 - 0.00317460317460317*k5)
    mat[2, 3] = mat[3, 2]
    mat[3, 3] = (-0.00634920634920635*k2 + 0.0571428571428571*k3 
        + 0.019047619047619*k4 + 0.019047619047619*k5)
    mat[4, 0] = (0.00158730158730159*k0 + 1.73472347597681e-18*k1 
        + 1.73472347597681e-18*k2 - 0.00317460317460317*k3 
        - 0.00634920634920635*k4 - 0.00317460317460317*k5)
    mat[0, 4] = mat[4, 0]
    mat[4, 1] = (1.73472347597681e-18*k0 + 0.00476190476190475*k1 
        - 0.00158730158730159*k2 - 0.00317460317460317*k5)
    mat[1, 4] = mat[4, 1]
    mat[4, 2] = (1.73472347597681e-18*k0 - 0.00158730158730159*k1 
        + 0.00476190476190475*k2 - 0.00317460317460317*k3)
    mat[2, 4] = mat[4, 2]
    mat[4, 3] = (-0.00317460317460317*k0 - 0.00317460317460317*k2 
        + 0.019047619047619*k3 + 0.019047619047619*k4 + 0.0126984126984127*k5)
    mat[3, 4] = mat[4, 3]
    mat[4, 4] = (-0.00634920634920635*k0 + 0.019047619047619*k3 
        + 0.0571428571428571*k4 + 0.019047619047619*k5)
    mat[5, 0] = (0.00476190476190475*k0 + 1.73472347597681e-18*k1 
        - 0.00158730158730159*k2 - 0.00317460317460317*k4)
    mat[0, 5] = mat[5, 0]
    mat[5, 1] = (1.73472347597681e-18*k0 + 0.00158730158730159*k1 
        + 1.73472347597681e-18*k2 - 0.00317460317460317*k3 
        - 0.00317460317460317*k4 - 0.00634920634920635*k5)
    mat[1, 5] = mat[5, 1]
    mat[5, 2] = (-0.00158730158730159*k0 + 1.73472347597681e-18*k1 
        + 0.00476190476190475*k2 - 0.00317460317460317*k3)
    mat[2, 5] = mat[5, 2]
    mat[5, 3] = (-0.00317460317460317*k1 - 0.00317460317460317*k2 
        + 0.019047619047619*k3 + 0.0126984126984127*k4 + 0.019047619047619*k5)
    mat[3, 5] = mat[5, 3]
    mat[5, 4] = (-0.00317460317460317*k0 - 0.00317460317460317*k1 
        + 0.0126984126984127*k3 + 0.019047619047619*k4 + 0.019047619047619*k5)
    mat[4, 5] = mat[5, 4]
    mat[5, 5] = (-0.00634920634920635*k1 + 0.019047619047619*k3 
        + 0.019047619047619*k4 + 0.0571428571428571*k5)
    return mat


def get_arrays_with_new_edge_points(vertices_array, elements_array):
    new_vertices_array = list(vertices_array)
    new_elements_array = []
    visited_edges = set()
    edge_to_its_vertex_dict = dict()
    for n in range(elements_array.shape[0]):
        element = elements_array[n]
        element_vertices = element[1:len(element)]
        new_element = list(element)
        for i, j in zip([0, 1, 2], [1, 2, 0]):
            e1 = element_vertices[i]
            e2 = element_vertices[j]
            e2_ge_e1 = True if e2 > e1 else False
            visited_edge = (frozenset((int(e1), int(e2))) if e2_ge_e1 else
                            frozenset((int(e2), int(e1))))
            if not (visited_edge in visited_edges):
                v1 = vertices_array[int(e1) - 1]
                v2 = vertices_array[int(e2) - 1]
                v12 = (v2 + v1)/2.0
                index = len(new_vertices_array) + 1
                v12[0] = index
                if (vertices_array[int(e1) - 1][3] < 1.0
                    or vertices_array[int(e2) - 1][3] < 1.0):
                    v12[3] = 0.0
                else:
                    v12[3] = 1.0
                new_vertices_array.append(v12)
                new_element.append(index)
                visited_edges.add(visited_edge)
                edge_to_its_vertex_dict[visited_edge] = index
            else:
                new_element.append(
                    edge_to_its_vertex_dict[visited_edge])
        new_elements_array.append(new_element)
    return np.array(new_vertices_array), np.array(new_elements_array)


vertices_array = np.loadtxt('./data/circle2.1.node', skiprows=1)
elements_array = np.loadtxt('./data/circle2.1.ele', skiprows=1)
# vertices_array = np.loadtxt('./data/rectangle.1.node', skiprows=1)
# elements_array = np.loadtxt('./data/rectangle.1.ele', skiprows=1)
print(elements_array.shape)
vertices_array2, elements_array2 = get_arrays_with_new_edge_points(
    vertices_array, elements_array)

# print(elements_array2)

# print(len(vertices_array), len(vertices_array2))
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
ax.scatter(vertices_array.T[1], vertices_array.T[2], 
           color='black', s=2.0
           )
n = len(vertices_array)
m = len(vertices_array2)
ax.scatter(vertices_array2.T[1, n: m], vertices_array2.T[2, n: m], 
           s=2.0
           )
for element in elements_array:
    e0, e1, e2 = int(element[1]) - 1, int(element[2]) - 1, int(element[3]) - 1
    v0, v1, v2 = vertices_array[e0], vertices_array[e1], vertices_array[e2]
    arr = np.array([v0, v1, v2, v0])
    ax.plot(arr.T[1], arr.T[2], linewidth=0.5, color='orange')
plt.show()
plt.close()

elements_array = elements_array2
vertices_array = vertices_array2

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
    element = elements_array[int(k)-1][1:7]
    element_vertices = [vertices_array[int(element[i])-1]
                        for i in range(len(element))]
    potential_values = np.array([V[int(element[i])-1]
                                 for i in range(len(element))])
    area = get_area_of_element(element_vertices)
    # if area == 0.0:
    #     print(element)
    potential_matrix = get_potential_matrix(potential_values, 
                                           element_vertices)
    mass_matrix = get_mass_matrix(element_vertices)
    stiffness_matrix = get_stiffness_matrix(element_vertices)
    for i in range(len(element_vertices)):
        v_i = element_vertices[i]
        if v_i[3] < 1.0:
            for j in range(0, i + 1):
                v_j = element_vertices[j]
                if v_j[3] < 1.0:
                    potential_val = potential_matrix[i, j]
                    stiffness_val = stiffness_matrix[i, j]
                    k = to_interiors_indices[v_i[0]]
                    l = to_interiors_indices[v_j[0]]
                    # print(k, l)
                    # if k == l:
                    #     print(i, j, area, mass_matrix[i, j])
                    M[k, l] += mass_matrix[i, j]
                    T[k, l] += (0.5*HBAR**2/M_E)*stiffness_val
                    U[k, l] += potential_val
                    if k != l:
                        M[l, k] += mass_matrix[i, j]
                        T[l, k] += (0.5*HBAR**2/M_E)*stiffness_val
                        U[l, k] += potential_val


# print(csr_matrix(M).toarray().shape)
# plt.imshow(csr_matrix(M).toarray()[800:900, 800:900])
# plt.show()
# plt.close()
# import sys; sys.exit()
n_states = 7
n_state = 6
# print(M.toarray().shape)
eigvals, eigvects = eigsh(csr_matrix(T+U), M=csr_matrix(M),
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
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
for k in elements_array.T[0]:
    element = elements_array[int(k)-1]
    arr = np.array([vertices_array[int(element[1])-1],
                    vertices_array[int(element[2])-1],
                    vertices_array[int(element[3])-1],
                    vertices_array[int(element[1])-1],])
    phi = 0.0
    for k in range(6):
        if int(element[k+1]) in to_interiors_indices.keys():
            phi += eigvect[to_interiors_indices[int(element[k+1])]]
    abs_phi = np.abs(phi)/6.0
    ax.plot(arr[0:3, 1], arr[0:3, 2], linewidth=0.25, alpha=0.5, color='grey')
    ax.fill(arr[0:3, 1], arr[0:3, 2],
             linewidth=0.25, alpha=1.0 if abs_phi > 1.0 else abs_phi,
             color='purple' if phi > 0.0 else 'yellow')

ax.scatter(interior_vertices.T[1], interior_vertices.T[2],
           alpha=np.abs(eigvect)/np.amax(np.abs(eigvect)),
           color='white', 
           s=50.0*np.abs(eigvect)/np.amax(np.abs(eigvect)))
ax.set_title(f'Stationary State of Circular Well Using FEM (n = {n_state})')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
plt.close()
