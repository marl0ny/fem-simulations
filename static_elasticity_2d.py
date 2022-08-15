"""
Static linear elasticity using the finite element method in 2D.

Plenty of bookkeeping is involved in writing this script; it is
possible some errors still remain.

Reference:
T. Jos, "The Finite Element Method for Partial Differential Equations," 
in Computational Physics, 2nd ed, CUP, 2013, ch 13, pp. 423 - 447. 

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import linalg
from scipy.sparse import csr_matrix, dok_matrix


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


def get_stiffness_matrix(element_vertices):
    D = get_deformation_matrix(element_vertices)
    area = get_area_of_element(element_vertices)
    return area*(D.T @ C @ D)


def get_force_on_element(f, element_vertices):
    G = np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    area = get_area_of_element(element_vertices)
    return area*f.T @ G /3.0


vertices_array = np.loadtxt('./data/rectangle.node', skiprows=1)
elements_array = np.loadtxt('./data/rectangle.1.ele', skiprows=1)
boundary_vertices = np.array([v for v in vertices_array if v[3] > 0.0])
interior_vertices = np.array([v for v in vertices_array if v[3] < 1.0])
# print(interior_vertices.shape)
to_all_indices = {i: int(v[0]) for i, v in enumerate(interior_vertices)}
to_interiors_indices = {int(v[0]): i for i, v in enumerate(interior_vertices)}
# print(interior_vertices)
N = len(interior_vertices)

K = dok_matrix((2*N, 2*N))
f = np.zeros([2*N])
Fg = -0.0005

for k in elements_array.T[0]:
    element = elements_array[int(k)-1]
    element_vertices = (vertices_array[int(element[1])-1],
                        vertices_array[int(element[2])-1],
                        vertices_array[int(element[3])-1])
    for i in range(3):
        v_i = element_vertices[i]
        if v_i[3] < 1.0:
            stiff_mat = get_stiffness_matrix(element_vertices)
            m = to_interiors_indices[v_i[0]]
            mx, my = m, m + N
            fg_vec = np.array([0.0, Fg])
            fe = get_force_on_element(fg_vec, element_vertices)
            f[mx] += fe[i]
            f[my] += fe[i + 3]
            for j in range(0, i+1):
                v_j = element_vertices[j]
                if v_j[3] < 1.0:
                    stiffness_mxnx = stiff_mat[i, j]
                    stiffness_mxny = stiff_mat[i, j + 3]
                    stiffness_mynx = stiff_mat[i + 3, j]
                    stiffness_myny = stiff_mat[i + 3, j + 3]
                    n = to_interiors_indices[v_j[0]]
                    nx, ny = n, n + N
                    K[mx, nx] += stiffness_mxnx
                    K[mx, ny] += stiffness_mxny
                    K[my, nx] += stiffness_mynx
                    K[my, ny] += stiffness_myny
                    if m != n:
                        K[nx, mx] += stiffness_mxnx
                        K[ny, mx] += stiffness_mxny
                        K[nx, my] += stiffness_mynx
                        K[ny, my] += stiffness_myny


x0 = interior_vertices.T[1]
y0 = interior_vertices.T[2]
xb = boundary_vertices.T[1]
yb = boundary_vertices.T[2]
uxy = linalg.cg(csr_matrix(K), f, tol=1e-8)[0]
ux, uy = uxy[0:N], uxy[N:2*N]
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
ax.set_title('Linear Elasticity Using FEM in 2D')
ax.scatter(xb, yb, color='black')
dat = ax.scatter(x0 + ux, y0 + uy)
dat.set(offsets=np.array([x0 + ux, y0 + uy]).T)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
plt.close()
print(x0.shape, ux.shape)

