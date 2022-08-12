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
DT = 20.0
# Density (Assume each element has the same mass)
RHO = 1.0
# Dissipation
# GAMMA = 0.0
# GAMMA = 0.0012
# GAMMA = 0.0007
GAMMA = 0.0004


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
plot_data = ax.scatter(x0, y0)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_ylim(-15.0, 15.0)
ax.set_xlim(-1.0, 35.0)


def animation_func(*args):
    # M*(u2 - 2*u1 + u0)/DT^2 = -K * (u0 + u2)/2 + f - GAMMA*(u2 - u0)/(2*DT)
    # M*(u2 - 2*u1 + u0) = DT^2*(-K*(u0 + u2)/2 + f - GAMMA*(u2 - u0)/(2*DT))
    # (M + DT^2*K/2 + DT*GAMMA/2)*u2 = -DT^2*K*u0/2 + DT^2*f + DT*GAMMA*u0/2
    #                                  + 2*M*u1 - M*u0
    ux, uy = np.zeros([N]), np.zeros([N])
    for _ in range(1):
        u0, u1 = frames[0], frames[1]
        # f2 = (-DT**2*K + 2.0*M) @ u1 + (-M + 0.5*GAMMA*DT*M) @ u0 + DT**2*f
        # uxy = linalg.cg(M + 0.5*GAMMA*DT*M, f2)[0]
        f2 = (DT**2*(-0.5*K @ u0 + f) + 0.5*DT*GAMMA*M @ u0
            + 2.0*M @ u1 - M @ u0)
        uxy = linalg.cg(M + DT**2*0.5*K + 0.5*DT*M*GAMMA, f2)[0]
        # f2 = ((-DT**2*K/3.0 + DT*GAMMA*M/2.0 - M) @ u0
        #       + (-DT**2*K/3.0 + 2.0*M) @ u1 + DT**2*f)
        # uxy = linalg.cg(M + DT**2*K/2.0 + DT*GAMMA*M/2.0, f2)[0]
        ux[0: N], uy[0: N] = uxy[0: N], uxy[N: 2*N]
        frames[0], frames[1] = u1, uxy
    plot_data.set(offsets=np.array([x0 + ux, y0 + uy]).T)
    return plot_data,


anim = animation.FuncAnimation(fig, animation_func,
                               blit=True, interval=1.0)
plt.show()
plt.close()


