"""
Get the transformation matrix for 3D Cartesian to
natural tetrahedral coordinates and vice versa.
"""
from sympy import Symbol, Matrix


x0, x1, x2, x3 = Symbol('x0'), Symbol('x1'), Symbol('x2'), Symbol('x3')
y0, y1, y2, y3 = Symbol('y0'), Symbol('y1'), Symbol('y2'), Symbol('y3')
z0, z1, z2, z3 = Symbol('z0'), Symbol('z1'), Symbol('z2'), Symbol('z3')


# Transformation matrix to go from natural tetrahedral coordinates
# to Cartesian.
matrix = Matrix([[1,  1,  1,  1],
                 [x0, x1, x2, x3],
                 [y0, y1, y2, y3],
                 [z0, z1, z2, z3]])
ndim = 4

# print(inv_matrix := matrix.inv())

v_6 = -(x0*y1*z2 - x0*y1*z3 - x0*y2*z1 + x0*y2*z3 + x0*y3*z1
        - x0*y3*z2 - x1*y0*z2 + x1*y0*z3 + x1*y2*z0 - x1*y2*z3
        - x1*y3*z0 + x1*y3*z2 + x2*y0*z1 - x2*y0*z3 - x2*y1*z0
        + x2*y1*z3 + x2*y3*z0 - x2*y3*z1 - x3*y0*z1 + x3*y0*z2
        + x3*y1*z0 - x3*y1*z2 - x3*y2*z0 + x3*y2*z1)

# inv_matrix = inv_matrix.subs(v_6, 'v_6').expand()
# inv_matrix.simplify()
# print(inv_matrix)
# The above lines when uncommented out produce:
inv_matrix = Matrix([[(x1*y2*z3 - x1*y3*z2 - x2*y1*z3
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
inv_matrix = inv_matrix.subs(v_6, 'v_6').expand()
inv_matrix.simplify()
print(inv_matrix.T[ndim*2:ndim*3])

