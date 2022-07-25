from sympy import Symbol, Integral
from sympy.abc import x


x0 = Symbol('x0')
x1 = Symbol('x1')
x2 = Symbol('x2')

phi2_int1 = Integral((x - x0)**2/(x1 - x0)**2, (x, x0, x1))
phi2_int2 = Integral((x2 - x)**2/(x2 - x1)**2, (x, x1, x2))
phi2_int = phi2_int1.simplify() + phi2_int2.simplify()
print(phi2_int)

phi3_int1 = Integral((x - x0)**3/(x1 - x0)**2, (x, x0, x1))
phi3_int2 = Integral((x2 - x)**3/(x2 - x1)**2, (x, x1, x2))
phi3_int = phi3_int1.simplify() + phi3_int2.simplify()
print(phi3_int)

phi_12_phi_2_int = Integral((x2 - x)**2*(x - x1)/(x2 - x1)**3, (x, x1, x2))
print(phi_12_phi_2_int.simplify())

phi_0_phi_12_int = Integral((x1 - x)*(x - x0)**2/(x1 - x0)**3, (x, x0, x1))
print(phi_0_phi_12_int.simplify())

