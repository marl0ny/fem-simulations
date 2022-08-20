"""
Symbolically compute relationships between quadratic basis functions
in triangle coordinates.

Reference:
 - T.J. Chung, "Finite Element Interpolation Functions", 
   in Computational Fluid Dynamics, 2nd ed, CUP, 2010, 
   ch 9, pp. 262-308.

"""
from sympy import symbols, diff, Rational
from math import factorial
import numpy as np


a0, a1, a2 = symbols('a0 a1 a2')
x21, x02, x10 = symbols('x21 x02 x10')
y12, y20, y01 = symbols('y12 y20 y01')
d0, d1, d2  = symbols('d0 d1 d2')
vars = [a0, a1, a2]
phi0 = (2*a0 - 1)*a0
phi1 = (2*a1 - 1)*a1
phi2 = (2*a2 - 1)*a2
phi01 = 4*a0*a1
phi12 = 4*a1*a2
phi02 = 4*a0*a2
basis_funcs = [phi0, phi1, phi2, phi01, phi12, phi02]


def break_long_expressions(expr: str, max_c=70):
    """
    This assumes that the density of certain symbols for each allocated
    line is one or greater.
    """
    lines = []
    prev_i = 0
    b_point = 0
    i = 0
    while i < len(expr):
        local_i = i - prev_i
        if expr[i] in ['=', '+', '-']:
            if i > 0 and expr[i-1] != 'e':
                b_point = i
        if local_i == max_c:
            lines.append(expr[prev_i: b_point])
            prev_i = b_point
            i = b_point
        i += 1
    lines.append(expr[prev_i:len(expr)])
    return '\n    '.join(lines)


def parse_term(term):
    # print(term, term.func)
    factors = []
    powers = []
    if 'Mul' not in str(term.func):
        if 'Pow' in str(term.func):
            powers.append(int(term.args[1]))
        elif 'Symbol' in str(term.func):
            powers.append(1)
        else:
            factors.append(float(term))
    else:
        for t in term.args:
            if 'Pow' in str(t.func):
                powers.append(int(t.args[1]))
            elif 'Symbol' in str(t.func):
                powers.append(1)
            else:
                # print(t)
                factors.append(float(t))
    factor = 1.0
    for f in factors:
        factor *= f
    den = factorial(2 + sum(powers))
    num_list = [factorial(power) for power in powers]
    num = 1.0
    for e in num_list:
        num *= e
    num_den_list = [num, den]
    print(term, factor, powers, num_den_list)
    return num, den


def get_stuff_related_to_mass_matrix(basis_funcs):
    num = np.zeros([6, 6])
    den = np.zeros([6, 6])
    for i, f in enumerate(basis_funcs):
        for j, g in enumerate(basis_funcs):
            fg = (f*g).simplify().expand()
            # https://stackoverflow.com/a/45126783
            if 'Add' in str(fg.func):
                lst = []
                for additive_term in fg.args:
                    num[i, j], den[i, j] = parse_term(additive_term)
                    lst.append(additive_term.args)
                print(lst)
            else:
                num[i, j], den[i, j] = parse_term(fg)
                    # print(fg)
    return num, den


def list_a0a1a2(expr, a_list):
    for arg in expr.args:
        if 'Pow' in str(arg.func):
            if arg.args[0] in (a0, a1, a2):
                a_list.append(arg)
        elif arg in [a0, a1, a2]:
            a_list.append(arg)
        else:
            list_a0a1a2(arg, a_list)


def get_stuff_related_to_stiffness_matrix(basis_funcs):
    phi_diff = []
    for f in basis_funcs:
        # phi_a_list = [x21 + y12, x02 + y20, x10 + y01]
        phi_a_list = [d0, d1, d2]
        s = sum([diff(f, v)*phi_a_list[i] for i, v in enumerate(vars)])
        phi_diff.append(s)
        #for v in vars:
        #     print(v, diff(f, v))
        # print()
    string = ''
    for i, e1 in enumerate(phi_diff):
        for j, e2 in enumerate(phi_diff):
            expr = (e1*e2).expand()
            new_expr = 0
            if 'Add' in str(expr.func):
                for term in expr.args:
                    power_list = []
                    list_a0a1a2(term, power_list)
                    subexpr = 1
                    for power_of_a in power_list:
                        subexpr *= power_of_a
                    factors = term/subexpr
                    num = 1
                    den = factorial(2 + sum([int(p.args[1]) if 
                                             len(p.args) > 1 else 1
                                             for p in power_list]))
                    for p in power_list:
                        if 'Pow' in str(p.func):
                            num *= factorial(p.args[1])
                    new_expr += factors*num/den
                    # string += \
                    #      f'mat[{i}, {j}] += '\
                    #         + f'{factors*num*1.0}/{1.0*den}\n'
            else:
                power_list = []
                list_a0a1a2(term, power_list)
                for power_of_a in power_list:
                    subexpr *= power_of_a
                factors = term/subexpr
                num = 1
                den = factorial(2 + sum([int(p.args[1]) if 
                                            len(p.args) > 1 else 1
                                            for p in power_list]))
                for p in power_list:
                    if 'Pow' in str(p.func):
                        num *= factorial(p.args[1])
                # string += \
                #         f'mat[{i}, {j}] += '\
                #             + f'{(1.0*factors*num).simplify()}/{1.0*den}\n'
                new_expr += factors*num/den
            sub_string = f'mat[{i}, {j}] = {new_expr}'
            # print(expr)
            # print(sub_string)
            string += sub_string + '\n'
    return string


def get_stuff_related_to_potential_matrix(basis_funcs):
    string = ''
    # https://stackoverflow.com/a/9493306
    k_vals = symbols(f'k0:{len(basis_funcs)}')
    # print(k_vals)
    for i in range(len(basis_funcs)):
        for j in range(0, i+1):
            expr = 0
            for k in range(len(basis_funcs)):
                k_val = k_vals[k]*basis_funcs[k]
                expr += k_val*basis_funcs[i]*basis_funcs[j]
            expr2 = expr.expand()
            expr3 = 0
            for t in expr2.args:
                a_vals = []
                not_a_vals = []
                for t2 in t.args:
                    if 'Pow' in str(t2.func) and t2.args[0] in [a0, a1, a2]:
                        a_vals.append(t2)
                    elif t2 in [a0, a1, a2]:
                        a_vals.append(t2)
                    else:
                        not_a_vals.append(t2)
                # print(not_a_vals, a_vals)
                pow_vals = []
                for a_val in a_vals:
                    if 'Pow' in str(a_val.func):
                        pow_vals.append(a_val.args[1])
                    else:
                        pow_vals.append(1)
                den = factorial(2 + sum(pow_vals))
                # print(den)
                num = 1
                for n in range(len(pow_vals)):
                    num *= factorial(pow_vals[n])
                expr4 = num/den
                # https://stackoverflow.com/a/45651175
                # expr4 = Rational(num/den)
                for val in not_a_vals:
                    expr4 *= val
                expr3 += expr4
            str_expr = f'mat[{i}, {j}] = ({expr3})'
            string += (str_expr + '\n')
    return string


print(get_stuff_related_to_mass_matrix(basis_funcs))
print(get_stuff_related_to_stiffness_matrix(basis_funcs))
print(get_stuff_related_to_potential_matrix(basis_funcs))