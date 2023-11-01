from sympy import solve

def solve_four_system():
    """
    E = S_e
    B = P_b
    U = P_u
    All other variables are the same.
    """
    from sympy.abc import a, y, k, E, P, W, S, j, B, z, U

    eqns = [
        y - k*E*P - (k/W)*S*P + j*B,
        k*E*P + (k/W) * S * P - j*B - a*B,
        a*B - z*U,
    ]
    return solve(eqns, [P, B, U])


def solve_full_model():
    """alpha = S_e
    B = P_b
    e = A_e
    p = A_p
    zeta = E_u
    nu = E_b
    """
    from sympy.abc import y, k, alpha, P, W, S, j, B, f, e, p, a, z, E, g, U, b, zeta, nu
    eqns = [
        y - k*alpha*P - (k/W)*S*P + j*B + f*(e/p)*E,  # dP/dt
        k*alpha*P + (k/W)*S*P - j*B - a*B,  # dP_b/dt
        a*B - g*U,  # dP_u/dt
        g*(p/e)*U - b*zeta + a*nu - z*zeta,  # dE/dt
        b*zeta - (k/W)*S*E + j*nu - f*E,  # dE_b/dt
        (k/W)*S*E - j*nu - a*nu  # dE_u/dt
    ]
    return solve(eqns, [P, B, U, E, nu, zeta])


sol = solve_full_model()
for sym, eq in sol.items():
    print(sym, eq)
