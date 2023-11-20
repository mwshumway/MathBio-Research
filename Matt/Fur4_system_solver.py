"""Matthew Shumway
Fur4 System Solver for Steady States."""

from sympy import solve  # We use sympy.solve module to solve the systems algebraically


def solve_four_system():
    """
    Solving Symbolically at Steady State at the Plasma Membrane Only

    returns:
            dict(sympy.abc, list[sympy.abc]): equations of concentrations at s.s.

    Abnormal Parameters (that don't match up with symbols in the paper):
            E = S_e
            B = P_b
            U = P_u
    All other variables are the same.
    """
    from sympy.abc import a, y, k, E, P, W, S, j, B, z, U

    # Define equations. Set to 0 by default
    equations = [
        y - k*E*P - (k/W)*S*P + j*B,
        k*E*P + (k/W) * S * P - j*B - a*B,
        a*B - z*U
    ]
    return solve(equations, [P, B, U])


def solve_full_model():
    """
    Solves Symbolically at Steady State for the Full Model.

    returns:
            dict(sympy.abc, list[sympy.abc]) equations of concentrations at s.s.

    Abnormal Parameters (that don't match up with symbols in the paper):
            alpha = S_e
            B = P_b
            e = A_e
            p = A_p
            zeta = E_u
            nu = E_b
    """
    from sympy.abc import y, k, alpha, P, W, S, j, B, f, e, p, a, z, E, g, U, b, zeta, nu

    # Define equations. Set to 0 by default
    equations = [
        y - k*alpha*P - (k/W)*S*P + j*B + f*(e/p)*E,  # dP/dt
        k*alpha*P + (k/W)*S*P - j*B - a*B,  # dP_b/dt
        a*B - g*U,  # dP_u/dt
        g*(p/e)*U - b*zeta + a*nu - z*zeta,  # dE/dt
        b*zeta - (k/W)*S*E + j*nu - f*E,  # dE_b/dt
        (k/W)*S*E - j*nu - a*nu  # dE_u/dt
    ]
    return solve(equations, [P, B, U, E, nu, zeta])

def solve_full_model_new():
    from sympy.abc import y, k, alpha, P, W, S, j, B, f, e, p, a, z, E, g, U, b, zeta, nu

    equations = [
        y - k*alpha*P - (k/W)*S*P + j*B + f*(e/p)*E,  # dP/dt
        k*alpha*P + (k/W)*S*P - j*B - a*B,  # dP_b/dt
        a*B - g*U,  # dP_u/dt
        b * zeta - (k / W) * S * E + j * nu - f * E,  # dE/dt
        (k / W) * S * E - j * nu - a * nu,  # dE_b/dt
        g*(p/e)*U - b*zeta + a*nu - z*zeta  # dE_u/dt
    ]
    return solve(equations, [P, B, U, E, nu, zeta])


def main():
    """This function is used to print in appealing format."""
    print()
    print("Plasma Membrane Only:")
    print("-" * 25)
    sol1 = solve_four_system()
    for sym, eq in sol1.items():
        print(f"{sym} = {eq}")
    print()
    print()
    print("Full Model:")
    print("-" * 25)
    sol2 = solve_full_model()
    for sym, eq in sol2.items():
        print(f"{sym} = {eq}")


if __name__ == '__main__':
    print(solve_four_system())
    main()
