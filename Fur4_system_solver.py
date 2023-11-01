from sympy import solve
import random as r


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
#
#
sol = solve_full_model()
for sym, eq in sol.items():
    print(sym, eq)

# for x in range(5):
#     # Get random variables
#     a = r.randint(1, 100)
#     k = r.randint(1, 100)
#     y = r.randint(1, 100)
#     W = r.randint(1, 100)
#     j = r.randint(1, 100)
#     z = r.randint(1, 100)
#     f = r.randint(1, 100)
#     b = r.randint(1, 100)
#     A_e = r.randint(1, 100)
#     A_p = r.randint(1, 100)
#     S = r.randint(1, 100)
#     g = r.randint(1, 100)
#     S_e = r.randint(1, 100)
#
#     # Define what we know
#     E_u = (A_p*y) / (A_e * z)
#     E_b = (a*b*A_p*y*W + b*j*A_p*W) / (a*A_e*k*z*S + a*A_e*f*z*W + A_e*f*j*z*W)
#     E = (W*a*b*A_p*y + W*b*j*A_p*y)/(S*a*A_e*k*z + W*a*A_e*f*z + W*A_e*f*j*z)
#     P_u = (S*a*k*y*z + W*a*b*f*y + W*a*f*y*z + W*b*f*j*y + W*f*j*y*z) /\
#           (S*a*g*k*z + W*a*f*g*z + W*f*g*j*z)
#     P_b = (S*a*k*y*z + W*a*b*f*y + W*a*f*y*z + W*b*f*j*y + W*f*j*y*z) /\
#           (S*a**2*k*z + W*a**2*f*z + W*a*f*j*z)
#     P = (S*W*a**2*k*y*z + S*W*a*j*k*y*z + W**2*a**2*b*f*y + W**2*a**2*f*y*z +
#          2*W**2*a*b*f*j*y + 2*W**2*a*f*j*y*z + W**2*b*f*j**2*y + W**2*f*j**2*y*z) /\
#         (S**2*a**2*k**2*z + S*W*a**2*S_e*k**2*z + S*W*a**2*f*k*z + S*W*a*f*j*k*z + W**2*a**2*S_e*f*k*z
#          + W**2*a*S_e*f*j*k*z)
#
#     print(y - k * S_e * P - (k / W) * S * P + j * P_b + f * (A_e / A_p) * E)
#     print(k * S_e * P + (k / W) * S * P - j * P_b - a * P_b)
#     print(a * P_b - g * P_u)
#     print(g * (A_p / A_e) * P_u - b * E_u + a * E_b)
#     print(b * E_u - (k / W) * S * E + j * E_b - f * E)
#     print((k / W) * S * E - j * E_b - a * E_b)
