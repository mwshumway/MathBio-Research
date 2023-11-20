"""
Matthew Shumway
"""
from sympy import solve
from scipy.integrate import odeint
import numpy as np
from sympy.abc import y, k, alpha, P, W, S, j, B, f, e, p, a, z, E, g, U, b, zeta, nu

# Equations
# alpha = S_e, B = P_b, U = P_u, nu = E_b, zeta = E_u, e = A_e, p = A_p
equations = [
        y - k*alpha*P - (k/W)*S*P + j*B + f*(e/p)*E,  # dP/dt
        k*alpha*P + (k/W)*S*P - j*B - a*B,  # dP_b/dt
        a*B - g*U,  # dP_u/dt
        b * zeta - (k / W) * S * E + j * nu - f * E,  # dE/dt
        (k / W) * S * E - j * nu - a * nu,  # dE_b/dt
        g*(p/e)*U - b*zeta + a*nu - z*zeta  # dE_u/dt
    ]

# Solve for steady state
ss_sol_dict = solve(equations, [P, B, U, E, nu, zeta])

# solve for numerical solution
def full_model(initial_val, t, y, k, S_e, W, j, f, A_e, A_p, a, g, b, z, v_max, V, K_m):
    P, P_b, P_u, E, E_b, E_u, S = initial_val

    dP_dt = y - (k * S_e * P) - ((k/W) * S * P) + (j*P_b) + (f*(A_e/A_p)*E)
    dPb_dt = (k * S_e * P) + ((k/W) * S * P) - (j * P_b) - (a*P_b)
    dPu_dt = (a * P_b) - (g*(A_p/A_e)*P_u)
    dE_dt = (b * E_u) - ((k/W) * S * E) + (j * E_b) - (f * (A_e/A_p) *E)
    dEb_dt = ((k / W) * S * E) - (j * E_b) - (a * E_b)
    dEu_dt = (g * (A_p/A_e) * P_u) - (b * E_u) + (a * E_b) - (z * E_u)
    dS_dt = (-(k/W) * S * ((A_p/V) * P + (A_e/V) * E)) + ((j + a) * ((A_p/V) * P_b + (A_e/V) * E_b)) - (v_max * S /
                                                                                                    (V * (K_m + S)))
    # dS_dt = ((-1*k/W) * S * (P + E)) + ((j + a) * (P_b + E_b)) - (v_max * S / (V * (K_m + S)))

    return [dP_dt, dPb_dt, dPu_dt, dE_dt, dEb_dt, dEu_dt, dS_dt]

# Params
a = 1
b = 1
f = 0.1
g = 0.1
j = 100
K_d = .74
k = j / K_d
y = 0.000083
z = 0.002
A_e, e = 47, 47
A_p, p = 314, 314
W = 32
v_max = 8.8 * 10 ** 3
K_m = 2.5
V = 523
S_e, alpha = 0.15, 0.15

initial_cond = [1, 0, 0, 0, 0, 0, 0]
t_range = np.linspace(0, 1000000, 100000)
sol = odeint(full_model, y0=initial_cond, t=t_range, args=(y, k, S_e, W, j, f, A_e, A_p, a, g, b,
                                                         z, v_max, V, K_m))

numerical_steady_state = sol[-1]

S = 1.1
analytical_steady_state = [eval(str(x)) for key, x in ss_sol_dict.items()]
print(numerical_steady_state, analytical_steady_state, sep='\n')

