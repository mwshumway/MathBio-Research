"""Matthew Shumway"""
from Plot_ODEs import plasma_membrane
from sympy import solve
from scipy.integrate import odeint
import numpy as np
from sympy.abc import y, k, alpha, P, W, S, j, B, a, z, U, v, K, V

# Equations
# alpha = S_e, B = P_b, U = P_u
equations = [
    y - k * alpha * P - (k / W) * S * P + j * B,
    k * alpha * P + (k / W) * S * P - j * B - a * B,
    a * B - z * U,
    -1 * (k / W) * S * P + (j + a) * U - v * S / (V * (K + S))
    ]

# solve for numerical solution
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
v_m = 8.8 * 10 ** 3
K_m = 2.5
V = 523
S_e, alpha = 0.15, 0.15

initial_cond = [1, 0, 0, 0]
t_range = np.linspace(0, 10000, 100000)
sol = odeint(plasma_membrane, y0=initial_cond, t=t_range, args=(y, k, S_e, W, j, a, z, v_m, V, K_m))

# Steady States
P = (-S_e*V*W*a*y - S_e*V*W*j*y - S_e*W*a*v_m + np.sqrt(S_e*W*(4*K_m*V*a**2*v_m*y + 4*K_m*V*a*j*v_m*y + S_e*V**2*W*a**2*y**2 + 2*S_e*V**2*W*a*j*y**2 + S_e*V**2*W*j**2*y**2 - 2*S_e*V*W*a**2*v_m*y - 2*S_e*V*W*a*j*v_m*y + S_e*W*a**2*v_m**2)))/(2*S_e*V*a*k*(K_m - S_e*W))
P_b = y/a
P_u = y/z
S = (S_e*W*(V*a*y + V*j*y - a*v_m) + np.sqrt(S_e*W*(4*K_m*V*a**2*v_m*y + 4*K_m*V*a*j*v_m*y + S_e*V**2*W*a**2*y**2 + 2*S_e*V**2*W*a*j*y**2 + S_e*V**2*W*j**2*y**2 - 2*S_e*V*W*a**2*v_m*y - 2*S_e*V*W*a*j*v_m*y + S_e*W*a**2*v_m**2)))/(2*a*v_m)


numerical_steady_state = sol[-1]

analytical_steady_state = [P, P_b, P_u, S]

print("Differences between analytical and numerical steady states")

for x, y in zip(numerical_steady_state, analytical_steady_state):
    print(round(np.abs(x - y), 5))
