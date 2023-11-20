from scipy.integrate import odeint
from matplotlib import pyplot as plt
import numpy as np


def plasma_membrane(initial_val, t, y, k, S_e, W, j, a, z, v_max, V, K_m):
    P, P_b, P_u, S = initial_val

    dP_dt = y - k * S_e * P - (k / W) * S * P + j * P_b
    dPb_dt = k * S_e * P + (k / W) * S * P - j * P_b - a * P_b
    dPu_dt = a * P_b - z * P_u
    dS_dt = -1 * (k / W) * S * P + (j + a) * P_b - v_max * S / (V * (K_m + S))

    return [dP_dt, dPb_dt, dPu_dt, dS_dt]


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


# params
a = 1
b = 1
f = 0.1
g = 0.1
j = 100
K_d = .74
k = j / K_d
y = 0.000083
z = 0.002
A_e = 47
A_p = 314
W = 32
v_max = 8.8 * 10 ** 3
K_m = 2.5
V = 523
t_range = np.linspace(0, 1000, 100000)
Se_vals = [0.01, 0.1, 0.15, 0.25, 0.35, 0.7]


def plot_PM():
    # Plasma Membrane Only
    for i, S_e in enumerate(Se_vals):
        plt.subplot(3, 2, i + 1)
        PM_sol = odeint(plasma_membrane, y0=[50, 0, 0, 0], t=t_range, args=(y, k, S_e, W, j, a, z, v_max, V, K_m))

        plt.plot(t_range, PM_sol[:, 0], 'b')
        plt.plot(t_range, PM_sol[:, 1], 'r')
        plt.plot(t_range, PM_sol[:, 2], 'g')
        plt.plot(t_range, PM_sol[:, 3], 'm')
        plt.title(f"S_e = {S_e}")

    plt.legend(["PM Unbound", "PM Bound", "PM ubiq.", "Intracellular Uracil"])
    plt.tight_layout()
    plt.show()


def plot_FM():
    # Full Model
    y0 = [1, 0, 0, 0, 0, 0, 0]
    for i, S_e in enumerate(Se_vals):
        plt.subplot(3, 2, i + 1)
        FM_sol = odeint(full_model, y0, t=t_range, args=(y, k, S_e, W, j, f, A_e, A_p, a, g, b,
                                                         z, v_max, V, K_m))
        plt.plot(t_range, FM_sol[:, 0], 'b')
        plt.plot(t_range, FM_sol[:, 1], 'r')
        plt.plot(t_range, FM_sol[:, 2], 'g')
        plt.plot(t_range, FM_sol[:, 3], 'm')  # dE/dt
        plt.plot(t_range, FM_sol[:, 4], 'orange')
        plt.plot(t_range, FM_sol[:, 5], 'k')
        plt.plot(t_range, FM_sol[:, 6], 'pink')
        plt.title(f"S_e = {S_e}")

    plt.legend(["PM Unbound", "PM Bound", "PM ubiq.", "End. Unbound", "End. Bound", "End. ubiq.",
                "Intracellular Uracil"])
    plt.tight_layout()
    plt.show()


# S_e ranges between 0 and 5 - Dr. Dixon
if __name__ == '__main__':
    plot_FM()
