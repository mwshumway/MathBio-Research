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
t_range = np.linspace(0, 8000, 100000)
Se_vals = [.01, .1, 1]


def plot_PM(Se1=.01, Se2=.1):
    plt.figure(figsize=(8,8))
    time = np.linspace(30000, 60000, 1000000)
    time1 = time[:len(time)//2]
    time2 = time[len(time)//2:]
    PM_sol1 = odeint(plasma_membrane, y0=[50, 0, 0, 0], t=time1, args=(y, k, Se1, W, j, a, z, v_max, V, K_m))
    PM_sol2 = odeint(plasma_membrane, y0=PM_sol1[-1], t=time2, args=(y, k, Se2, W, j, a, z, v_max, V, K_m))

    PM_sol = np.vstack((PM_sol1, PM_sol2))

    plt.plot(time, PM_sol[:, 0], label=r"$P$")
    plt.plot(time, PM_sol[:, 1], label=r"$P_b$")
    plt.plot(time, PM_sol[:, 2], label=r"$P_u$")
    plt.plot(time, PM_sol[:, 3], label=r"$S$")

    plt.legend()
    plt.title(rf"$Se_1$={Se1}, Se_2 = {Se2}")
    plt.ylim((0, .2))
    plt.show()


def plot_FM(Se1=.01, Se2=.1):
    # Full Model
    plt.figure(figsize=(8, 8))
    time = np.linspace(1e2, 1e6, 1000000)
    time1 = time[:len(time) // 2]
    time2 = time[len(time) // 2:]
    FM_sol1 = odeint(full_model, y0=[50, 0, 0, 0, 0, 0, 0], t=time1, args=(y, k, Se1, W, j, f, A_e, A_p, a, g, b, z, v_max, V, K_m))
    FM_sol2 = odeint(full_model, y0=FM_sol1[-1], t=time2, args=(y, k, Se2, W, j, f, A_e, A_p, a, g, b, z, v_max, V, K_m))

    FM_sol = np.vstack((FM_sol1, FM_sol2))

    plt.plot(time, FM_sol[:, 0], 'b', label=r"P")
    plt.plot(time, FM_sol[:, 1], 'r', label=r"P_b")
    plt.plot(time, FM_sol[:, 2], 'g', label=r"P_u")
    plt.plot(time, FM_sol[:, 3], 'm', label=r"E")  # dE/dt
    plt.plot(time, FM_sol[:, 4], 'orange', label=r"E_b")
    plt.plot(time, FM_sol[:, 5], 'k', label=r"E_u")
    plt.plot(time, FM_sol[:, 6], 'pink', label=r"S")
    plt.title(rf"$Se_1$={Se1}, Se_2 = {Se2}")
    plt.legend()

    plt.ylim((0, 4))
    plt.show()


if __name__ == '__main__':
    plot_PM()
    plot_FM()
