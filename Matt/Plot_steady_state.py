from matplotlib import pyplot as plt
import numpy as np


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


def plasma_membrane(S_e):
    P_b = y / a
    P_u = y / z
    S = (S_e * W * (V * a * y + V * j * y - a * v_max) + np.sqrt(S_e * W * (
                4 * K_m * V * a ** 2 * v_max * y + 4 * K_m * V * a * j * v_max * y + S_e * V ** 2 * W * a ** 2 * y ** 2
                + 2 * S_e * V ** 2 * W * a * j * y ** 2 + S_e * V ** 2 * W * j ** 2 * y ** 2 - 2 * S_e * V * W * a ** 2
                * v_max * y - 2 * S_e * V * W * a * j * v_max * y + S_e * W * a ** 2 * v_max ** 2))) / (
                    2 * a * v_max)
    P = ((y * W * a) + (y * W * j)) / ((S_e * a * k * W) + (a * k * S))

    return [P, P_b, P_u, S]


def plasma_membrane_noS(S_e, S):
    P = (y*W*a + y*W*j) / (S_e*a*k*W + a*k*S)
    Pb = y/a
    Pu = y/z

    return P, Pb, Pu

def plot_plasma_membrane():
    S_e = np.linspace(0.001, 0.05, 1000)
    P_vals = []
    Pb_vals = []
    Pu_vals = []
    S_vals = []
    for s in S_e:
         P, Pb, Pu, S = plasma_membrane(s)
         P_vals.append(P)
         Pb_vals.append(Pb)
         Pu_vals.append(Pu)
         S_vals.append(S)
    plt.plot(S_e, P_vals, label="P")
    plt.plot(S_e, Pb_vals, label="P_b")
    plt.plot(S_e, Pu_vals, label="P_u")
    plt.plot(S_e, S_vals, label="S")

    plt.legend()
    plt.xlabel("S_e")
    plt.show()


def plot_plasma_noS():
    S_range = np.linspace(0.001, 1, 1000)
    P_vals = []
    Pb_vals = []
    Pu_vals = []
    S_e = 0.001
    for s in S_range:
        P, Pb, Pu = plasma_membrane_noS(S_e, s)
        P_vals.append(P)
        Pb_vals.append(Pb)
        Pu_vals.append(Pu)

    plt.plot(S_range, P_vals, label="P")
    plt.plot(S_range, Pb_vals, label="P_b")
    plt.plot(S_range, Pu_vals, label="P_u")

    plt.legend()
    plt.xlabel("S")
    plt.title(f"S_e = {S_e}")
    plt.show()


if __name__ == '__main__':
    plot_plasma_noS()
    plot_plasma_membrane()
