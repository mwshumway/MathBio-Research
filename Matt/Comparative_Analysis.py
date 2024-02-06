import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from scipy import optimize as opt


# Define parameter set
# a = 1
# b = 1
# A_p = 314
# j = 10 ** 2
# K_d = 0.74
# K_m = 2.5
# k = j / K_d
# V_m = 8.8 * 10 ** 3
# V = 523
# W = 32
# y = 0.000083
# z = .002
# f = 0.1
# A_e = 47
# g = 0.1
# Se = 10

# Steady State Analytical Solutions
def P_pm_eval(S_e, a=1, A_p=314, j=100, K_m=2.5, k=100 / 0.74, V_m=8.8e3, W=32, y=0.000083):
    return (A_p * S_e * W * a * y + A_p * S_e * W * j * y + S_e * V_m * W * a -
            np.sqrt(S_e * W * (
                    A_p ** 2 * S_e * W * a ** 2 * y ** 2 + 2 * A_p ** 2 * S_e * W * a * j * y ** 2 + A_p ** 2 * S_e * W * j ** 2 * y ** 2 -
                    2 * A_p * S_e * V_m * W * a ** 2 * y - 2 * A_p * S_e * V_m * W * a * j * y + 4 * A_p * K_m * V_m * a ** 2 * y +
                    4 * A_p * K_m * V_m * a * j * y + S_e * V_m ** 2 * W * a ** 2))) / (
            2 * A_p * S_e * a * k * (S_e * W - K_m))


def P_b_pm_eval(y=0.000083, a=1):
    return y / a


def P_u_pm_eval(y=0.000083, z=0.002):
    return y / z


def S_pm_eval(S_e, a=1, A_p=314, j=100, K_m=2.5, V_m=8.8e3, W=32, y=0.000083):
    return (S_e * W * (A_p * a * y + A_p * j * y - V_m * a) +
            np.sqrt(S_e * W * (
                    A_p ** 2 * S_e * W * a ** 2 * y ** 2 + 2 * A_p ** 2 * S_e * W * a * j * y ** 2 + A_p ** 2 * S_e * W * j ** 2 * y ** 2 -
                    2 * A_p * S_e * V_m * W * a ** 2 * y - 2 * A_p * S_e * V_m * W * a * j * y + 4 * A_p * K_m * V_m * a ** 2 * y +
                    4 * A_p * K_m * V_m * a * j * y + S_e * V_m ** 2 * W * a ** 2))) / (2 * V_m * a)


# P_pm = lambda S_e: (A_p * S_e * W * a * y + A_p * S_e * W * j * y + S_e * V_m * W * a -
#                     np.sqrt(S_e * W * (
#                             A_p ** 2 * S_e * W * a ** 2 * y ** 2 + 2 * A_p ** 2 * S_e * W * a * j * y ** 2 + A_p ** 2 * S_e * W * j ** 2 * y ** 2 -
#                             2 * A_p * S_e * V_m * W * a ** 2 * y - 2 * A_p * S_e * V_m * W * a * j * y + 4 * A_p * K_m * V_m * a ** 2 * y +
#                             4 * A_p * K_m * V_m * a * j * y + S_e * V_m ** 2 * W * a ** 2))) / (
#                            2 * A_p * S_e * a * k * (S_e * W - K_m))
#
# P_b_pm = y / a
#
# P_u_pm = y / z
#
# S_pm = lambda S_e: (S_e * W * (A_p * a * y + A_p * j * y - V_m * a) +
#                     np.sqrt(S_e * W * (
#                             A_p ** 2 * S_e * W * a ** 2 * y ** 2 + 2 * A_p ** 2 * S_e * W * a * j * y ** 2 + A_p ** 2 * S_e * W * j ** 2 * y ** 2 -
#                             2 * A_p * S_e * V_m * W * a ** 2 * y - 2 * A_p * S_e * V_m * W * a * j * y + 4 * A_p * K_m * V_m * a ** 2 * y +
#                             4 * A_p * K_m * V_m * a * j * y + S_e * V_m ** 2 * W * a ** 2))) / (2 * V_m * a)
#

def eval_fm_alg(S, S_e, a=1, b=1, A_p=314, j=100, k=100 / 0.74, W=32, y=0.000083,
                z=0.002, f=0.1, A_e=47, g=0.1):
    P = (
                S * W * a ** 2 * k * y * z + S * W * a * j * k * y * z + W ** 2 * a ** 2 * b * f * y + W ** 2 * a ** 2 * f * y * z + 2 * W ** 2 * a * b * f * j * y + 2 * W ** 2 * a * f * j * y * z + W ** 2 * b * f * j ** 2 * y + W ** 2 * f * j ** 2 * y * z) / (
                S ** 2 * a ** 2 * k ** 2 * z + S * S_e * W * a ** 2 * k ** 2 * z + S * W * a ** 2 * f * k * z + S * W * a * f * j * k * z + S_e * W ** 2 * a ** 2 * f * k * z + S_e * W ** 2 * a * f * j * k * z)
    Pb = (S * a * k * y * z + W * a * b * f * y + W * a * f * y * z + W * b * f * j * y + W * f * j * y * z) / (
            S * a ** 2 * k * z + W * a ** 2 * f * z + W * a * f * j * z)
    Pu = (S * a * k * y * z + W * a * b * f * y + W * a * f * y * z + W * b * f * j * y + W * f * j * y * z) / (
            S * a * g * k * z + W * a * f * g * z + W * f * g * j * z)
    E = (A_p * W * a * b * y + A_p * W * b * j * y) / (A_e * S * a * k * z + A_e * W * a * f * z + A_e * W * f * j * z)
    Eb = A_p * S * b * k * y / (A_e * S * a * k * z + A_e * W * a * f * z + A_e * W * f * j * z)
    Eu = A_p * y / (A_e * z)

    return [P, Pb, Pu, E, Eb, Eu]


#
# P_fm = lambda S, S_e, a: (
#                               S * W * a ** 2 * k * y * z + S * W * a * j * k * y * z + W ** 2 * a ** 2 * b * f * y + W ** 2 * a ** 2 * f * y * z + 2 * W ** 2 * a * b * f * j * y + 2 * W ** 2 * a * f * j * y * z + W ** 2 * b * f * j ** 2 * y + W ** 2 * f * j ** 2 * y * z) / (
#                               S ** 2 * a ** 2 * k ** 2 * z + S * S_e * W * a ** 2 * k ** 2 * z + S * W * a ** 2 * f * k * z + S * W * a * f * j * k * z + S_e * W ** 2 * a ** 2 * f * k * z + S_e * W ** 2 * a * f * j * k * z)
#
# P_b_fm = lambda S, a: (
#                            S * a * k * y * z + W * a * b * f * y + W * a * f * y * z + W * b * f * j * y + W * f * j * y * z) / (
#                            S * a ** 2 * k * z + W * a ** 2 * f * z + W * a * f * j * z)
#
# E_fm = lambda S, a: (A_p * W * a * b * y + A_p * W * b * j * y) / (
#         A_e * S * a * k * z + A_e * W * a * f * z + A_e * W * f * j * z)
#
# E_b_fm = lambda S, a: A_p * S * b * k * y / (A_e * S * a * k * z + A_e * W * a * f * z + A_e * W * f * j * z)


def E_u_fm_eval(A_p=314,y=0.000083, z=0.002, A_e=47):
    return A_p * y / (A_e * z)


def E_b_fm_eval(S, a=1, b=1, A_p=314, j=100, k=100 / 0.74, W=32, y=0.000083, z=0.002, f=0.1, A_e=47):
    return A_p * S * b * k * y / (A_e * S * a * k * z + A_e * W * a * f * z + A_e * W * f * j * z)


def E_fm_eval(S, a=1, b=1, A_p=314, j=100, k=100 / 0.74, W=32, y=0.000083, z=0.002, f=0.1, A_e=47):
    return (A_p * W * a * b * y + A_p * W * b * j * y) / (
            A_e * S * a * k * z + A_e * W * a * f * z + A_e * W * f * j * z)


def P_u_fm_eval(S, a=1, b=1, j=100, k=100 / 0.74, W=32, y=0.000083,
                z=0.002, f=0.1, g=0.1):
    return (S * a * k * y * z + W * a * b * f * y + W * a * f * y * z + W * b * f * j * y + W * f * j * y * z) / (
            S * a * g * k * z + W * a * f * g * z + W * f * g * j * z)


def P_b_fm_eval(S, a=1, b=1, j=100, k=100 / 0.74, W=32, y=0.000083, z=0.002, f=0.1):
    return (
            S * a * k * y * z + W * a * b * f * y + W * a * f * y * z + W * b * f * j * y + W * f * j * y * z) / (
            S * a ** 2 * k * z + W * a ** 2 * f * z + W * a * f * j * z)


def P_fm_eval(S, S_e, a=1, b=1, j=100, k=100 / 0.74, W=32, y=0.000083,
              z=0.002, f=0.1):
    return (
            S * W * a ** 2 * k * y * z + S * W * a * j * k * y * z + W ** 2 * a ** 2 * b * f * y + W ** 2 * a ** 2 * f * y * z + 2 * W ** 2 * a * b * f * j * y + 2 * W ** 2 * a * f * j * y * z + W ** 2 * b * f * j ** 2 * y + W ** 2 * f * j ** 2 * y * z) / (
            S ** 2 * a ** 2 * k ** 2 * z + S * S_e * W * a ** 2 * k ** 2 * z + S * W * a ** 2 * f * k * z + S * W * a * f * j * k * z + S_e * W ** 2 * a ** 2 * f * k * z + S_e * W ** 2 * a * f * j * k * z)


def steady_S_bisect(Se, a=1, b=1, A_p=314, j=100, K_m=2.5, k=100 / 0.74, V_m=8.8e3, V=523, W=32, y=0.000083,
                    z=0.002, f=0.1, A_e=47):
    """Use bisection method to get steady state of S"""
    dSdt = lambda S: -(k / W) * S * (
                (A_p / V) * P_fm_eval(S, Se, a, b, A_p, j) + (
                    A_e / V) * E_fm_eval(S, a, b, A_p, j, k, W, y, z, f, A_e)) + (j + a) * (
                             (A_p / V) * P_b_fm_eval(S, a, b, j, k, W, y, z, f) + (A_e / V) * E_b_fm_eval(S, a, b, A_p,
                                                                                                          j, k, W, y, z,
                                                                                                          f,
                                                                                                          A_e)) - V_m * S / (
                                 V * (K_m + S))
    return opt.bisect(dSdt, 0, 1)


# -----------------------------------------------------------------------------------------------------------------------
# Analysis

def comparison_extracellular_uracil():
    # Full model
    plt.figure(figsize=(10, 10))
    a = 1
    b = 1
    A_p = 314
    j = 10 ** 2
    K_d = 0.74
    K_m = 2.5
    k = j / K_d
    V_m = 8.8 * 10 ** 3
    V = 523
    W = 32
    y = 0.000083
    z = .002
    f = 0.1
    A_e = 47
    g = 0.1

    Se_vals = np.linspace(0.01, 1, 250)
    total_pm_fur4 = []
    total_end_fur4 = []
    intracellular_uracil = []
    percent_pm_fur4 = []
    percent_end_fur4 = []
    percent_intr = []
    for s_e in Se_vals:
        dSdt = lambda S: -(k / W) * S * ((A_p / V) * P_fm_eval(S, s_e) + (A_e / V) * E_fm_eval(S)) + (j + a) * (
                (A_p / V) * P_b_fm_eval(S) + (A_e / V) * E_b_fm_eval(S)) - V_m * S / (V * (K_m + S))
        S_i = opt.bisect(dSdt, 0, 1)
        intracellular_uracil.append(S_i)
        fur4 = eval_fm_alg(S_i, s_e)
        total_pm_fur4.append(fur4[0] + fur4[1] + fur4[2])
        total_end_fur4.append(fur4[3] + fur4[4] + fur4[5])
        percent_pm_fur4.append((fur4[0] + fur4[1] + fur4[2]) / total_pm_fur4[0])
        percent_end_fur4.append((fur4[3] + fur4[4] + fur4[5]) / total_end_fur4[0])
        percent_intr.append(S_i / intracellular_uracil[0])

    plt.subplot(221)
    plt.plot(Se_vals, total_pm_fur4, 'r', label="Total P.M. Fur4")
    plt.plot(Se_vals, total_end_fur4, 'purple', label="Total Endosomal Fur4")
    plt.plot(Se_vals, [p + q for p, q in zip(total_end_fur4, total_pm_fur4)], 'k', label="Total Fur4")
    plt.title("Full Model Steady States", fontsize=10)
    plt.legend()
    plt.xlabel("Extracellular Uracil")
    plt.ylabel("Total Fur4")

    plt.subplot(222)
    plt.plot(Se_vals, intracellular_uracil)
    plt.title("Full Model Steady States", fontsize=10)
    plt.xlabel("Extracellular Uracil")
    plt.ylabel("Intracellular Uracil")

    # plt.subplot(423)
    # plt.plot(Se_vals, percent_pm_fur4, 'r', label=r"$\frac{P.M. Fur4}{initial P.M. Fur4}$")
    # plt.plot(Se_vals, percent_end_fur4, 'purple', label=r"$\frac{End Fur4}{initial End Fur4}$")
    # plt.legend()
    # plt.xlabel("Extracellular Uracil")
    # plt.ylabel("Percent")
    # plt.ylim((0, 1))
    #
    # plt.subplot(424)
    # plt.plot(Se_vals, percent_intr, label="% intracellular")
    # plt.legend()
    # plt.xlabel("Extracellular Uracil")
    # plt.ylabel("Percent")
    # plt.ylim((0.9, 2))

    # Plasma Membrane Only Model
    plt.subplot(223)
    total_fur4 = P_pm_eval(S_e=Se_vals) + P_b_pm_eval() + P_u_pm_eval()
    plt.plot(Se_vals, total_fur4, 'r')
    plt.xlabel("Extracellular Uracil")
    plt.ylabel("Total Fur4")
    plt.title("Plasma Membrane Only Model Steady States", fontsize=10)
    plt.subplot(224)
    plt.plot(Se_vals, S_pm_eval(S_e=Se_vals))
    plt.xlabel("Extracellular Uracil")
    plt.ylabel("Intracellular Uracil")
    plt.title("Plasma Membrane Only Model Steady States", fontsize=10)

    # plt.subplot(427)
    # plt.plot(Se_vals, total_fur4 / total_fur4[0])
    # plt.ylim((0, 1))
    #
    # plt.subplot(428)
    # initial = S_pm_eval(S_e=Se_vals)[0]
    # plt.plot(Se_vals, S_pm_eval(S_e=Se_vals) / initial)
    # plt.ylim((0.9, 2))

    plt.tight_layout()
    plt.show()


def comparison_ubiq_rate():
    plt.figure(figsize=(10, 10))
    b = 1
    A_p = 314
    j = 10 ** 2
    K_d = 0.74
    K_m = 2.5
    k = j / K_d
    V_m = 8.8 * 10 ** 3
    V = 523
    W = 32
    y = 0.000083
    z = .002
    f = 0.1
    A_e = 47
    g = 0.1

    s_e = 1  # fix extracellular uracil level
    ubiq_rate = np.linspace(0.01, 2, 250)  # range of ubiquitination rates

    def eval_fm_alg(S, ub_r, S_e):
        P = (
                    S * W * ub_r ** 2 * k * y * z + S * W * ub_r * j * k * y * z + W ** 2 * ub_r ** 2 * b * f * y + W ** 2 * ub_r ** 2 * f * y * z + 2 * W ** 2 * ub_r * b * f * j * y + 2 * W ** 2 * ub_r * f * j * y * z + W ** 2 * b * f * j ** 2 * y + W ** 2 * f * j ** 2 * y * z) / (
                    S ** 2 * ub_r ** 2 * k ** 2 * z + S * S_e * W * ub_r ** 2 * k ** 2 * z + S * W * ub_r ** 2 * f * k * z + S * W * ub_r * f * j * k * z + S_e * W ** 2 * ub_r ** 2 * f * k * z + S_e * W ** 2 * ub_r * f * j * k * z)
        Pb = (
                     S * ub_r * k * y * z + W * ub_r * b * f * y + W * ub_r * f * y * z + W * b * f * j * y + W * f * j * y * z) / (
                     S * ub_r ** 2 * k * z + W * ub_r ** 2 * f * z + W * ub_r * f * j * z)
        Pu = (
                     S * ub_r * k * y * z + W * ub_r * b * f * y + W * ub_r * f * y * z + W * b * f * j * y + W * f * j * y * z) / (
                     S * ub_r * g * k * z + W * ub_r * f * g * z + W * f * g * j * z)
        E = (A_p * W * ub_r * b * y + A_p * W * b * j * y) / (
                A_e * S * ub_r * k * z + A_e * W * ub_r * f * z + A_e * W * f * j * z)
        Eb = A_p * S * b * k * y / (A_e * S * ub_r * k * z + A_e * W * ub_r * f * z + A_e * W * f * j * z)
        Eu = A_p * y / (A_e * z)

        return [P, Pb, Pu, E, Eb, Eu]

    def eval_pm_alg(S_e, ub_r):
        P_pm = (A_p * S_e * W * ub_r * y + A_p * S_e * W * j * y + S_e * V_m * W * ub_r -
                np.sqrt(S_e * W * (
                        A_p ** 2 * S_e * W * ub_r ** 2 * y ** 2 + 2 * A_p ** 2 * S_e * W * ub_r * j * y ** 2 + A_p ** 2 * S_e * W * j ** 2 * y ** 2 -
                        2 * A_p * S_e * V_m * W * ub_r ** 2 * y - 2 * A_p * S_e * V_m * W * ub_r * j * y + 4 * A_p * K_m * V_m * ub_r ** 2 * y +
                        4 * A_p * K_m * V_m * ub_r * j * y + S_e * V_m ** 2 * W * ub_r ** 2))) / (
                       2 * A_p * S_e * ub_r * k * (S_e * W - K_m))

        P_b_pm = y / ub_r

        P_u_pm = y / z

        S_pm = (S_e * W * (A_p * ub_r * y + A_p * j * y - V_m * ub_r) +
                np.sqrt(S_e * W * (
                        A_p ** 2 * S_e * W * ub_r ** 2 * y ** 2 + 2 * A_p ** 2 * S_e * W * ub_r * j * y ** 2 + A_p ** 2 * S_e * W * j ** 2 * y ** 2 -
                        2 * A_p * S_e * V_m * W * ub_r ** 2 * y - 2 * A_p * S_e * V_m * W * ub_r * j * y + 4 * A_p * K_m * V_m * ub_r ** 2 * y +
                        4 * A_p * K_m * V_m * ub_r * j * y + S_e * V_m ** 2 * W * ub_r ** 2))) / (2 * V_m * ub_r)

        return [P_pm, P_b_pm, P_u_pm, S_pm]

    # Full System
    total_pm_fur4 = []
    total_end_fur4 = []
    total_intra_uracil = []
    for ur in ubiq_rate:
        dSdt_ur = lambda S: -(k / W) * S * ((A_p / V) * P_fm_eval(S, s_e, a=ur) + (A_e / V) * E_fm_eval(S, a=ur)) + (
                    j + ur) * (
                                    (A_p / V) * P_b_fm_eval(S, a=ur) + (A_e / V) * E_b_fm_eval(S, a=ur)) - V_m * S / (
                                        V * (K_m + S))
        S_i = opt.bisect(dSdt_ur, -1e-10, 1e10)
        total_intra_uracil.append(S_i)
        fur4 = eval_fm_alg(S_i, ur, s_e)
        total_pm_fur4.append(fur4[0] + fur4[1] + fur4[2])
        total_end_fur4.append(fur4[3] + fur4[4] + fur4[5])

    plt.subplot(221)
    plt.plot(ubiq_rate, total_end_fur4, 'purple', label="Total Endosomal Fur4")
    plt.plot(ubiq_rate, total_pm_fur4, 'red', label="Total P.M. Fur4")
    plt.plot(ubiq_rate, [p + q for p, q in zip(total_end_fur4, total_pm_fur4)], 'k', label="Total Fur4")
    plt.title(fr"Full Model Steady States, $S_e={s_e}$", fontsize=10)
    plt.legend()
    plt.xlabel("Ubiquitination Rate")
    plt.ylabel("Total Fur4")

    plt.subplot(222)
    plt.plot(ubiq_rate, total_intra_uracil)
    plt.title(fr"Full Model Steady States, $S_e={s_e}$", fontsize=10)
    plt.xlabel("Ubiquitination Rate")
    plt.ylabel("Intracellular Uracil")

    total_fur4_pm_model = []
    total_uracil_pm = []
    for ur in ubiq_rate:
        f4 = eval_pm_alg(s_e, ur)
        total_fur4_pm_model.append(f4[0] + f4[1] + f4[2])
        total_uracil_pm.append(f4[-1])

    plt.subplot(223)
    plt.plot(ubiq_rate, total_fur4_pm_model, 'r', label="Total Fur4")
    plt.title(fr"Plasma Membrane Only Steady States, $S_e={s_e}$", fontsize=10)
    plt.legend()
    plt.xlabel("Ubiquitination Rate")
    plt.ylabel("Total Fur4")

    plt.subplot(224)
    plt.plot(ubiq_rate, total_uracil_pm)
    plt.title(fr"Plasma Membrane Only Steady States, $S_e={s_e}$", fontsize=10)
    plt.xlabel("Ubiquitination Rate")
    plt.ylabel("Intracellular Uracil")

    plt.tight_layout()
    plt.show()


def comparison_recycling_rate():
    """Alter j, the recycling rate of Fur4 to Plasma Membrane"""
    S_e = 1  # fix extracellular uracil
    a = 1  # fix ubiquitination rate
    b = 1
    A_p = 314
    j = 10 ** 2
    K_d = 0.74
    K_m = 2.5
    k = j / K_d
    V_m = 8.8 * 10 ** 3
    V = 523
    W = 32
    y = 0.000083
    z = .002
    # f = 0.1  we alter the recycling rate
    A_e = 47
    g = 0.1

    plt.figure(figsize=(10, 5))

    # Full Model
    recycle_rate = np.linspace(0.01, 1)  # change recycling rate of Fur4 to Plasma Membrane
    pm_fur4 = []
    percent_pm_fur4 = []
    end_fur4 = []
    percent_end_fur = []
    intra_uracil = []
    percent_intra_uracil = []
    for rr in recycle_rate:
        dSdt_ur = lambda S: -(k / W) * S * ((A_p / V) * P_fm_eval(S, S_e, f=rr) + (A_e / V) * E_fm_eval(S, f=rr)) + (
                j + a) * (
                                    (A_p / V) * P_b_fm_eval(S, f=rr) + (A_e / V) * E_b_fm_eval(S, f=rr)) - V_m * S / (
                                    V * (K_m + S))
        S_i = opt.bisect(dSdt_ur, 0, 50)
        intra_uracil.append(S_i)
        percent_intra_uracil.append(S_i / intra_uracil[0])
        pm = P_fm_eval(S_i, S_e, f=rr) + P_b_fm_eval(S_i, f=rr) + P_u_fm_eval(S_i, f=rr)
        pm_fur4.append(pm)
        percent_pm_fur4.append(pm / pm_fur4[0])
        end = E_fm_eval(S_i, f=rr) + E_b_fm_eval(S_i, f=rr) + E_u_fm_eval()
        end_fur4.append(end)
        percent_end_fur.append(end / end_fur4[0])
    total = np.array(pm_fur4) + np.array(end_fur4)

    plt.subplot(121)
    plt.plot(recycle_rate, pm_fur4, "r", label="Total P.M. Fur4")
    plt.plot(recycle_rate, end_fur4, 'purple', label="Total Endosomal Fur4")
    plt.plot(recycle_rate, total, "k", label="Total Fur4")
    plt.xlabel("Recycling Rate of Fur4 to Plasma Membrane")
    plt.ylabel("Fur4 Concentration")
    plt.title(fr"Full Model Steady States, $S_e={S_e}$")
    plt.legend()

    plt.subplot(122)
    plt.plot(recycle_rate, intra_uracil)
    plt.ylabel("Intracellular Uracil Concentration")
    plt.xlabel("Recycling Rate of Fur4 to Plasma Membrane")
    plt.title(fr"Full Model Steady States, $S_e={S_e}$")


    # plt.subplot(223)
    # plt.plot(recycle_rate, percent_pm_fur4, "r", label="P.M. Fur4")
    # plt.plot(recycle_rate, percent_end_fur, "purple", label="Endosomal Fur4")
    # plt.plot(recycle_rate, total / total[0], "k", label="Total Fur4")
    # plt.xlabel("Recycling Rate of Fur4 to Plasma Membrane")
    # plt.ylabel("Percentage of Initial")
    # plt.title(fr"Full Model Steady States, $S_e={S_e}$")
    # plt.ylim((0, 1))
    # plt.legend()
    #
    # plt.subplot(224)
    # plt.plot(recycle_rate, percent_intra_uracil)
    # plt.xlabel("Recycling Rate of Fur4 to Plasma Membrane")
    # plt.ylabel("Percentage of Initial")


    # Parameter f has no effect on the Plasma Membrane Only Model, so there is no point graphing it.

    plt.tight_layout()
    plt.show()


def comparison_endocytosis_rate():
    S_e = 0.01  # fix extracellular uracil
    a = 1  # fix ubiquitination rate
    b = 1
    A_p = 314
    j = 10 ** 2
    K_d = 0.74
    K_m = 2.5
    k = j / K_d
    V_m = 8.8 * 10 ** 3
    V = 523
    W = 32
    y = 0.000083
    z = .002
    f = 0.1
    A_e = 47
    # g = 0.1  we alter endocytosis rate

    plt.figure(figsize=(10, 5))

    # Full Model
    endocytos_rate = np.linspace(0.01, 1)  # changer recycling rate of Fur4 to Plasma Membrane
    pm_fur4 = []
    percent_pm_fur4 = []
    end_fur4 = []
    percent_end_fur = []
    intra_uracil = []
    percent_intra_uracil = []
    for er in endocytos_rate:
        dSdt_ur = lambda S: -(k / W) * S * ((A_p / V) * P_fm_eval(S, S_e) + (A_e / V) * E_fm_eval(S)) + (
                j + a) * (
                                    (A_p / V) * P_b_fm_eval(S) + (A_e / V) * E_b_fm_eval(S)) - V_m * S / (
                                    V * (K_m + S))
        S_i = opt.bisect(dSdt_ur, 0, 50)
        intra_uracil.append(S_i)
        percent_intra_uracil.append(S_i / intra_uracil[0])
        pm = P_fm_eval(S_i, S_e) + P_b_fm_eval(S_i) + P_u_fm_eval(S_i, g=er)
        pm_fur4.append(pm)
        percent_pm_fur4.append(pm / pm_fur4[0])
        end = E_fm_eval(S_i) + E_b_fm_eval(S_i) + E_u_fm_eval()
        end_fur4.append(end)
        percent_end_fur.append(end / end_fur4[0])
    total = np.array(pm_fur4) + np.array(end_fur4)

    plt.subplot(121)
    plt.plot(endocytos_rate, pm_fur4, "r", label="Total P.M. Fur4")
    plt.plot(endocytos_rate, end_fur4, 'purple', label="Total Endosomal Fur4")
    plt.plot(endocytos_rate, total, "k", label="Total Fur4")
    plt.xlabel("Endocytosis Rate")
    plt.ylabel("Fur4 Concentration")
    plt.title(fr"Full Model Steady State, $S_e={S_e}$")
    plt.legend()

    plt.subplot(122)
    plt.plot(endocytos_rate, intra_uracil)
    plt.ylabel("Intracellular Uracil Concentration")
    plt.xlabel("Endocytosis Rate")
    plt.title(fr"Full Model Steady State, $S_e={S_e}$")

    # plt.subplot(223)
    # plt.plot(endocytos_rate, percent_pm_fur4, "r", label="P.M. Fur4")
    # plt.plot(endocytos_rate, percent_end_fur, "purple", label="Endosomal Fur4")
    # plt.plot(endocytos_rate, total / total[0], "k", label="Total Fur4")
    # plt.xlabel("Endocytosis Rate")
    # plt.ylabel("Percentage of Initial")
    # plt.ylim((0, 1))
    # plt.legend()
    #
    # plt.subplot(224)
    # plt.plot(endocytos_rate, percent_intra_uracil)
    # plt.xlabel("Endocytosis Rate")
    # plt.ylabel("Percentage of Initial")

    # Parameter g has no effect on the Plasma Membrane Only Model, so there is no point graphing it.

    plt.tight_layout()
    plt.show()


def comparison_deubiq_rate():
    S_e = 1  # fix extracellular uracil
    a = 1  # fix ubiquitination rate
    # b = 1  We change b, the de-ubiquitination rate
    A_p = 314
    j = 10 ** 2
    K_d = 0.74
    K_m = 2.5
    k = j / K_d
    V_m = 8.8 * 10 ** 3
    V = 523
    W = 32
    y = 0.000083
    z = .002
    f = 0.1
    A_e = 47
    # g = 0.1  we alter endocytosis rate

    plt.figure(figsize=(10, 5))

    # Full Model
    deubiq_rate = np.linspace(0.01, 10)  # changer deubiq rate of Fur4 to Plasma Membrane
    pm_fur4 = []
    percent_pm_fur4 = []
    end_fur4 = []
    percent_end_fur = []
    intra_uracil = []
    percent_intra_uracil = []
    for dur in deubiq_rate:
        dSdt_ur = lambda S: -(k / W) * S * ((A_p / V) * P_fm_eval(S, S_e, b=dur) + (A_e / V) * E_fm_eval(S, b=dur)) + (
                j + a) * (
                                    (A_p / V) * P_b_fm_eval(S, b=dur) + (A_e / V) * E_b_fm_eval(S, b=dur)) - V_m * S / (
                                    V * (K_m + S))
        S_i = opt.bisect(dSdt_ur, 0, 50)
        intra_uracil.append(S_i)
        percent_intra_uracil.append(S_i / intra_uracil[0])
        pm = P_fm_eval(S_i, S_e, b=dur) + P_b_fm_eval(S_i, b=dur) + P_u_fm_eval(S_i, b=dur)
        pm_fur4.append(pm)
        percent_pm_fur4.append(pm / pm_fur4[0])
        end = E_fm_eval(S_i, b=dur) + E_b_fm_eval(S_i, b=dur) + E_u_fm_eval()
        end_fur4.append(end)
        percent_end_fur.append(end / end_fur4[0])
    total = np.array(pm_fur4) + np.array(end_fur4)

    plt.subplot(121)
    plt.plot(deubiq_rate, pm_fur4, "r", label="Total P.M. Fur4")
    plt.plot(deubiq_rate, end_fur4, 'purple', label="Total Endosomal Fur4")
    plt.plot(deubiq_rate, total, "k", label="Total Fur4")
    plt.xlabel("De-Ubiquitination Rate")
    plt.ylabel("Fur4 Concentration")
    plt.title(fr"Full Model Steady State, $S_e={S_e}$")
    plt.legend()

    plt.subplot(122)
    plt.plot(deubiq_rate, intra_uracil)
    plt.ylabel("Intracellular Uracil Concentration")
    plt.title(fr"Full Model Steady State, $S_e={S_e}$")
    plt.xlabel("De-Ubiquitination Rate")


    plasma_fur4 = []
    plasma_uracil = []
    for dur in deubiq_rate:
        plasma_uracil.append(S_pm_eval(S_e))

    # plt.subplot(223)
    # plt.plot(deubiq_rate, percent_pm_fur4, "r", label="P.M. Fur4")
    # plt.plot(deubiq_rate, percent_end_fur, "purple", label="Endosomal Fur4")
    # plt.plot(deubiq_rate, total / total[0], "k", label="Total Fur4")
    # plt.xlabel("De-Ubiquitination Rate")
    # plt.ylabel("Percentage of Initial")
    # plt.ylim((0, 1))
    # plt.legend()
    #
    # plt.subplot(224)
    # plt.plot(deubiq_rate, percent_intra_uracil)
    # plt.xlabel("De-Ubiquitination Rate")
    # plt.ylabel("Percentage of Initial")

    # Parameter g has no effect on the Plasma Membrane Only Model, so there is no point graphing it.
    # Is that strange? Is that a problem?

    plt.tight_layout()
    plt.show()


def comparison_unbinding_rate():
    """We alter j, the unbinding rate of uracil/fur4"""
    plt.figure(figsize=(10, 10))

    S_e = 1  # fix extracellular uracil
    a = 1
    b = 1
    A_p = 314
    # j = 10 ** 2
    K_d = 0.74
    K_m = 2.5
    # k = j / K_d need to also edit this
    V_m = 8.8 * 10 ** 3
    V = 523
    W = 32
    y = 0.000083
    z = .002
    f = 0.1
    A_e = 47
    g = 0.1

    unbinding_rate = np.linspace(10, 10 ** 3)

    fm_pm_fur4, fm_end_fur4, fm_ex_ur = [], [], []
    pm_fur4, pm_ex_ur = [], []
    for ubr in unbinding_rate:
        k = ubr / K_d
        dSdt_ur = lambda S: -(k / W) * S * ((A_p / V) * P_fm_eval(S, S_e, j=ubr, k=k) + (A_e / V) * E_fm_eval(S, j=ubr, k=k)) + (
                ubr + a) * (
                                    (A_p / V) * P_b_fm_eval(S, j=ubr, k=k) + (A_e / V) * E_b_fm_eval(S, j=ubr, k=k)) - V_m * S / (
                                    V * (K_m + S))
        S_i = opt.bisect(dSdt_ur, 0, 50)
        fm_ex_ur.append(S_i)
        fm_end_fur4.append(E_fm_eval(S_i, j=ubr, k=k) + E_b_fm_eval(S_i, j=ubr, k=k) + E_u_fm_eval())
        fm_pm_fur4.append(P_fm_eval(S_i, S_e, j=ubr, k=k) + P_b_fm_eval(S_i, j=ubr, k=k) + P_u_fm_eval(S_i, j=ubr, k=k))
        pm_fur4.append(P_b_pm_eval() + P_u_pm_eval() + P_pm_eval(S_e, j=ubr, k=k))
        pm_ex_ur.append(S_pm_eval(S_e, j=ubr))

    plt.subplot(221)
    plt.plot(unbinding_rate, fm_pm_fur4, "r", label="Total P.M. Fur4")
    plt.plot(unbinding_rate, fm_end_fur4, 'purple', label="Total Endosomal Fur4")
    plt.plot(unbinding_rate, [x + y for x,y in zip(fm_pm_fur4, fm_end_fur4)], "k", label="Total Fur4")
    plt.xlabel("Unbinding Uracil/Fur4 Rate")
    plt.ylabel("Fur4 Concentration")
    plt.title(fr"Full Model Steady State, $S_e={S_e}$")
    plt.legend()

    plt.subplot(222)
    plt.plot(unbinding_rate, fm_ex_ur)
    plt.ylabel("Intracellular Uracil Concentration")
    plt.title(fr"Full Model Steady State, $S_e={S_e}$")
    plt.xlabel("Unbinding Uracil/Fur4 Rate")

    plt.subplot(223)
    plt.plot(unbinding_rate, pm_fur4)
    plt.xlabel("Unbinding Uracil/Fur4 Rate")
    plt.ylabel("Fur4 Concentration")
    plt.title(fr"PM Model Steady State, $S_e={S_e}$")
    plt.ylim((0, 0.1))


    plt.subplot(224)
    plt.plot(unbinding_rate, pm_ex_ur)
    plt.ylabel("Intracellular Uracil Concentration")
    plt.title(fr"PM Steady State, $S_e={S_e}$")
    plt.xlabel("Unbinding Uracil/Fur4 Rate")

    plt.tight_layout()
    plt.show()


def comparison_prod_rate():
    plt.figure(figsize=(10, 10))

    S_e = .01  # fix extracellular uracil
    a = 1
    b = 1
    A_p = 314
    j = 10 ** 2
    K_d = 0.74
    K_m = 2.5
    k = j / K_d
    V_m = 8.8 * 10 ** 3
    V = 523
    W = 32
    # y = 0.000083  alter production rate
    z = .002
    f = 0.1
    A_e = 47
    g = 0.1

    prod_rate = np.linspace(.0000083, .00083)  # order of 10 lower and higher
    fm_pm_fur4, fm_end_fur4, fm_ex_ur = [], [], []
    pm_fur4, pm_ex_ur = [], []
    for pr in prod_rate:
        dSdt_ur = lambda S: -(k / W) * S * (
                    (A_p / V) * P_fm_eval(S, S_e, y=pr) + (A_e / V) * E_fm_eval(S, y=pr)) + (
                                    j + a) * (
                                    (A_p / V) * P_b_fm_eval(S, y=pr) + (A_e / V) * E_b_fm_eval(S, y=pr)) - V_m * S / (
                                    V * (K_m + S))
        S_i = opt.bisect(dSdt_ur, 0, 50)
        fm_ex_ur.append(S_i)
        fm_end_fur4.append(E_fm_eval(S_i, y=pr) + E_b_fm_eval(S_i, y=pr) + E_u_fm_eval(y=pr))
        fm_pm_fur4.append(P_fm_eval(S_i, S_e, y=pr) + P_b_fm_eval(S_i, y=pr) + P_u_fm_eval(S_i, y=pr))
        pm_fur4.append(P_b_pm_eval(y=pr) + P_u_pm_eval(y=pr) + P_pm_eval(S_e, y=pr))
        pm_ex_ur.append(S_pm_eval(S_e, y=pr))

    plt.subplot(221)
    plt.plot(prod_rate, fm_pm_fur4, "r", label="Total P.M. Fur4")
    plt.plot(prod_rate, fm_end_fur4, 'purple', label="Total Endosomal Fur4")
    plt.plot(prod_rate, [x + y for x, y in zip(fm_pm_fur4, fm_end_fur4)], "k", label="Total Fur4")
    plt.xlabel("Fur4 Production Rate")
    plt.ylabel("Fur4 Concentration")
    plt.title(fr"Full Model Steady State, $S_e={S_e}$")
    plt.legend()

    plt.subplot(222)
    plt.plot(prod_rate, fm_ex_ur)
    plt.ylabel("Intracellular Uracil Concentration")
    plt.title(fr"Full Model Steady State, $S_e={S_e}$")
    plt.xlabel("Fur4 Production Rate")

    plt.subplot(223)
    plt.plot(prod_rate, pm_fur4)
    plt.xlabel("Fur4 Production Rate")
    plt.ylabel("Fur4 Concentration")
    plt.title(fr"PM Model Steady State, $S_e={S_e}$")

    plt.subplot(224)
    plt.plot(prod_rate, pm_ex_ur)
    plt.ylabel("Intracellular Uracil Concentration")
    plt.title(fr"PM Steady State, $S_e={S_e}$")
    plt.xlabel("Fur4 Production Rate")

    plt.tight_layout()
    plt.show()


def compare_degredation_rate():
    plt.figure(figsize=(10, 10))

    S_e = 1  # fix extracellular uracil
    a = 1
    b = 1
    A_p = 314
    j = 10 ** 2
    K_d = 0.74
    K_m = 2.5
    k = j / K_d
    V_m = 8.8 * 10 ** 3
    V = 523
    W = 32
    y = 0.000083
    # z = .002  alter degredation rate
    f = 0.1
    A_e = 47
    g = 0.1

    deg_rate = np.linspace(0.0002, 0.02, 150)  # order of 10 lower and higher
    fm_pm_fur4, fm_end_fur4, fm_ex_ur = [], [], []
    pm_fur4, pm_ex_ur = [], []
    for dr in deg_rate:
        dSdt_ur = lambda S: -(k / W) * S * (
                (A_p / V) * P_fm_eval(S, S_e, z=dr) + (A_e / V) * E_fm_eval(S, z=dr)) + (
                                    j + a) * (
                                    (A_p / V) * P_b_fm_eval(S, z=dr) + (A_e / V) * E_b_fm_eval(S, z=dr)) - V_m * S / (
                                    V * (K_m + S))
        S_i = opt.bisect(dSdt_ur, 0, 50)
        fm_ex_ur.append(S_i)
        fm_end_fur4.append(E_fm_eval(S_i, z=dr) + E_b_fm_eval(S_i, z=dr) + E_u_fm_eval(z=dr))
        fm_pm_fur4.append(P_fm_eval(S_i, S_e, z=dr) + P_b_fm_eval(S_i, z=dr) + P_u_fm_eval(S_i, z=dr))
        pm_fur4.append(P_b_pm_eval() + P_u_pm_eval(z=dr) + P_pm_eval(S_e))
        pm_ex_ur.append(S_pm_eval(S_e))

    plt.subplot(221)
    plt.plot(deg_rate, fm_pm_fur4, "r", label="Total P.M. Fur4")
    plt.plot(deg_rate, fm_end_fur4, 'purple', label="Total Endosomal Fur4")
    plt.plot(deg_rate, [x + y for x, y in zip(fm_pm_fur4, fm_end_fur4)], "k", label="Total Fur4")
    plt.xlabel("Fur4 Degradation Rate")
    plt.ylabel("Fur4 Concentration")
    plt.title(fr"Full Model Steady State, $S_e={S_e}$")
    plt.legend()
    plt.locator_params(axis='x', nbins=5)

    plt.subplot(222)
    plt.plot(deg_rate, fm_ex_ur)
    plt.ylabel("Intracellular Uracil Concentration")
    plt.title(fr"Full Model Steady State, $S_e={S_e}$")
    plt.xlabel("Fur4 Degradation Rate")
    plt.locator_params(axis='x', nbins=5)

    plt.subplot(223)
    plt.plot(deg_rate, pm_fur4)
    plt.xlabel("Fur4 Degradation Rate")
    plt.ylabel("Fur4 Concentration")
    plt.title(fr"PM Model Steady State, $S_e={S_e}$")
    plt.locator_params(axis='x', nbins=5)

    plt.subplot(224)
    plt.plot(deg_rate, pm_ex_ur)
    plt.ylabel("Intracellular Uracil Concentration")
    plt.title(fr"PM Steady State, $S_e={S_e}$")
    plt.xlabel("Fur4 Degradation Rate")
    plt.locator_params(axis='x', nbins=5)

    plt.tight_layout()
    plt.show()


def time_series_uracil_jump():
    pass




if __name__ == '__main__':
    # comparison_extracellular_uracil()
    # comparison_ubiq_rate()
    # comparison_recycling_rate()
    # comparison_endocytosis_rate()
    # comparison_deubiq_rate()
    # comparison_unbinding_rate()
    # comparison_prod_rate()
    compare_degredation_rate()
