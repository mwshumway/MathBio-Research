from typing import List, Any

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt
from Comparative_Analysis import P_pm_eval, P_b_pm_eval, P_u_pm_eval, S_pm_eval


# Define Steady State equations with default parameters
def eval_P(S, S_e, a=1, b=1, j=100, k=100 / 0.74, W=32, y=0.000083, z=0.002, f=0.1):
    """returns the steady state of P"""
    return (
                S * W * a ** 2 * k * y * z + S * W * a * j * k * y * z + W ** 2 * a ** 2 * b * f * y + W ** 2 * a ** 2 * f * y * z + 2 * W ** 2 * a * b * f * j * y + 2 * W ** 2 * a * f * j * y * z +
                W ** 2 * b * f * j ** 2 * y + W ** 2 * f * j ** 2 * y * z) / (
                S ** 2 * a ** 2 * k ** 2 * z + S * S_e * W * a ** 2 * k ** 2 * z)


def eval_Pb(S, a=1, b=1, j=100, k=100 / 0.74, W=32, y=0.000083, z=0.002, f=0.1):
    """returns the steady state of P_b"""
    return (S * a * k * y * z + W * a * b * f * y + W * a * f * y * z + W * b * f * j * y + W * f * j * y * z) / (
                S * a ** 2 * k * z)


def eval_Pu(S, a=1, b=1, j=100, k=100 / 0.74, W=32, y=0.000083, z=0.002, f=0.1, g=0.1):
    """returns the steady state of P_u"""
    return (S * a * k * y * z + W * a * b * f * y + W * a * f * y * z + W * b * f * j * y + W * f * j * y * z) / (
                S * a * g * k * z)


def eval_E(S, A_p=314, y=0.000083, z=0.002, A_e=47, W=32, a=1, b=1, j=100, k=100 / .74):
    """returns the steady state of E"""
    return (A_p * W * a * b * y + A_p * W * a * y * z + A_p * W * b * j * y + A_p * W * j * y * z) / (
                A_e * S * a * k * z)


def eval_Eb(A_p=314, b=1, y=0.000083, z=.002, a=1, A_e=47):
    """returns the steady state of E_b"""
    return (A_p * b * y + A_p * y * z) / (A_e * a * z)


def eval_Eu(A_p=314, A_e=47, y=0.000083, z=.002):
    """returns the steady state of E_e"""
    return A_p * y / (A_e * z)


# ----------------------------------------------------------------------------------------------------------------------
# Plotting Code to Make Comparisons

def extracellular_uracil(plot_PM=False):
    """Creates and shows two plots one showing Fur4 and another with Intracellular Uracil concentrations.
    In these plots, we vary extracellular uracil."""

    # Define relevant parameters
    a, b, A_p, j, K_d, K_m = 1, 1, 314, 100, 0.74, 2.5
    k, V_m, V, W, y, A_e, g = j / K_d, 8.8 * 10 ** 3, 523, 32, 0.000083, 47, 0.1

    Se = np.linspace(.01, 1, 500)  # valid parameter range of S_e, experimentally S_e = 1

    total_Fur4, PM_Fur4, END_Fur4, Uracil = [], [], [], []  # initialize lists to hold data
    if plot_PM:
        total_Fur4_PM, Uracil_PM = [], []
    for se in Se:
        dSdt = lambda S: -(k / W) * S * ((A_p / V) * eval_P(S, S_e=se) + (A_e / V) * eval_E(S)) + (j + a) * (
                (A_p / V) * eval_Pb(S) + (A_e / V) * eval_Eb()) - V_m * S / (V * (K_m + S))

        S_i = opt.bisect(dSdt, .000001, 50)  # find the root

        Uracil.append(S_i)
        pm = eval_P(S_i, S_e=se) + eval_Pb(S_i) + eval_Pu(S_i)  # total plasma membrane Fur4
        PM_Fur4.append(pm)
        end = eval_E(S_i) + eval_Eu() + eval_Eb()  # total endosomal fur4
        END_Fur4.append(end)
        total_Fur4.append(pm + end)

        if plot_PM:
            total_Fur4_PM.append(P_pm_eval(S_e=se) + P_u_pm_eval() + P_b_pm_eval())
            Uracil_PM.append(S_pm_eval(S_e=se))

    # Plot
    plt.figure(figsize=(10, 5))

    plt.subplot(121)

    if plot_PM:
        plt.plot(Se, total_Fur4, "k", label="Full Model Total Fur4", linewidth=5)
        plt.plot(Se, total_Fur4_PM, 'r--', label="PM Only Total Fur4", linewidth=5)
        plt.title("Total Fur4 vs. Extracellular Uracil")
    else:
        plt.plot(Se, PM_Fur4, 'r', label="Total P.M. Fur4", linewidth=5)
        plt.plot(Se, END_Fur4, 'purple', label="Total Endosomal Fur4", linewidth=5)
        plt.plot(Se, total_Fur4, "k", label="Total Fur4", linewidth=5)
        plt.title("Full Model Steady States", fontsize=20)

    plt.legend()
    plt.xlabel("Extracellular Uracil", fontsize=15)
    plt.ylabel("Total Fur4", fontsize=15)
    plt.locator_params(axis='both', nbins=4)

    plt.subplot(122)
    if plot_PM:
        plt.plot(Se, Uracil, 'k', linewidth=5, label='Full Model Uracil')
        plt.plot(Se, Uracil_PM, 'r--', linewidth=5, label='PM Only Uracil')
        plt.legend()
    else:
        plt.plot(Se, Uracil, linewidth=5)
    plt.title("Intra- vs. Extracellular Uracil", fontsize=20)
    plt.xlabel("Extracellular Uracil", fontsize=15)
    plt.ylabel("Intracellular Uracil", fontsize=15)
    plt.locator_params(axis='both', nbins=4)

    plt.suptitle("Varying Extracellular Uracil", fontsize=20)
    plt.tight_layout()
    plt.show()


def recycling_rate():
    """Creates and shows two plots one showing Fur4 and another with Intracellular Uracil concentrations.
        In these plots, we vary Recycling Rate."""

    # Define relevant parameters
    S_e, a, b, A_p, j, K_d, K_m = .1, 1, 1, 314, 100, 0.74, 2.5
    k, V_m, V, W, y, A_e, g = j / K_d, 8.8 * 10 ** 3, 523, 32, 0.000083, 47, 0.1

    F = np.linspace(.01, 1, 500)  # valid parameter range of recycling rate, our default is .1

    total_Fur4, PM_Fur4, END_Fur4, Uracil = [], [], [], []  # initialize lists to hold data
    for f in F:
        # define dS/dt steady state to use bisection method on to solve for steady state
        # dSdt = lambda S: -(k/W)*S*(A_p*eval_P(S, S_e=se) / V + A_e*eval_E(S) / V) + (j+a) * (A_p*eval_Pb(S) / V +
        #                                                             A_e * eval_Eb() / V) - (V_m*S) / (V*(K_m + S))
        dSdt = lambda S: -(k / W) * S * ((A_p / V) * eval_P(S, S_e=S_e, f=f) + (A_e / V) * eval_E(S)) + (j + a) * (
                (A_p / V) * eval_Pb(S, f=f) + (A_e / V) * eval_Eb(S)) - V_m * S / (V * (K_m + S))

        S_i = opt.bisect(dSdt, .000001, 50)  # find the root

        Uracil.append(S_i)
        pm = eval_P(S_i, S_e=S_e, f=f) + eval_Pb(S_i, f=f) + eval_Pu(S_i, f=f)  # total plasma membrane Fur4
        PM_Fur4.append(pm)
        end = eval_E(S_i) + eval_Eu() + eval_Eb()  # total endosomal fur4
        END_Fur4.append(end)
        total_Fur4.append(pm + end)

    # Plot
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.plot(F, PM_Fur4, 'r', label="Total P.M. Fur4", linewidth=5)
    plt.plot(F, END_Fur4, 'purple', label="Total Endosomal Fur4", linewidth=5)
    plt.plot(F, total_Fur4, "k", label="Total Fur4", linewidth=5)
    plt.title("Full Model Steady States", fontsize=20)
    plt.legend()
    plt.xlabel("Recycling Rate", fontsize=15)
    plt.ylabel("Total Fur4", fontsize=15)
    plt.locator_params(axis='both', nbins=4)
    plt.ylim((0, 20))

    plt.subplot(122)
    plt.plot(F, Uracil, linewidth=5)
    plt.title("Full Model Steady States", fontsize=20)
    plt.xlabel("Recycling Rate", fontsize=15)
    plt.ylabel("Intracellular Uracil", fontsize=15)
    plt.locator_params(axis='both', nbins=4)

    plt.suptitle("Varying Recycling Rates", fontsize=20)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    extracellular_uracil(plot_PM=True)
    # recycling_rate()
