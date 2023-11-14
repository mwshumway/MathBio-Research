"""Matthew Shumway
Note:
    sympy.solve takes:
    params:
            equations (list) : a list of equations implicitly set to 0. This is very helpful in our case. But, for
                                example, 2x + 1 = 4 would be inputted as 2x - 3 instead.
            vars to solve (list) : a list of variables to solve for

    returns: list(tup) : a list of tuples giving the possible solutions. Tuples are outputted in the order of variables
                            given as an argument in sympy.solve()
    """

from sympy import solve
from sympy.abc import alpha, beta, zeta, S, Z, R, I, rho, k, Q, sigma, gamma, c


def basic_model():
    """System of Equations given in Section 2."""
    equations = [-1 * beta * S * Z,
                 beta * S * Z + zeta * R - alpha * S * Z,
                 alpha * S * Z - zeta * R
                 ]

    sol = solve(equations, [S, Z, R])
    return sol

print(basic_model())

def SIZR_model():
    """System of Equations given in section 3."""
    equations = [
        -1 * beta * S * Z,
        -1 * I * rho + beta * S * Z,
        I * rho - alpha * S * Z + R * zeta,
        alpha * S * Z - R * zeta
    ]
    sol = solve(equations, [S, I, Z, R])
    return sol


def short_outbreak():
    """System of Equations given in section 4. The model with Quarantine."""
    equations = [
        -1 * beta * S * Z,
        beta * S * Z - rho * I - k * I,
        rho * I, zeta * R - alpha * S * Z - sigma * Z,
        alpha * S * Z - zeta * R + gamma * Q,
        k * I + sigma * Z - gamma * Q
    ]
    sol = solve(equations, [S, I, Z, R, Q])
    return sol


def Model_with_Treatment():
    """System given in section 5. Model with Treatment."""
    equations = [
        -1 * beta * S * Z + c * Z,
        beta * S * Z - rho * I,
        rho * I + zeta * R - alpha * S * Z - c * Z,
        alpha * S * Z - zeta * R
    ]
    sol = solve(equations, [S, I, Z, R])
    return sol


# Store the results in a dictionary
res = {"Basic Model": basic_model(),
       "SIZR Model": SIZR_model(),
       "Short Outbreak": short_outbreak(),
       "Model with Treatment": Model_with_Treatment()}
#
#
# # formatting to print
# for n, o in res.items():
#     print(n + ":", o)

# These are the outputs, if you don't want to run them:

# Basic Model: [(0, Z, 0), (S, 0, 0)]
# SIZR Model: [(0, 0, Z, 0), (S, 0, 0, 0)]
# Short Outbreak: [(0, 0, Z, Z*sigma/zeta, Z*sigma/gamma), (S, 0, 0, 0, 0)]
# Model with Treatment: [(S, 0, 0, 0), (c/beta, Z*c/rho, Z, Z*alpha*c/(beta*zeta))]
import sys

if __name__ == '__main__':
    print(f'Simple SZR: {basic_model()}')
