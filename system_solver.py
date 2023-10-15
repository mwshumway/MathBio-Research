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
    equations = [-1 * beta * S * Z,
                 beta * S * Z + zeta * R - alpha * S * Z,
                 alpha * S * Z - zeta * R
                 ]

    sol = solve(equations, [S, Z, R])
    return sol


def SIZR_model():
    equations = [
        -1 * beta * S * Z,
        -1 * I * rho + beta * S * Z,
        I * rho - alpha * S * Z + R * zeta,
        alpha * S * Z - R * zeta
    ]
    sol = solve(equations, [S, I, Z, R])
    return sol


def short_outbreak():
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
    equations = [
        -1*beta*S*Z + c*Z,
        beta*S*Z - rho*I,
        rho*I + zeta*R - alpha*S*Z - c*Z,
        alpha*S*Z - zeta*R
    ]
    sol = solve(equations, [S, I, Z, R])
    return sol


res = {"Basic Model": basic_model(),
       "SIZR Model": SIZR_model(),
       "Short Outbreak": short_outbreak(),
       "Model with Treatment": Model_with_Treatment()}

# formatting to print
for n, o in res.items():
    print(n + ":", o)
