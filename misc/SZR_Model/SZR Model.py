#Importing Initial packages
from matplotlib import pyplot as plt
import numpy as np

#Defining System of Differential Equations
def func(SZR,t,alpha, beta, gamma, delta, Pi):
    S,Z,R = SZR
    dSdt = Pi-beta*S*Z-delta*S
    dZdt = beta*S*Z+gamma*R-alpha*S*Z
    dRdt = delta*S+alpha*S*Z-gamma*R
    dydt = [dSdt, dZdt, dRdt]
    return dydt


#Setting up initial values and time interval
y0 = [500,0,5]
t = np.linspace(0,10,101)

#Defining variables
Pi = 0
alpha = 0.005
beta = 0.0095
gamma = 0.0001
delta = 0.0001

#Importing ODE Solver
from scipy.integrate import odeint
#Solving system of ODE's
sol = odeint(func, y0, t, args=(alpha, beta, gamma, delta, Pi))

#Plotting Solutions
plt.plot(t, sol[:,0], 'b', label = "Survivors")
plt.plot(t, sol[:,1], 'r', label = "Zombies")
plt.plot(t, sol[:,2], 'g', label = "Removed")
plt.legend()
plt.xlabel('t')
plt.grid()
plt.show()



#Importing Solver for system of homogenous equations
from sympy import solve
#Importing symbolic variables
from sympy.abc import alpha, beta, gamma, S, Z, R

#Solving System of equations
sol = solve([-beta*S*Z,
             beta*S*Z + gamma*R -alpha*S*Z,
             alpha*S*Z - gamma*R],
             (S,Z,R))
#Printing solution
print(sol)