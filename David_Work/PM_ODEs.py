from sqlite3 import Time
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import odeint

def func(PM,t,a,j,k,K_m,S_e,v_m,V,W,y,z):
    P,P_b,P_u,S = PM
    dPdt = y - k*S_e*P - k/W*S*P + j*P_b
    dP_bdt = k*S_e*P + k/W*S*P - j*P_b - a*P_b
    dP_udt = a*P_b - z*P_u
    dSdt = -k/W*S*P + (j+a)*P_b - v_m*S/(V*(K_m + S))
    dydt = [dPdt, dP_bdt, dP_udt, dSdt]
    return dydt

y0 = [50,0,0,0]
t = np.linspace(0,100,10000)

v_m = 8.8 * 10**3
y = 0.000083
W = 32
a = 1
j = 10**2
K_d = 0.74
k = j/K_d
K_m = 2.5
V = 523
z = .002

S_e = .01


sol = odeint(func,y0,t,args = (a,j,k,K_m,S_e,v_m,V,W,y,z))
plt.plot(t,sol[:,0],'b',label = "Ground State Fur4")
plt.plot(t,sol[:,1],'g',label = "Bound Fur4")
plt.plot(t,sol[:,2],'r',label = "Ubiquitinated Fur4")
plt.plot(t,sol[:,3],'k',label = "Intracellullar Uracil")
plt.xlabel("Time")
plt.legend()
plt.grid()
plt.show()
