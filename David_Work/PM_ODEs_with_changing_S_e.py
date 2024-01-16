#Importing Packages
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import odeint

#Defining the ODE
def func(PM,t,A_p,a,j,k,K_m,S_e,v_m,V,W,y,z):
    P,P_b,P_u,S = PM
    dPdt = y - k*S_e*P - k/W*S*P + j*P_b
    dP_bdt = k*S_e*P + k/W*S*P - j*P_b - a*P_b
    dP_udt = a*P_b - z*P_u
    dSdt = -k/W*S*(A_p/V)*P + (j+a)*(A_p/V)*P_b +  - v_m*S/(V*(K_m + S))
    dydt = [dPdt, dP_bdt, dP_udt, dSdt]
    return dydt

#Defining Variables
a = 1
A_p = 314
j = 10**2
K_d = 0.74
K_m = 2.5
k = j/K_d
v_m = 8.8 * 10**3
V = 523
W = 32
y = 0.000083
z = .002

#Defining initial Extracellular Uracil
S_e = 1

#Setting Initial Condition and Timespan
y0 = [.1,0,0,0]
t1 = np.linspace(0,5000,10000)

#Solving the ODE
sol1 = odeint(func,y0,t1,args = (A_p,a,j,k,K_m,S_e,v_m,V,W,y,z))

y0 = sol1[-1,:]
t2 = np.linspace(0,5000,10000)
S_e = .001

#Solving the ODE
sol2 = odeint(func,y0,t2,args = (A_p,a,j,k,K_m,S_e,v_m,V,W,y,z))

t_span = np.hstack([t1,t2+5000])
sol_tot = np.vstack([sol1,sol2])

#Plotting the Solution
fig, ax = plt.subplots(3,1)

ax[0].plot(t_span,sol_tot[:,0]+sol_tot[:,1]+sol_tot[:,2])
ax[0].set_title("Total FUR4")

ax[1].plot(t_span,sol_tot[:,0],'b',label = "Ground State Fur4")
ax[1].plot(t_span,sol_tot[:,1],'g',label = "Bound Fur4")
ax[1].plot(t_span,sol_tot[:,2],'r',label = "Ubiquitinated Fur4")
ax[1].set_xlabel("Time")
ax[1].legend()
ax[1].set_title("Individual FUR4")

ax[2].plot(t_span,sol_tot[:,3],'k',label = "Intracellullar Uracil")
ax[2].set_xlabel("Time")
ax[2].set_title("Uracil")

plt.tight_layout()

plt.grid()
plt.show()

#Defining extracellular uracil rates
# rates = [0.01,0.2,0.5,0.5,.7,1,5,10,20]
# #Ranging through the rates
# for i in range(3):
#     #Defining extracellular uracil
#     S_e = rates[i]
#     #Defining Subplot
#     plt1 = plt.subplot(int(f"23{i+1}"))
#     #Solving the ODE
#     sol = odeint(func,y0,t,args = (A_p,a,j,k,K_m,S_e,v_m,V,W,y,z))
#     #Plotting the Solution
#     plt1.plot(t,sol[:,0],'b',label = "Ground State Fur4")
#     plt1.plot(t,sol[:,1],'g',label = "Bound Fur4")
#     plt1.plot(t,sol[:,2],'r',label = "Ubiquitinated Fur4")
#     plt1.set_xlabel("Time")
#     plt.legend()
#     plt.grid()
#     plt1.set_title(f"S_e = {S_e}")
#     plt2 = plt.subplot(int(f"23{i+4}"))
#     plt2.plot(t,sol[:,3],'k',label = "Intracellullar Uracil")
#     plt2.set_xlabel("Time")
#     plt2.set_title("Intracellullar Uracil")
#     #Graph Layout
#     plt.grid()
#     print(sol[:,0][-1])

# plt.tight_layout()
# plt.title("Plasma Membrane Model")
# plt.show()
