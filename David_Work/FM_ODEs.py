#Importing Packages
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import odeint


#Defining ODE 
def func(FM,t,a,A_e,A_p,b,f,g,j,k,K_m,S_e,v_m,V,W,y,z):
    P,P_b,P_u,E,E_b,E_u,S = FM
    dPdt = y - (k*S_e + (k/W)*S)*P + j*P_b + f*(A_e/A_p)*E
    dP_bdt = (k*S_e + (k/W)*S)*P - j*P_b - a*P_b
    dP_udt = a*P_b - g*(A_p/A_e)*P_u
    dEdt = b*E_u + j*E_b - (k/W)*S*E - f*(A_e/A_p)*E
    dE_bdt = (k/W)*S*E - j*E_b - a*E_b
    dE_udt = g*(A_p/A_e)*P_u + a*E_b - b*E_u - z*E_u
    dSdt = -(k/W)*S*((A_p/V)*P+(A_e/V)*E) + (j+a)*((A_p/V)*P_b+(A_e/V)*E_b) - (v_m*S)/(V*(K_m+S))
    dydt = [dPdt, dP_bdt, dP_udt, dEdt, dE_bdt, dE_udt, dSdt]
    return dydt

#Defining Variables
A_e = 47
A_p = 314
a = 1
b = 1
f = .1
g = .1
j = 10**2
K_m = 2.5
K_d = .74
k = j/K_d
y = .000083
W = 32
v_m = 8.8 * 10**3
V = 523
z = .002



#Defining Initial Condition, Timespan, and rates to test
y0 = [1,0,0,0,0,0,0]
rates = [0.01,0.2,0.5,0.5,.7,1,5,10,20]
rates = [.7,1,5,10,20]
t = np.linspace(0,50000,100000)
#Going through each rate
for i in range(3):
    #Defining Extracellular Uracil
    S_e = rates[i]
    #Defining subplot
    plt1 = plt.subplot(int(f"23{i+1}"))
    #Finding solution to ODE
    sol = odeint(func,y0,t,args = (a,A_e,A_p,b,f,g,j,k,K_m,S_e,v_m,V,W,y,z))
    #Plotting solution
    plt1.plot(t,sol[:,0],'b',label = "PM Ground State")
    plt1.plot(t,sol[:,1],'g',label = "PM Bound")
    plt1.plot(t,sol[:,2],'r',label = "PM Ubiquitinated")
    plt1.plot(t,sol[:,3],'b--',label = "EM Ground State")
    plt1.plot(t,sol[:,4],'g--',label = "EM Bound")
    plt1.plot(t,sol[:,5],'r--',label = "EM Ubiquitinated")
    plt1.set_xlabel("Time")
    plt.legend()
    plt.grid()
    plt1.set_title(f"S_e = {S_e}")
    plt2 = plt.subplot(int(f"23{i+4}"))
    plt2.plot(t,sol[:,6],'k',label = "Intracellullar Uracil")
    plt2.set_xlabel("Time")
    plt2.set_title("Intracellullar Uracil")
    #Setting graph layout
    plt.legend()
    plt.grid()
    plt1.set_xlabel("Time")
    print(sol[:,0][-1])


plt.tight_layout()
plt.show()
