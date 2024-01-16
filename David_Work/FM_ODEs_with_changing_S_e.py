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
x = .002
W = 32
v_m = 8.8 * 10**3
V = 523
z = .002



#Defining Initial Condition, Timespan, and rates to test
y0 = [1,0,0,0,0,0,0]
S_e = .5
t1 = np.linspace(0,200000,500000)
#Going through each rate
#Defining subplot
# plt1 = plt.subplot(211)
#Finding solution to ODE
sol1 = odeint(func,y0,t1,args = (a,A_e,A_p,b,f,g,j,k,K_m,S_e,v_m,V,W,y,z))
#Plotting solution
# plt1.plot(t1,sol1[:,0],'b',label = "PM Ground State")
# plt1.plot(t1,sol1[:,1],'g',label = "PM Bound")
# plt1.plot(t1,sol1[:,2],'r',label = "PM Ubiquitinated")
# plt1.plot(t1,sol1[:,3],'b--',label = "EM Ground State")
# plt1.plot(t1,sol1[:,4],'g--',label = "EM Bound")
# plt1.plot(t1,sol1[:,5],'r--',label = "EM Ubiquitinated")
# plt1.set_xlabel("Time")
# plt.legend()
# plt.grid()
# plt1.set_title(f"S_e = {S_e}")

# plt2 = plt.subplot(212)
# plt2.plot(t1,sol1[:,6],'k',label = "Intracellullar Uracil")
# plt2.set_xlabel("Time")
# plt2.set_title("Intracellullar Uracil")
# #Setting graph layout
# plt.legend()
# plt.grid()
# plt1.set_xlabel("Time")
# plt.show()




#Defining Initial Condition, Timespan, and rates to test
y0 = sol1[-1,:]
S_e = .01
t2 = np.linspace(0,200000,500000)
#Finding solution to ODE
sol2 = odeint(func,y0,t2,args = (a,A_e,A_p,b,f,g,j,k,K_m,S_e,v_m,V,W,y,z))

#Defining Initial Condition, Timespan, and rates to test
y0 = sol2[-1,:]
S_e = .5
t3 = np.linspace(0,200000,500000)
#Finding solution to ODE
sol3 = odeint(func,y0,t3,args = (a,A_e,A_p,b,f,g,j,k,K_m,S_e,v_m,V,W,y,z))

sol = np.vstack([sol1,sol2,sol3])
t = np.hstack([t1,t2+200000,t3+400000])
#Plotting solution
plt1 = plt.subplot(231)
plt1.plot(t,sol[:,0],'b',label = "PM Ground State")
plt1.plot(t,sol[:,1],'g',label = "PM Bound")
plt1.plot(t,sol[:,2],'r',label = "PM Ubiquitinated")
plt1.plot(t,sol[:,3],'b--',label = "EM Ground State")
plt1.plot(t,sol[:,4],'g--',label = "EM Bound")
plt1.plot(t,sol[:,5],'r--',label = "EM Ubiquitinated")
plt1.set_xlabel("Time")
# plt1.set(xlim=(199990,200010),ylim=(0,.1),)
# plt.legend()
plt.grid()
plt1.set_title(f"S_e = 0.5 -> 0.01 -> {S_e}")

plt2 = plt.subplot(234)
plt2.plot(t,sol[:,6],'k',label = "Intracellullar Uracil")
plt2.set_xlabel("Time")
plt2.set_title("Intracellullar Uracil")
#Setting graph layout
plt.legend()
plt.grid()
plt2.set_xlabel("Time")

#Plotting solution
arg1=len(t)//3-5
arg2=len(t)//3+75
plt3 = plt.subplot(232)
plt3.plot(t[arg1:arg2],sol[arg1:arg2,0],'b',label = "PM Ground State")
plt3.plot(t[arg1:arg2],sol[arg1:arg2,1],'g',label = "PM Bound")
plt3.plot(t[arg1:arg2],sol[arg1:arg2,2],'r',label = "PM Ubiquitinated")
plt3.plot(t[arg1:arg2],sol[arg1:arg2,3],'b--',label = "EM Ground State")
plt3.plot(t[arg1:arg2],sol[arg1:arg2,4],'g--',label = "EM Bound")
plt3.plot(t[arg1:arg2],sol[arg1:arg2,5],'r--',label = "EM Ubiquitinated")
plt3.set_xlabel("Time")
plt.legend()
plt.grid()
plt3.set(ylim=(0,.1))
plt3.set_title(f"S_e = 0.5 -> 0.01 -> {S_e}")

plt4 = plt.subplot(235)
plt4.plot(t[arg1:arg2],sol[arg1:arg2,6],'k',label = "Intracellullar Uracil")
plt4.set_xlabel("Time")
plt4.set_title("Intracellullar Uracil")
#Setting graph layout
plt.legend()
plt.grid()
plt4.set_xlabel("Time")

#Plotting solution
arg1=2*len(t)//3-5
arg2=2*len(t)//3+45
plt5 = plt.subplot(233)
plt5.plot(t[arg1:arg2],sol[arg1:arg2,0],'b',label = "PM Ground State")
plt5.plot(t[arg1:arg2],sol[arg1:arg2,1],'g',label = "PM Bound")
plt5.plot(t[arg1:arg2],sol[arg1:arg2,2],'r',label = "PM Ubiquitinated")
plt5.plot(t[arg1:arg2],sol[arg1:arg2,3],'b--',label = "EM Ground State")
plt5.plot(t[arg1:arg2],sol[arg1:arg2,4],'g--',label = "EM Bound")
plt5.plot(t[arg1:arg2],sol[arg1:arg2,5],'r--',label = "EM Ubiquitinated")
plt5.set_xlabel("Time")
plt.legend()
plt.grid()
plt5.set(ylim=(0,.8))
plt5.set_title(f"S_e = 0.5 -> {S_e}")

plt6 = plt.subplot(236)
plt6.plot(t[arg1:arg2],sol[arg1:arg2,6],'k',label = "Intracellullar Uracil")
plt6.set_xlabel("Time")
plt6.set_title("Intracellullar Uracil")
#Setting graph layout
plt.legend()
plt.grid()
plt6.set_xlabel("Time")


plt.tight_layout()
plt.show()
