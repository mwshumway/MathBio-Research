from sympy.solvers import solve
from sympy.abc import b, y, k, v, P, W ,S ,j, z, a, K, V, f, E, g 
from sympy import symbols
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint

S_e, P_b, P_u,A_e, A_p, E_b, E_u = symbols("S_e P_b P_u A_e A_p E_b E_u")

def plas_mem_steady(): #solves for steady state for just plasma membrane
    dPdt = y - k*S_e*P - k/W*S*P + j*P_b
    dP_bdt = k*S_e*P + (k/W)*S*P - j*P_b - a*P_b
    dP_udt = a*P_b - z*P_u
    #dSdt = -k/W*S*P+(j+a)*P_b-(v*S)/(V*(K+S))
    
    sol = solve([dPdt,dP_bdt,dP_udt],[P, P_b, P_u])
    return sol

def full_model_steady(): # dolves steady state for full model
    dPdt = y - k*S_e*P - k/W*S*P + j*P_b + f*(A_e/A_p)*E
    dP_bdt = k*S_e*P + (k/W)*S*P - j*P_b - a*P_b
    dP_udt = a*P_b - g*P_u
    dE_bdt = g*(A_p/A_e)*P_u - b*E_u + a*E_b - z*E_u
    dE_udt = b*E_u- (k/W)*S*E + j*E_b - f*E
    dEdt = (k/W)*S*E - j*E_b - a*E_b

    sol = solve([dPdt,dP_bdt,dP_udt,dEdt,dE_bdt,dE_udt],[P, P_b, P_u, E, E_b, E_u])
    return sol

def plasmamem_numer(y0,t,y,k,j,a,z,S_e,W,V,v_m, K_m):
    P,P_b,P_u,S = y0

    # equations
    dPdt = y - k*S_e*P - k/W*S*P + j*P_b
    dP_bdt = k*S_e*P + (k/W)*S*P - j*P_b - a*P_b
    dP_udt = a*P_b - z*P_u
    dSdt = -k/W*S*P+(j+a)*P_b-(v_m*S)/(V*(K_m+S))
    
    return [dPdt,dP_bdt,dP_udt,dSdt]

def full_numer(y0,t,y,k,j,a,z,S_e,W,V,v_m, K_m,g,f,A_e, A_p,b):
    P,P_b,P_u,E,E_b,E_u,S = y0

    # equations
    dPdt = y - k*S_e*P - k/W*S*P + j*P_b + g*(A_e/A_p)*E
    dP_bdt = k*S_e*P + (k/W)*S*P - j*P_b - a*P_b
    dP_udt = a*P_b - f*P_u
    dE_bdt = g*(A_p/A_e)*P_u - b*E_u + a*E_b - z*E_u
    dE_udt = b*E_u- (k/W)*S*E + j*E_b - g*E
    dEdt = (k/W)*S*E - j*E_b - a*E_b
    dSdt = -k/W*S*(P+E)+(j+a)*(P_b+E_b)-(v_m*S)/(V*(K_m+S))

    
    return [dPdt,dP_bdt,dP_udt,dEdt,dE_bdt,dE_udt,dSdt]

# intial conditions
y0 = [50,0,0,0]
rang = np.linspace(0,20,10000)


# defining variables
a,b,g,f = 1,1,.1,.1
j,K_d,y = 100, .74, .000083 
k,z,W = j/K_d,.002,32
A_p,A_e = 314,47
S_e,V,v_m, K_m = .01,523, 8.8 * 1000, 2.5

# numeric solution for plasma membrane only 
sol_pm = odeint(plasmamem_numer,y0,rang,args = (y,k,j,a,z,S_e,W,V,v_m, K_m))
y0.extend([0,0,0])

# plotting plasma membrane only
plt.plot(rang,sol_pm[:,0],'b',label = "Ground State Fur4")
plt.plot(rang,sol_pm[:,1],'y',label = "Bound Fur4")
plt.plot(rang,sol_pm[:,2],'tab:orange',label = "Ubiquitinated Fur4")
plt.plot(rang,sol_pm[:,3],'tab:pink',label = "Intracellullar Uracil")
plt.xlabel("Time")
plt.legend()
plt.grid()
plt.show()
plt.clf()

# numeric solution for full model
sol_full = odeint(full_numer,y0,rang,args = (y,k,j,a,z,S_e,W,V,v_m, K_m,g,f,A_e, A_p,b))

# plot full model
sol = odeint(full_numer,y0,rang,args = (a,A_e,A_p,b,f,g,j,k,K_m,S_e,v_m,V,W,y,z))
plt.plot(rang,sol_full[:,0],'b',label = "PM Ground State")
plt.plot(rang,sol_full[:,1],'y',label = "PM Bound")
plt.plot(rang,sol_full[:,2],'tab:orange',label = "PM Ubiquitinated")
plt.plot(rang,sol_full[:,3],'b--',label = "EM Ground State")
plt.plot(rang,sol_full[:,4],'y--',label = "EM Bound")
plt.plot(rang,sol_full[:,5],'tab:orange','--',label = "EM Ubiquitinated")
plt.plot(rang,sol_full[:,6],'k',label = "Intracellullar Uracil")
plt.xlabel("Time")
plt.title(f"S_e = {S_e}")
plt.legend()
plt.grid()
plt.show()