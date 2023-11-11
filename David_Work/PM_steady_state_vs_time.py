from matplotlib import pyplot as plt
import numpy as np

constant = 1000

S_e = np.linspace(0.001,1,constant)
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




#Steady State 1
P = (-S_e*V*W*a*y - S_e*V*W*j*y - S_e*W*a*v_m + np.sqrt(S_e*W*(4*K_m*V*a**2*v_m*y + 4*K_m*V*a*j*v_m*y + S_e*V**2*W*a**2*y**2 + 2*S_e*V**2*W*a*j*y**2 + S_e*V**2*W*j**2*y**2 - 2*S_e*V*W*a**2*v_m*y - 2*S_e*V*W*a*j*v_m*y + S_e*W*a**2*v_m**2)))/(2*S_e*V*a*k*(K_m - S_e*W))
P_b = y/a
P_u = y/z
S = (S_e*W*(V*a*y + V*j*y - a*v_m) + np.sqrt(S_e*W*(4*K_m*V*a**2*v_m*y + 4*K_m*V*a*j*v_m*y + S_e*V**2*W*a**2*y**2 + 2*S_e*V**2*W*a*j*y**2 + S_e*V**2*W*j**2*y**2 - 2*S_e*V*W*a**2*v_m*y - 2*S_e*V*W*a*j*v_m*y + S_e*W*a**2*v_m**2)))/(2*a*v_m)

plt.plot(S_e,P,'b',label = "Plasma Membrane Ground State")
plt.plot(S_e,[P_b for _ in range(constant)],'g',label = "Plasma Membrane Bound")
plt.plot(S_e,[P_u for _ in range(constant)],'r',label = "Plasma Membrane Ubiquitinated")
plt.plot(S_e,S,'k',label = "Intracellular Uracil")
plt.legend()
plt.xlabel("Extracellular Uracil")
plt.show()