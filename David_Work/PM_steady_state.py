from matplotlib import pyplot as plt
import numpy as np

constant = 1000
S_e = np.linspace(0.001,.05,constant)

A_p = 314
a = 1
j = 10**2
K_d = 0.74
K_m = 2.5
k = j/K_d
v_m = 8.8 * 10**3
V = 523
W = 32
y = 0.000083
z = .002


P = (-A_p*S_e*W*a*y - A_p*S_e*W*j*y - S_e*W*a*v_m + np.sqrt(S_e*W*(A_p**2*S_e*W*a**2*y**2 + 2*A_p**2*S_e*W*a*j*y**2 + A_p**2*S_e*W*j**2*y**2 + 4*A_p*K_m*a**2*v_m*y + 4*A_p*K_m*a*j*v_m*y - 2*A_p*S_e*W*a**2*v_m*y - 2*A_p*S_e*W*a*j*v_m*y + S_e*W*a**2*v_m**2)))/(2*A_p*S_e*a*k*(K_m - S_e*W))
P_b = y/a
P_u = y/z
S = (S_e*W*(A_p*a*y + A_p*j*y - a*v_m) + np.sqrt(S_e*W*(A_p**2*S_e*W*a**2*y**2 + 2*A_p**2*S_e*W*a*j*y**2 + A_p**2*S_e*W*j**2*y**2 + 4*A_p*K_m*a**2*v_m*y + 4*A_p*K_m*a*j*v_m*y - 2*A_p*S_e*W*a**2*v_m*y - 2*A_p*S_e*W*a*j*v_m*y + S_e*W*a**2*v_m**2)))/(2*a*v_m)

#Steady State
P = (-S_e*V*W*a*y - S_e*V*W*j*y - S_e*W*a*v_m + np.sqrt(S_e*W*(4*K_m*V*a**2*v_m*y + 4*K_m*V*a*j*v_m*y + S_e*V**2*W*a**2*y**2 + 2*S_e*V**2*W*a*j*y**2 + S_e*V**2*W*j**2*y**2 - 2*S_e*V*W*a**2*v_m*y - 2*S_e*V*W*a*j*v_m*y + S_e*W*a**2*v_m**2)))/(2*S_e*V*a*k*(K_m - S_e*W))
P_b = y/a
P_u = y/z
S = (S_e*W*(V*a*y + V*j*y - a*v_m) + np.sqrt(S_e*W*(4*K_m*V*a**2*v_m*y + 4*K_m*V*a*j*v_m*y + S_e*V**2*W*a**2*y**2 + 2*S_e*V**2*W*a*j*y**2 + S_e*V**2*W*j**2*y**2 - 2*S_e*V*W*a**2*v_m*y - 2*S_e*V*W*a*j*v_m*y + S_e*W*a**2*v_m**2)))/(2*a*v_m)

fig,ax = plt.subplots(1,2)
ax[0].plot(S_e,P,'b',label = "Ground State")
ax[0].plot(S_e,[P_b for _ in range(constant)],'g',label = "Bound")
ax[0].plot(S_e,[P_u for _ in range(constant)],'r',label = "Ubiquitinated")
ax[0].set_xlabel("Extracellular Uracil")
ax[0].set_title("Fur4 Steady States")
ax[0].legend()

ax[1].plot(S_e,S,'k')
ax[1].set_title("Intracellular Uracil")
ax[1].set_xlabel("Extracellular Uracil")

plt.tight_layout()
plt.show()



fig,ax = plt.subplots(1,2)

ax[0].plot(S_e,P+P_b+P_u,'b')
ax[0].set_xlabel("Extracellular Uracil")
ax[0].set_ylabel("Total FUR4")
ax[0].set_title("Total FUR4 vs. Extracellular Uracil")
ax[1].plot(S_e,S,'r')
ax[1].set_xlabel("Extracellular Uracil")
ax[1].set_ylabel("Intracellular Uracil")
ax[1].set_title("Intracellular Uracil vs. Extracellular Uracil")

plt.tight_layout()
plt.show()