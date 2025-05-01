import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid as trapz

L = 1 
r = np.linspace(0.1*L, L, 100)
T = 100  
A = 1
T_b_0 = [20, 1e4]  

# (a) alpha propto r 
alpha_a = A * r
tau_a = trapz(alpha_a, r, initial=0)  

# (b) alpha propto r*exp(-r/L) 
alpha_b = A * r * np.exp(-r / L)
tau_b = trapz(alpha_b, r, initial=0)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

for i, x in enumerate(T_b_0):
    T_b_a = T + (x - T) * np.exp(-tau_a)
    T_b_b = T + (x - T) * np.exp(-tau_b)

    # (a)
    axs[i][0].plot(r, T_b_a, color='blue')
    axs[i][0].set_title(f"(a) α ∝ r, $T_{{b0}}$ = {x}")
    axs[i][0].set_xlabel("r")
    axs[i][0].set_ylabel("T_b (K)")

    # (b)
    axs[i][1].plot(r, T_b_b, color='green')
    axs[i][1].set_title(f"(b) α ∝ r·e⁻ʳ⁄ᴸ, $T_{{b0}}$ = {x}")
    axs[i][1].set_xlabel("r")
    axs[i][1].set_ylabel("T_b (K)")

plt.tight_layout()
plt.show()

plt.tight_layout()
plt.show()
