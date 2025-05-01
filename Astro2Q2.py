import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid as trapz

# Defining Constants
L = 1
r_vals = np.linspace(0.1 * L, L, 1000)
A = 1
B = 1

# (a): j propto 1/r^2, alpha propto r 
j_a = B / r_vals**2
alpha_a = A * r_vals
tau_a = trapz(alpha_a, r_vals, initial=0)  # Optical depth from 0.1L to r 
brightness_a = trapz(j_a * np.exp(-tau_a), r_vals, initial=0)

# (b): j propto ln(r/L), alpha propto r*exp(-r/L) 
j_b = B * np.log(r_vals / L)
alpha_b = A * r_vals * np.exp(-r_vals / L)
tau_b = trapz(alpha_b, r_vals, initial=0)
brightness_b = trapz(j_b * np.exp(-tau_b), r_vals, initial=0)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs[0, 0].plot(r_vals, tau_a)
axs[0, 0].set_title("Optical Depth (a)")
axs[0, 0].set_xlabel("r")
axs[0, 0].set_ylabel("Tau(r)")

axs[0, 1].plot(r_vals, brightness_a)
axs[0, 1].set_title("Brightness (a)")
axs[0, 1].set_xlabel("r")
axs[0, 1].set_ylabel("I_ν(r)")

axs[1, 0].plot(r_vals, tau_b)
axs[1, 0].set_title("Optical Depth (b)")
axs[1, 0].set_xlabel("r")
axs[1, 0].set_ylabel("Tau(r)")

axs[1, 1].plot(r_vals, brightness_b)
axs[1, 1].set_title("Brightness (b)")
axs[1, 1].set_xlabel("r")
axs[1, 1].set_ylabel("I_ν(r)")

plt.tight_layout()
plt.show()

