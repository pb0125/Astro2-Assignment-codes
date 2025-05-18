import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_e, c, pi
from scipy.special import gamma

nu = np.logspace(7, 12, 500)  # in Hz

n_i = 50 * 1e6  # convert cm^-3 to m^-3


alpha = np.pi / 4  # Assuming
sin_alpha = np.sin(alpha)

# Assumed energy limits for the electrons
E_min = 1e6 * e  # 1 MeV in Joules
E_max = 1e11 * e  # 100 GeV in Joules


B_fields = [1e-9, 0.1, 0.1]  # in Tesla
velocities = [0.999 * c, 0.999 * c, 0.1 * c]


L_vals = [1e5, 1e13]  # in meters
 
def power_law_index(nu_val):
    return 2.5 if nu_val < 1e10 else 4.0


def compute_C(p):
    denom = E_min**(-(p-1)) - E_max**(-(p-1))
    return n_i * (p - 1) / denom


def alpha_nu(nu, B, p, C):
    factor1 = (np.sqrt(3) * e**3) / (8 * pi * m_e)
    factor2 = ((3 * e) / (2 * pi * m_e**3 * c**5))**(p / 2)
    factor3 = (B * sin_alpha)**((p + 2) / 2)
    G1 = gamma((3 * p + 2) / 12)
    G2 = gamma((3 * p + 22) / 12)
    return factor1 * factor2 * C * factor3 * G1 * G2 * nu**(-(p + 4) / 2)


def compute_tau_nu(nu_array, B, L):
    tau_vals = []
    for nu_i in nu_array:
        p = power_law_index(nu_i)
        C = compute_C(p)
        alpha_val = alpha_nu(nu_i, B, p, C)
        tau_vals.append(alpha_val * L)
    return np.array(tau_vals)


plt.figure(figsize=(10, 6))

for i in range(3):  
    for L in L_vals:
        tau = compute_tau_nu(nu, B_fields[i], L)
        label = f"B={B_fields[i]:.0e} T, L={L:.0e} m"
        plt.loglog(nu, tau, label=label)


plt.xlabel('Frequency (Hz)')
plt.ylabel('Optical Depth')
plt.title('Synchrotron Optical Depth vs Frequency')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig('optical_depth.png', dpi=300)
plt.show()
