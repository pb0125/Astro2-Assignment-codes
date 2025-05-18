import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_e, c, pi
from scipy.special import gamma


nu = np.logspace(6, 12, 1000)  
omega = 2 * pi * nu

alpha = pi / 4  # Assuming for calculation
sin_alpha = np.sin(alpha)


p1 = 2.5
p2 = 4.0

C_norm = 1.0 # Constant C in the power expression (setting to 1 for normalization)


def synchrotron_power(omega, B, p):
    prefactor = (np.sqrt(3) * e**3 * C_norm * B * sin_alpha) / (2 * pi * m_e * c**2 * (p + 1))
    gamma_term = gamma(p/4 + 19/12) * gamma(p/4 - 1/12)
    power_law_term = (m_e * c * omega / (3 * e * B * sin_alpha))**(-(p - 1)/2)
    return prefactor * gamma_term * power_law_term


def polarization_fraction(p):
    return (p + 1) / (p + 7/3)


cases = {
    'Case 1: B=1e-5 G, v=0.999c': 1e-5,
    'Case 2: B=1e6 G, v=0.999c': 1e6,
    'Case 3: B=1e6 G, v=0.1c': 1e6
}


powers = {}
polarizations = {}

for label, B in cases.items():
    power = np.piecewise(nu,
                         [nu <= 1e10, nu > 1e10],
                         [lambda f: synchrotron_power(2 * pi * f, B, p1),
                          lambda f: synchrotron_power(2 * pi * f, B, p2)])
    powers[label] = power / np.max(power)  # Normalizing for comparison

    
    pol_frac = np.piecewise(nu,
                            [nu <= 1e10, nu > 1e10],
                            [polarization_fraction(p1), polarization_fraction(p2)])
    polarizations[label] = pol_frac * 100  # converting to percentage


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)


for label in powers:
    ax1.loglog(nu, powers[label], label=label)
ax1.set_ylabel("Normalized Power (a.u.)")
ax1.set_title("Synchrotron Emission Spectrum (MHz to THz)")
ax1.legend()
ax1.grid(True, which='both')


for label in polarizations:
    ax2.semilogx(nu, polarizations[label], label=label)
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Polarization Fraction (%)")
ax2.set_title("Polarization Fraction (MHz to THz)")
ax2.legend()
ax2.grid(True, which='both')

plt.tight_layout()
plt.savefig("synchrotron_emission.png", dpi=300)
plt.show()
