__authors__ = "Zachary Klouchnikov and Hannah Semple"

"""
IMPORTS
"""
import numpy as np
import matplotlib.pyplot as plt

from gaussxw import gaussxwab
from collections.abc import Callable

"""
FUNCTIONS
"""
def gaussian_quadrature(f: Callable[[float], float], a: float, b: float,
                        n: int) -> tuple[np.ndarray, np.ndarray,
                                         np.ndarray]:
    """Returns the integral of f from [a, b] using Gaussian quadrature 
    with n sample points.

    Arguments:
    f -- the function to integrate
    a -- the lower limit of integration
    b -- the upper limit of integration
    n -- the number of sample points to use
    """
    # Calculate the sample points and weights
    x, w = gaussxwab(n, a, b)
    
    # Perform the integration
    integral = np.array([])
    for i in range(n):
        integral = np.append(integral, w[i] * f(x[i]))

    return x, w, integral

"""
PART A)
"""
"Constants"
K = 12.0
M = 1.0
X_0 = 0.001

"Integrate the function g(x) using N = 8 and N = 16 sample points"
g = lambda x: 4 / np.sqrt(K * (X_0 ** 2 - x ** 2))

g_8 = gaussian_quadrature(g, 0.0, X_0, 8) # Integrate g(x) with N = 8
g_16 = gaussian_quadrature(g, 0.0, X_0, 16) # Integrate g(x) with N = 16

print(np.sum(g_8[2]))
print(np.sum(g_16[2]))

"Plotting Bessel functions using Simpson's rule"
plt.figure()

# Plotting Bessel functions
plt.plot(g_8[0], 4 / g_8[2], ls = '-', color = 'Teal', label = "$J_0(x)$")
plt.plot(g_16[0], 4 / g_16[2], ls = '-', color = 'Purple', label = "$J_0(x)$")

# Labels
# plt.title("Bessel Functions Using Simpson's Rule", fontsize = 12)
# plt.xlabel("x", fontsize = 12)
# plt.ylabel("$J_n(x)$", fontsize = 12)

# plt.legend()
# plt.grid()

# plt.savefig('Figures\\Bessel Functions Using Simpson\'s Rule.pdf')
plt.show()

"Plotting Bessel functions using Simpson's rule"
plt.figure()

# Plotting Bessel functions
plt.plot(g_8[0], 4 * g_8[1] / g_8[2], ls = '-', color = 'Teal', label = "$J_3(x)$")
plt.plot(g_16[0], 4 * g_16[1] / g_16[2], ls = '-', color = 'Purple', label = "$J_3(x)$")

# Labels
# plt.title("Bessel Functions Using Simpson's Rule", fontsize = 12)
# plt.xlabel("x", fontsize = 12)
# plt.ylabel("$J_n(x)$", fontsize = 12)

# plt.legend()
# plt.grid()

# plt.savefig('Figures\\Bessel Functions Using Simpson\'s Rule.pdf')
plt.show()
