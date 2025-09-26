__authors__ = "Zachary Klouchnikov and Hannah Semple"

"""
IMPORTS
"""
import numpy as np
import matplotlib.pyplot as plt

from gaussxw import gaussxwab
from collections.abc import Callable

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

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

"Plotting the Integrands at the Sample Points"
plt.figure()

# Plotting integrands at the sample points
plt.plot(g_8[0], 4 / g_8[2], ls = '-', color = 'Teal', label = "N = 8")
plt.plot(g_16[0], 4 / g_16[2], ls = '-', color = 'Purple',
         label = "N = 16")

# Labels
plt.title("Integrands at the Sample Points", fontsize = 12)
plt.xlabel("$x_k$", fontsize = 12)
plt.ylabel("$4 / g(x_k)$", fontsize = 12)

plt.legend()
plt.grid()

# Limits
plt.xlim(0.0, X_0)

plt.savefig('Figures\\Integrands at the Sample Points.pdf')
plt.show()

"Plotting Weighted Integrands at the Sample Points"
plt.figure()

# Plotting weighted integrands
plt.plot(g_8[0], 4 * g_8[1] / g_8[2], ls = '-', color = 'Teal',
         label = "N = 8")
plt.plot(g_16[0], 4 * g_16[1] / g_16[2], ls = '-', color = 'Purple',
         label = "N = 16")

# Labels
plt.title("Weighted Integrands at the Sample Points", fontsize = 12)
plt.xlabel("$x_k$", fontsize = 12)
plt.ylabel("$4w_k / g(x_k)$", fontsize = 12)

plt.legend()
plt.grid()

# Limits
plt.xlim(0.0, X_0)

plt.savefig('Figures\\Weighted Integrands at the Sample Points.pdf')
plt.show()

"""
PART C)
"""
"Constants"
K = 12.0
M = 1.0
X_C = 86602540.38

"Integrate the function g(x) using N = 200 sample points"
g = lambda x: 4 / np.sqrt(K * np.abs(x_0 ** 2 - x ** 2))

x_0 = 0.001
g_200 = gaussian_quadrature(g, 0.0, x_0, 200) # Integrate g(x) with N = 200

print(np.sum(g_200[2]))

x_0 = np.linspace(1.0, 10.0 * X_C, 200)
period = np.zeros_like(x_0)

for i in range(len(x_0)):
    period[i] = np.sum(gaussian_quadrature(g, 0.0, x_0[i], 200)[0])

"Plotting Period vs Maximum Displacement"
plt.figure()

# Plotting period vs maximum displacement
plt.loglog(x_0, period, ls = '-', color = 'Teal', label = "Period")

# Labels
plt.title("Period vs Maximum Displacement", fontsize = 12)
plt.xlabel("$x_0$", fontsize = 12)
plt.ylabel("$T(x_0)$", fontsize = 12)

plt.legend()
plt.grid()

# Limits
plt.xlim(1.0, 10.0 * X_C)

plt.savefig('Figures\\Period vs Maximum Displacement.pdf')
plt.show()
