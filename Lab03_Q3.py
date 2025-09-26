__authors__ = "Zachary Klouchnikov and Hannah Semple"

"""
IMPORTS
"""
import math
import numpy as np
import matplotlib.pyplot as plt

from collections.abc import Callable

"""
FUNCTIONS
"""
def forward_difference(f: Callable[[float], float], x: float, h: float
                       ) -> float:
    """Returns the numerical derivative of f at x using the forward
    difference method with step size h.
    
    Arguments:
    f -- the function to differentiate
    x -- the point at which to evaluate the derivative
    h -- the step size
    """

    return (f(x + h) - f(x)) / h

def central_difference(f: Callable[[float], float], x: float, h: float
                       ) -> float:
    """Returns the numerical derivative of f at x using the central
    difference method with step size h.
    
    Arguments:
    f -- the function to differentiate
    x -- the point at which to evaluate the derivative
    h -- the step size
    """

    return (f(x + (h / 2)) - f(x - (h / 2))) / h

"""
PART A)
"""
f = lambda x: math.exp(-x ** 2) # The function to differentiate

h = np.logspace(-16, 0, 17) # Step sizes from 10^-16 to 10^0
f_numerical = np.zeros_like(h) # Array to store numerical derivatives

# Calculate the numerical derivatives using forward difference
for i in range(len(h)):
    f_numerical[i] = forward_difference(f, 0.5, h[i])

"""
PART B)
"""
# The analytical derivative of f
f_prime = lambda x: -2 * x * math.exp(-x ** 2) 

h = np.logspace(-16, 0, 17) # Step sizes from 10^-16 to 10^0
f_analytical = np.zeros_like(h) # Array to store analytical derivatives
f_forward_error = np.zeros_like(h) # Array to store absolute errors

# Calculate the absolute errors
for i in range(len(h)):
    f_analytical[i] = f_prime(0.5)
    f_forward_error[i] = np.abs(f_analytical[i] - f_numerical[i])

print(f_forward_error)

"""
PART C) AND D)
"""
f = lambda x: math.exp(-x ** 2) # The function to differentiate

h = np.logspace(-16, 0, 17) # Step sizes from 10^-16 to 10^0
f_numerical = np.zeros_like(h) # Array to store numerical derivatives
f_analytical = np.zeros_like(h) # Array to store analytical derivatives
f_central_error = np.zeros_like(h) # Array to store absolute errors

# Calculate the numerical derivatives using central difference and
# absolute errors
for i in range(len(h)):
    f_numerical[i] = central_difference(f, 0.5, h[i])
    f_analytical[i] = f_prime(0.5)
    f_central_error[i] = np.abs(f_analytical[i] - f_numerical[i])

"Plotting the Absolute Error vs Step Size"
plt.figure()

# Plotting absolute error vs step size
plt.loglog(h, f_forward_error, ls = '-', color = 'Teal', label = "Forward Error")
plt.loglog(h, f_central_error, ls = '-', color = 'Purple', label = "Central Error")
plt.vlines(10 ** -8, 10 ** -16, 10 ** 1, linestyles = '--', colors = 'Coral', label = "Minimum Error")

# Labels
plt.title("Absolute Error vs Step Size", fontsize = 12)
plt.xlabel("Step Size", fontsize = 12)
plt.ylabel("Absolute Error", fontsize = 12)

plt.legend()
plt.grid()

# Limits
plt.xlim(10 ** -16, 10 ** 0)

plt.savefig('Figures\\Absolute Error vs Step Size.pdf')
plt.show()
