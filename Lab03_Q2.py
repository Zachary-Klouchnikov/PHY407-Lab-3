__authors__ = "Zachary Klouchnikov and Hannah Semple"

"""
IMPORTS
"""
import numpy as np

from gaussxw import gaussxw

"""
FUNCTIONS
"""

"""
PART A)
"""
"Constants"
K = 12.0
M = 1.0
X_0 = 0.001

"Stuff"
g = lambda x: 4 / np.sqrt(K * (X_0 ** 2 - x ** 2))

n = 8
a = 0.0
b = X_0

# Calculate the sample points and weights, then map them to the required
# integration domain
x, w = gaussxw(n)
xp = 0.5 * (b - a) * x + 0.5 * (b + a)
wp = 0.5 * (b - a) * w

# Perform the integration
s = 0.0
for k in range(n):
    s += wp[k] * g(xp[k])

print(s)
