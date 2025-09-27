__authors__ = "Zachary Klouchnikov and Hannah Semple"

# HEADER

"""
IMPORTS
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel
from matplotlib import cm
from numpy import ones,copy,cos,tan,pi,linspace

#Graphing
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
plt.rcParams["figure.figsize"] = (8,6)
plt.rcParams['font.size'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

"""
FUNCTIONS
"""

def trapezoidal_rule(f, a: float, b: float, n: int) -> float:
    """Returns the integral of a function using the trapezoidal rule.
    
    Arguments:
    f -- the function to be integrated
    a -- the lower limit of integration
    b -- the upper limit of integration
    n -- the number of slices to use in the approximation
    """
    # Width of the slices
    h = (b - a) / n

    # Calculating the endpoints of the integral
    s = 0.5 * f(a) + 0.5 * f(b)

    # Iterating over the slices between the endpoints
    for k in range(1, n):
        s += f(a + k * h)

    return h * s


def simpsons_rule(f, a: float, b: float, n: int) -> float:
    """Returns the integral of a function using Simpson's rule.

    Arguments:
    f -- the function to be integrated
    a -- the lower limit of integration
    b -- the upper limit of integration
    n -- the number of slices to use in the approximation

    Constraint: n must be even.
    """
    # Width of the slices
    h = (b - a) / n

    # Calculating the endpoints of the integral
    s = f(a) + f(b)

    # Iterating over the odd slices between the endpoints
    for k in range(1, n, 2):
        s += 4 * f(a + k * h)

    # Iterating over the even slices between the endpoints
    for k in range(2, n, 2):
        s += 2 * f(a + k * h)

    return h * s / 3


def integrand(x):
    """
    Returns the integrand for the integral I in Lab03-407-2025.pdf

    INPUT:
    x [float] is the variable of integration

    OUTPUT:
    i [float] is the value of the integrand evaluated at x
    """
    i = 4 / (1+x**2)
    return i


def gauss_err(f,a,b,N):
    """
    Returns the relative error using Gaussian quadrature for the integrand f

    INPUT:
    f [function] is the integrand that is being evaluated
    a [float] is the lower bound of integration
    b [float] is the upper bound of integration
    N [int] is the number of slices

    OUTPUT:
    err [float] is the relative error
    """
    x,w = gaussxwab(N,a,b)
    g_N = np.sum(w*f(x))  #Gauss. approximation for N
    
    x2,w2 = gaussxwab(2*N,a,b)
    g_2N = np.sum(w2*f(x2))  #Gauss. approximatoin for 2N
    
    err = g_2N - g_N
    return err


def fresnel_s(t):
    """
    Returns the integrand of the Fresnel equation S evaluated at t

    INPUT:
    t [float] is the variable of integration

    OUTPUT:
    s [float] is the calculated value of the integrand
    """
    s = np.sin(np.pi*(t**2)/2)
    return s


def fresnel_c(t):
    """
    Returns the integrand of the Fresnel equation C evaluated at t

    INPUT:
    t [float] is the variable of integration

    OUTPUT:
    c [float] is the calculated value of the integrand
    """
    c = np.cos(np.pi*(t**2)/2)
    return c
    

def diffraction(x,z,lam,N):
    """
    Returns the intensity of a wave diffracting around a wall at position (x,z), divided by the
    non-diffracted wave intensity, using our implementation of the Fresnel equations

    INPUT:
    x [float] is the horizontal distance from the wall in meters
    z [float] is the vertical distance from the wall in meters
    lam [float] is the wavelength in meters
    N [int] is the number of slices

    OUTPUT:
    I [float] is the wave intensity
    """
    u = x*np.sqrt(2/(lam*z))
    
    x_i,w = gaussxwab(N,0,u)
    c_gauss = np.sum(w*fresnel_c(x_i))
    s_gauss = np.sum(w*fresnel_s(x_i))
    
    I = (2*c_gauss + 1)**2 + (2*s_gauss + 1)**2
    
    return I/8


def diffraction_sc(x,z,lam,N):
    """
    Returns the intensity of a wave diffracting around a wall at position (x,z), divided by the
    non-diffracted wave intensity, using SciPy's implementation of the Fresnel equations

    INPUT:
    x [float] is the horizontal distance from the wall in meters
    z [float] is the vertical distance from the wall in meters
    lam [float] is the wavelength in meters
    N [int] is the number of slices

    OUTPUT:
    I [float] is the wave intensity
    """
    u = x*np.sqrt(2/(lam*z))
    
    s,c = fresnel(u)
    
    I = (2*c + 1)**2 + (2*s + 1)**2
    
    return I/8


def relative_diff(I_sp, I_g):
    """
    DOCSTRING
    """
    d = np.abs(I_sp - I_g) / I_sp
    return d


#Mark Newman functions
def gaussxw(N):

    # Initial approximation to roots of the Legendre polynomial
    a = linspace(3,4*N-1,N)/(4*N+2)
    x = cos(pi*a+1/(8*N*N*tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = ones(N,float)
        p1 = copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))

    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)

    return x,w

def gaussxwab(N,a,b):
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w
"""
PART A
"""

"""
PART B
"""

"""
PART C
"""
