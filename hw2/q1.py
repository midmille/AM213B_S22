"""
Author: Miles D. Miller UCSC
Created: 12:07pm April 27, 2022

This file is for the solution to AM213B HW2 Question 1
"""

## [External]
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import fsolve

## [Internal ]
import ODE_Solvers as ODES



def Dtlam(theta): 
    """ 
    This comes up with the region of absolute stability for question 1. 

    """

    ## [Numerator.]
    dtlam_t = 12*(np.exp(3*1j*theta) - np.exp(2*1j*theta))
    ## [Denomerator.]
    dtlam_b = (23*np.exp(2*1j*theta) - 16 * np.exp(1j*theta) + 5)

    ## [Frac.]
    dtlam = dtlam_t / dtlam_b

    return dtlam


def Dtlam_zeros(theta):
    """
    This is to find the zeros of the Imaginary 

    """
    lam = -0.96662891+30.12547j
    ## [Lambda is defined globally]
    sol = Dtlam(theta) / lam
    ## [The imaginary portion.]
    sol_im = sol.imag

    return sol_im 


def Plot_Dtlam(dtlam): 
    """
    Plots the real against the imaginary part of the region of absolute stability
    for question 1.
    """

    fig, ax = plt.subplots()

    ax.plot(dtlam.real, dtlam.imag)
    ax.set_xlabel(r'$\mathbb{R}(\Delta t \lambda_j)$')
    ax.set_ylabel(r'$\mathbb{I}(\Delta t \lambda_j)$')
    ax.set_title("Region of Absolute Stability AB3")
    ax.grid()
    
    fig.show()

    return 


def Dydt(y, t): 
    """
    dydt for Question 1. 
    """

    ## [A is definde globally.]
    dydt = A @ y
    
    return dydt


def Plot_AM3_y(t, y): 
    """
    Plotting the resulting y solution vector. 
    
    """

    fig,ax = plt.subplots()

    ax.plot(t, y, label = [1,2,3])

    ax.set_ylabel('Y')
    ax.set_xlabel('t [s]')
    ax.set_title('The resulting AB3 Solution')
    ax.grid()
    ax.legend()

    fig.show()





if __name__ == '__main__':

    A = np.array([[0, 10, -10], 
                  [-100, -1, 0], 
                  [0, 10, -100]])

    ## [The eigenvalues of A.]
    sol = np.linalg.eig(A)

    ## [The eigen values.]
    e_vals = sol[0] 

    N_theta = 100

    theta = np.linspace(0, 2*np.pi, N_theta)

    dtlam = Dtlam(theta)

    ## fidning the theta zeroes of the equation above. 
    Dtlam_zeros(theta)
    lam = e_vals[0]
    fsol = fsolve(Dtlam_zeros, .0002)

    Plot_Dtlam(dtlam)

    ## [Some parameters.]
    dt = 1e-4
    t_tot = 5
    t = np.arange(0, t_tot, dt)
    N = len(t)
    y0 = np.array([10, 10, 10])

    ## [Solve for the AB3 solution.]
    y = ODES.AB3(N, dt, t, Dydt, y0)

    Plot_AM3_y(t, y)

