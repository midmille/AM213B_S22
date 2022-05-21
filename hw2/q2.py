"""
Author: Miles D. Miller UCSC
Created: 1:46pm April 27, 2022

This file is for the solution to AM213B HW2 Question 2. 
"""

## [External]
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np 

def Rho(z): 
    """
    The function for the first charecteristic roots of the problem. 
    """

    rho = z**3 - (18/11)*z**2 + (9/11)*z - (2/11) 

    return rho
     

def Plot_Rho(z, rho): 
    """
    This plots the frist charectersitic polynomial so that  I can more eaily guess
    the zeros of the function for fsolve.
    """

    fig, ax = plt.subplots()

    ax.plot(z, rho)
    ax.set_ylabel(r"$\rho(z)$")
    ax.set_xlabel("z")
    ax.set_title("BDF3 First Charecteristic Polynomial")
    ax.grid()

    fig.show()

    return


def Dtlam(theta): 
    """
    This is the function for the region of absolute stability of the problem
    """

    dtlam = (11/6) * (1 - (18/11) * np.exp(-1j*theta) + (9/11)*np.exp(-2j*theta) - (2/11)*np.exp(-3j*theta))

    return dtlam


def Plot_Dtlam(dtlam): 
    """
    This plots the region of absolute stability
    """

    fig, ax = plt.subplots()

    ax.plot(dtlam.real, dtlam.imag)
    ax.set_xlabel(r'$\mathbb{R}(\Delta t \lambda_j)$')
    ax.set_ylabel(r'$\mathbb{I}(\Delta t \lambda_j)$')
    ax.set_title("Region of Absolute Stability BDF3")
    ax.grid()
    
    fig.show()

    return 

if __name__ == '__main__': 

    z = np.linspace(.2,1.1, 100)
    rho = Rho(z)

    ## [Plotting the charectersitic poolynomial]
    Plot_Rho(z, rho)
    
    ## [Solving for the root of the charecteristic polynomial.]
    sol = fsolve(Rho,1)

    ## [Theta.]
    theta = np.linspace(0, 2*np.pi, 100)
    ## [Dtlam]
    dtlam = Dtlam(theta)
    ## [plotting the region of absolute stability]
    Plot_Dtlam(dtlam)



    

