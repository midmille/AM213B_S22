"""
analytical_sol.py
Author : Miles D. Miller
Created : June 7, 2022 1:47pm 
About : This file is the analytcial solution to the provided Initital boundary value problem 
        for the solution to AM213B final question 1. 
"""

import numpy as np
import matplotlib.pyplot as plt
## [Import scipy first and second type bessel functions respectively.]
from scipy.special import jv, yn
from scipy.optimize import fsolve
import scipy.integrate as integrate

def Rk(betak, r2, r): 
    """
    This is the rhs of the R_k(r) equation. The first of equations (5) in the Final Exam.

    All the bessel functions used are of the order 0. 
    """
    order = 0

    rhs = jv(order, betak*r) * yn(order, betak*r2) - jv(order, betak*r2) * yn(order, betak*r)

    return rhs 


def Rk_Lw(betak, r2, r1): 
    """
    """

    order = 0

    rhs = (2/(np.pi**2 * betak**2)) * (((jv(order, betak*r1))**2 - (jv(order, betak*r2))**2) / ((jv(order, betak*r1))**2))

    ## ||Rk||_Lw = sqrt(rhs)
    rhs = np.sqrt(rhs)

    return rhs


def Rk_hat(r, betak, r2, r1):
    """
    """
    
    rhs = Rk(betak, r2, r) / Rk_Lw(betak, r2, r1)

    return rhs


def Betak(Nk, r2, r1, plot=False): 
    """
    This solves for the betak zeros of the function Rk(betak, r2, r1) using fsolve.

    Parameters
    ----------
    Nk: Integer
        The number of ks to use, from 1 to Nk including Nk.
    """

    ## [The array of k values we are solving for the zero betak's fo.]
    ks = np.arange(1, Nk+1, 1)

    ## [The initial guess for beta_k, this is provided on the final exam problem outline.]
    betak_init = 1 + 1.05 * (ks-1)

    ## [Using fsolve to solve for the zeros.]
    betak = fsolve(Rk, betak_init, args = (r2,r1))

    if plot: 
        fig,ax = plt.subplots()
        beta = np.linspace(1, 1+1.05*(Nk+1-1), 500)
        rk = Rk(beta, r2, r1)
        ax.plot(beta, rk, label= 'Rk')
        ax.plot(betak_init, Rk(betak_init, r2, r1), 'bo', fillstyle='none', label=r'$\beta_k$ Initial Guess')
        ax.plot(betak, Rk(betak, r2, r1), 'ro', fillstyle='none', label=r'$\beta_k$ Solution')
        ax.set_ylabel('Rk')
        ax.set_xlabel(r'$\beta$')
        ax.set_title(r'fsolve() result for $\beta_k$')
        ax.legend()
        ax.grid()
        fig.show()

    return betak


def Uk(r, t, betak, r2, r1, U0, nu):
    """
    This function computes U at r, t, and scalar k.

    i.e. this inside of the summnation for equaiotn (3) in the final exam.
    
    Parameters
    ----------
    r: 2-D array, [Nt, Nx]
        The mesh grid for radius.
    t: 2-D array, [Nt, Nx]
        The mesh grid for time.
    r2, r1: Integers
        The bounnds of space.
    U0: callable initial condition function of r
    nu: Float
        The scalar constan nu for the problem.

    Returns
    -------
    Uk: 2-D Array, [Nt, NX]
        The solution for U at k.
    """

    ## [Calculate the definite integral of rprime.]
    integral = integrate.quad(lambda x: x*Rk_hat(x, betak, r2, r1)*U0(x), r1, r2)[0]
    
    ## [The rest of the rhs of equation (3) in the final exam.]
    rhs = np.exp(- betak**2 * nu * t) * Rk_hat(r, betak, r2, r1) * integral

    return rhs

def U_ana(r, t, Nk, r2, r1, U0, nu): 
    """
    The anlytical solution to equation (1) in the final exam, i.e., equation (3).
    """
    ## [The ks.]
    ks = np.arange(1, Nk+1, 1)

    ## [Solve for the betaks.]
    betaks = Betak(Nk, r2, r1)

    ## [init the rhs of the sum.]
    rhs = 0

    ## [Loop over and implement the sum.]
    for k in ks: 
        
        rhs = rhs + Uk(r, t, betaks[k-1], r2, r1, U0, nu)
        
    return rhs 


if __name__ == '__main__': 
    pass 
