"""

question2.py

Author : Miles D. Miller 

Created : April 6, 2022 10:34 am

This file is for the region of absolute stability for Question 2 of the AM 213B Midterm. 
"""

## [External Modules.]
import numpy as np
import matplotlib.pyplot as plt

def S(z): 
    """
    The stability function of the provided RK3 implicit method

    The function denotes the boundary of the region of absolute stability when plotted on the 
    imaginary plane.
    """

    ## [The components of the provided butcher table for given RK3 implicit.]
    A = np.array([[0,0,0], 
                  [1/4, 1/4,  0], 
                  [0,1,0]])

    h = np.reshape(np.array([1,1,1]), (3,1))

    b = np.reshape(np.array([1/6,2/3,1/6]), (3,1))

    Id = np.identity(3)


    ## [The stability function.]
    s = (np.linalg.det(Id - z*A + z* h@ np.transpose(b))) / (np.linalg.det(Id - z*A))

    return s


def Plot_S(zz_real, zz_imag, s_mod): 
    """
    The plot of the boundary of the region of absolute stablility for RK3
    """

    fig, ax = plt.subplots()

    ax.contour(zz_real, zz_imag.imag, s_mod, levels=np.array([1]))

    ax.text(-4, 0.2, "REGION IS WITHIN BOUNDARY")

    ax.set_ylabel(r'$\mathbb{I}(z)$')
    ax.set_xlabel(r'$\mathbb{R}(z)$')
    ax.set_title("Region of Absolute Stability for Implicit RK3")
    ax.grid()

    fig.show()

    return 


if __name__ == '__main__': 

    N = 100
    M = N +1

    ## [The stability region can be plotted as a contour of imaginary and real z space.]
    z_real = np.linspace(-6, 1, N)
    z_imag = 1j * np.linspace(-5, 5, M)

    zz_real, zz_imag = np.meshgrid(z_real, z_imag)

    ## [Empty s_mesh stabilty result.]
    s_mesh = np.zeros((M,N), dtype=np.cdouble)

    ## [Loop over the meshes.]
    for k in range (N): 
        for j in range(M): 
            z = zz_real[j,k] + zz_imag[j,k]
            s_mesh[j,k] = S(z)

    ## [The modulus of the complex array.]
    s_mod = np.abs(s_mesh)

    ## [Plot the modulus one contour of RK3 implicit.]
    Plot_S(zz_real, zz_imag, s_mod)
