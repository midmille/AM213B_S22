"""
Author: Miles D. Miller UCSC
Created: 3:51pm April 11, 2022 

This file is for the solution of AM213B HW 1 Question 2 on the Hamilton dynamic 
system for the solution of the spring pendulum problem
"""

## [External Modules.]
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate 
## [Internal Modules.]
from ODE_Solvers import Euler_Forward
from ODE_Solvers import RK2


def Midpoint_Implicit_Q2_Spec(N, dt, t, dydt, y0, tol=10e-5, maxit=30): 
    """
    This method uses the implicit midpoint method to solve the hamiltonian system of question 2 in homework 1. 
    This algorithm is specific to the problem because it assumes an analytical solution to the jacobian
    of the given problem. The algorithm uses newtons method with the given analytical jacobian. 

    The algorithm assumes a uniform grid. 

    It also assumes that y is a vector. 

    Parameters
    ----------
    N: Integer
        The number of grid points. 
    dt: Float
        The time step. 
    t: 1-D Array, [N]
        The uniform time grid. 
    dydt: Function
        The rhs side of the differential equation. 
    y0: 1-D Array, [M]
        The initial condition of the IVP. 
    tol: Optional, Float, default=10e-5
        The tolerance for the netwon's method convergence.
    maxit: Optional, Float, default=30
        The maximum number of newton's method iterations allowed for each step of the 
        implcit midpoint method. An error will be provided if exceeded.

    Returns 
    -------
    y: 2-D Array, [N, M]
        The solution to the ODE, with y as a vector. 
        
    """

    def J_G(y, dt):
        """
        The Jacobian of the rhs of the implicit midpoint method on the system of equations in 
        hw 1, question 2. 

        This is used for the derivative in newton's method.

        Parameters
        ----------
        y: 1-D Array, [M]
            The y-vector, for this system y1, y2, y3, y4 = [x, theta, p_x, p_theta]

        Returns
        -------
        jg: 2-D Array, [M,M] 
            The jacobian of the rhs of the implicit midpoint method for question 2. 
            This jacobian was solved for analyticaly.   

        """

        jg = [[0, 0, 1/m, 0], 
               [-(2*y[3]) / (m*(l0+y[0])**3), 0, 0, 1/(m*(l0+y[0])**2)], 
               [-(3*y[3]**2)/(m*(l0+y[0])**4) - kap, -m*g*np.sin(y[1]), 0, (2*y[3])/(m*(l0 + y[0])**3)], 
               [-m*g*np.sin(y[1]), -m*g*(y[0]+l0)*np.cos(y[1]), 0, 0]]

        ## [This includes the dt and the 0.5 from chain rule.]
        jg = 0.5*dt*np.array(jg)

        return jg

    ## [The length of y0 tells us how many vectors there are]
    M = len(y0)

    ## [Init the solution y array.]
    y = np.zeros((N,M))

    ## [4x4 matrix for Newton's.]
    I = np.identity(4)

    ## [Set the initial conditions.]
    y[0,:] = y0

    ## [Loop over the t-grid.]
    for k in range(N-1): 
        ## [Implement newton's method on the current midpoint iteration step] 
        ## [Init the convergence metric.]
        conv = 1 
        ## [Init the iteration counter.]
        cnt = 0

        ## [While loop until converge on u_k+1]
        while (conv>tol): 
            ## [Store previous old y. ]
            y_prev = y[k+1,:]

            ## [Check counter.]
            if (cnt>maxit):
                raise Exception("Max iteration count")

            ## [Init the initial guess to be the previous step value.]
            if cnt == 0:  
                y[k+1,:] = y[k,:]
                ## [Set the conv to be large such that it iterates through this init step.]
                conv = 1

            ## [Otherwise perform newton's method iterations on system to find y[k+1,:]
            else: 
                ## [solving the newton's linear system. See AM213B chpt 2 notes eqns 13-18.]
                x = np.linalg.solve(I-J_G(y[k+1,:], dt), y[k,:] + dt*dydt(0.5*(y[k,:] + y[k+1,:]), t[k] + 0.5*dt) - y[k+1,:]) 
                if any(np.isnan(x)):
                    raise Exception("NAN")

                y[k+1,:] = x + y[k+1,:]

                ## [Calculate the conv.]
                conv = np.linalg.norm(y[k+1,:] - y_prev, ord=2)

            ## [Increase the counter]
            cnt += 1

    return y


def dydt(y, t): 
    """
    This is the system of differential equations for the Hamilton problem. 

    The system in our vector coordinates: 
    dy0/dt = y2/m 
    dy1/dt = y3/ (m(l0 + y0)**2)
    dy2/dt = (y3/(m(l0+y0)**3)) + mgcos(y1) - kap*y0
    dy3/dt = -mg(l0 + y0)sin(y1)

    Parameters
    ----------
    y: 1-D Array, [4]
        The vector valued y. The length of the input vector should be 4. 
    t: Float
        The time coordinate, the problem is time independent so it is not actually 
        necessary. 

    Returns
    -------
    dydt: 1-D Array, [4]
        The vector derivative solution of the system of ODEs. The length should be 4.

    """

    ## [Init dydt.]
    dydt = np.zeros(len(y))

    ## [Solving the system of equations.]
    dydt[0] = y[2] / m  
    dydt[1] = y[3] / (m * (l0 + y[0])**2) 
    dydt[2] = (y[3]**2 / (m * (l0 + y[0])**3)) + m*g*np.cos(y[1]) - kap*y[0]
    dydt[3] = -m*g*(l0 + y[0])*np.sin(y[1])

    return dydt


def Hamiltonian(y): 
    """

    """
    if len(y.shape) == 2: 
        H = y[:,2]**2/(2*m) + y[:,3]**2/(2*m*(l0 + y[:,0])**2) - m*g*(l0 + y[:,0])*np.cos(y[:,1]) + 0.5 * kap * y[:,0]**2 

    if len(y.shape) == 1: 
        H = y[2]**2/(2*m) + y[3]**2/(2*m*(l0 + y[0])**2) - m*g*(l0 + y[0])*np.cos(y[1]) + 0.5 * kap * y[0]**2 

    return H
if __name__ == '__main__': 
    
    ## Global Parameters
    ## ------------------
    g = 9.8 ## [m s^-2]
    l0 = 1 ## [m]
    m = 0.5 ## [kg] 
    kap = 10 ## [N m^-1] 

    y0 = np.zeros(4)
    ## [The initial values.]
    y0[0] = 0 
    y0[1] = (3/4) * np.pi
    y0[2] = 0 
    y0[3] = 0
    
    
    ## [The total time.]
    T = 60

    ## [Value of dt.]
    dt = 4e-2
    
    ## [The t grid.]
    t = np.arange(0, T, dt)
    ## [The number of grip points.]
    N = len(t)
    
    #y_true = integrate.solve_ivp(dydt, (0,T) , y0, atol=10e-4, rtol=10e-3, method='BDF')

    ## [Solving for Euler Forward Solution.]
    y_eulf = Euler_Forward(N, dt, t, dydt, y0)
    H_eulf = Hamiltonian(y_eulf) 

    ## [Solving for Heun Solution.]
    y_rk2 = RK2(N, dt, t, dydt, y0)
    H_rk2 = Hamiltonian(y_rk2) 

    ## [Solving for the implicit midpoint solution.]
    y_immd = Midpoint_Implicit_Q2_Spec(N, dt, t, dydt, y0, tol=10e-12, maxit=30)
    H_immd = Hamiltonian(y_immd)

    ## [Plotting the x and theta for all methods.]
    fig, axs = plt.subplots(ncols = 3, nrows =1)

    ## [Plotting euler theta and x.]
    ax = axs[0]
    ax.plot(t, y_eulf[:,0], label=r'$x$')
    ax.plot(t, y_eulf[:,1], label=r'$\theta$')
    ax.set_title(f'Euler Forward, dt = {dt}')
    ax.set_ylabel(r'$\theta$ [rad], $x$ [m]')
    ax.set_xlabel('t [s]')
    ax.grid()
    ax.legend()

    ## [Plotting RK2.]
    ax = axs[1]
    ax.plot(t, y_rk2[:,0], label=r'$x$')
    ax.plot(t, y_rk2[:,1], label=r'$\theta$')
    ax.set_title(f'Heun, dt = {dt}')
    ax.set_ylabel(r'$\theta$ [rad], $x$ [m]')
    ax.set_xlabel('t [s]')
    ax.grid()
    ax.legend()

    ## [Plotting RK2.]
    ax = axs[2]
    ax.plot(t, y_immd[:,0], label=r'$x$')
    ax.plot(t, y_immd[:,1], label=r'$\theta$')
    ax.set_title(f'Implicit Midpoint, dt = {dt}')
    ax.set_ylabel(r'$\theta$ [rad], $x$ [m]')
    ax.set_xlabel('t [s]')
    ax.grid()
    ax.legend()

    fig.show()

    ## [Plotting the Hamiltonian for all methods.]
    fig, ax = plt.subplots() 

    ax.plot(t, H_eulf, label='Euler Forward')
    ax.plot(t, H_rk2, label='Heun')
    ax.plot(t, H_immd, label='Implicit Midpoint')
    ax.plot(t, np.full(len(t), Hamiltonian(y0)), '--k', label='True Hamiltonian')
    ax.set_ylabel('H')
    ax.set_xlabel('t [s]')
    ax.set_title(f'Hamiltonian of Different Schemes, dt={dt}')
    ax.grid()
    ax.legend()

    fig.show()



