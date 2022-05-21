"""

question3.py

Author : Miles D. Miller 

Created : April 6, 2022 2:55 pm

This file solves for delta t max and implements the implicit RK3 Question 3 of the AM 213B Midterm. 
"""

## [External Modules.]
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp


def func(t, y):
    """
    The linear ODE system function formatted for use with  scipy solver. 

    This is merely included as a check to my RK3 solution

    Furthermore B is defined globally in if __name__=='__main__':

    """

    dydt = B @ y

    return dydt

def S_Zeros(dt): 
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

    ## [z is the product of the current eigenvalue and delta t]
    z = dt * e_val

    ## [The stability function.]
    s_min = 1 - np.abs((np.linalg.det(Id - z*A + z* h@ np.transpose(b))) / (np.linalg.det(Id - z*A)))

    return s_min


def Q3_RK3(N, dt, t, B, y0): 
    """
    This is the implementation of the implicit RK3 scheme for the solution of Question 3. 

    The system is linear and time independent so the function describing the ODE is given by 
    the linear system matrix B. i.e. f = By for this problem. 

    This also means that we can solve for the implcit k2 peice of the RK3 algorithm as a linear
    system instead oif using a newton's iteration. 
    
    """

    ## [The length of y0 tells us how many vectors there are]
    M = len(y0)

    ## [Init the solution y array.]
    y = np.zeros((N,M))

    ## [Set the initial conditions.]
    y[0,:] = y0

    ## [Loop over the t-grid.]
    for k in range(N-1): 
        ## [The resize is so that the chunk of y in a column.]
        y_k = np.reshape(y[k,:], (M,1))
        ## [The k1 step is just the linear function at u_k] 
        k1 = B@y_k

        ## [The implicit k2 step to be solved as a linear system.]
        Id = np.identity(M)
        k2 = np.linalg.solve((Id- dt*(1/4)*B), (B@y_k + dt*(1/4)*B@k1))

        ## [The k3 explicit step.]
        k3 = B@(y_k + dt*k2)

        ## [The next step.]
        y[k+1,:] = np.squeeze(y_k + dt*((1/6)*k1 + (2/3)*k2 + (1/6)*k3))

    return y
 

def Plot_Soly(M, t, y, title): 
    """
    Plotting the solution of y with regards to time
    """
    
    fig,ax = plt.subplots()

    ## [Loop over the vector compoenents of y.]
    for k in range(M): 
        ax.plot(t, y[:,k], label= f'y_{k}')

    ax.set_title(title)

    ax.grid()
    ax.legend()

    fig.show()

    

if __name__ == '__main__': 

    
    ## [The matrix of the linear systme we are solving.]
    B = np.array([[ -1, 3, -5, 7], 
                  [ 0, -2, 4, -6], 
                  [ 0, 0, -4, 6 ], 
                  [ 0, 0, 0, -16 ]])

    ## [The initital conditions.]
    y0 = np.array([1, 1, 1, 1])

    ## [The number of vector componeents.]
    M = len(y0)

    ## [solving for the dt_max of each respective eigenvalue.]
    ## [Caclulate the eigenvalues of B]
    e_vals = np.linalg.eig(B)[0]

    ## [An array of dt to hone in on the zeros of the max dt.]
    dt_arr = np.linspace(0, 1, 100)

    ## [The critical valuye for dts for a given eigenvalue.]
    dt_crit = np.zeros(M) 

    ## [loop over the eigenvalues]
    for k in range(len(e_vals)): 
        e_val = e_vals[k]

## [This is stuff for visualizing the zeros of each eigenvalue to hone in on the 
##  the best initial guess for the fsolve function.]
#        s_zeros_arr = np.zeros(len(dt_arr)) 
#        for j in range(len(dt_arr)): 
#            s_zeros_arr[j] = S_Zeros(dt_arr[j])
#        
#        ## [Plotting the zeros function.]
#        fig, ax = plt.subplots()
#        ax.plot(dt_arr, s_zeros_arr)
#        ax.set_title(f'Eig_Val: {e_val}')
#        ax.grid()
#        fig.show()

        ## [Solving for the zeroes using fsolve.]
        dt_crit[k] = fsolve(S_Zeros, 5)
        print(e_val, dt_crit[k])

    ## [Finding the minimum dt critical.]
    dt_max = dt_crit.min()
    
    ## [dt slightly above the critical value.]
    dt = dt_max + 0.01
    T = 10
    t = np.arange(0, T, dt)
    N = len(t)
    ## [Implementing the implicit RK3.]
    y = Q3_RK3(N, dt, t, B, y0)
    title = r'RK3 Implicit $\Delta t = \Delta t_{\mathrm{max}} + 0.01$'
    Plot_Soly(M, t, y, title=title)

    ## [Solving for dt just below critical value.]
    dt = dt_max - 0.01
    T = 10
    t = np.arange(0, T, dt)
    N = len(t)
    ## [Implementing the implicit RK3.]
    y = Q3_RK3(N, dt, t, B, y0)
    title = r'RK3 Implicit $\Delta t = \Delta t_{\mathrm{max}} - 0.01$'
    Plot_Soly(M, t, y, title=title)

    ## [Solving for very small dt.]
    dt = 0.01
    T = 10
    t = np.arange(0, T, dt)
    N = len(t)
    ## [Implementing the implicit RK3.]
    y = Q3_RK3(N, dt, t, B, y0)
    title = r'RK3 Implicit $\Delta t = 0.01$'
    Plot_Soly(M, t, y, title=title)

    ## [Solving the system with scipy solver to check my solution against=.]
    sp_sol = solve_ivp(func, [0,T], y0)
    t_sp = sp_sol.t
    y_sp = np.transpose(sp_sol.y)

    title = 'Scipy Solution'
    Plot_Soly(M, t_sp, y_sp, title=title)

    
    
    


