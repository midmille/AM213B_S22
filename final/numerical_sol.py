"""
numerical_sol.py
Author : Miles D. Miller
Created : June 8, 2022 11:01am 
About : This file is the numerical solution and the accompanying studies for the AM 213B final.
"""
## [User mods.]
import analytical_sol

## [External mods.]
import numpy as np
import matplotlib.pyplot as plt
import pickle


def Create_Space_Grid(r2, r1, Nrp2): 
    """
    This function forms the spatial grid for the solution following equation (9) on the 
    final exam.
    """

#    dr = (r2 - r1) / (Np1)

    ## [This is equivalent to r[j] = r1 + j*dr ; where j = 0,...,Np1 (including Np1)]
    ## [Thus the total length of r should be Np2.]
#    r = np.arange(r1, r2, dr)
    r = np.linspace(r1, r2, Nrp2)

    return r


def Create_M(N, nu, r, dr): 
    """
    This formulates the rhs matrix of the IBVP scheme for the final. The matrix construction 
    is outlined in the report.

    Parameters
    ----------
    N: Integer
        The number of nodes not including BC nodes, so j=1,..,N, but for r, j=0,..,Np1
    nu : Float
        Problem constant.
    r: 1-D Array, [N+2]
        This array is indexed from j=0,..,Np1 including Np1.
    dr: Float, 
        The seperation between spatial nodes.

    Returns: 
    --------
    M: 2-D Array, [N,N]
        The finite difference array for the innier nodes of the problem.
    """

    ## [The matrix will only be NxN since the BCs at j=0, j=N+1 are static and known.]
    M = np.zeros((N,N))
    
    ## [Eta only depends on dr, not r.]
    eta = 1/(dr**2)

    ## [python is zero based indexing.]
    for j in range(N):
        
        ## [Calculate the FD constant alpha, see report for deinition and derivation.]
        ## [recall that r includes the BC nodes here. Therefor must use j+1 for inner.]
        alpha = 1 / (2*dr*r[j+1])
        ## [The lower BC.]
        if j == 0: 
            M[j,j] = -2*eta 
            M[j,j+1] = eta + alpha
        ## [Upper BC.]
        elif j == N-1:
            M[j,j-1] = eta - alpha  
            M[j,j] = -2*eta 
        else:
            M[j,j-1] = eta - alpha  
            M[j,j] = -2*eta 
            M[j,j+1] = eta + alpha
    
    ## [Multiplit the matrix by the problem constant.]
    M = nu*M

    return M


def Spectral_Radius_M(r2, r1, nu, plot=False):
    """
    This implements the spectral radius study for the rhs matrix M from above.

    This is the solution to question 2 part b on the final exam.

    The result is also plotted.
    """

    ## [This N is just the N=20 for the looping of ks for the study.]
    N = 20

    ## [The spectral radius array.]
    specr_arr = np.zeros(N)
    ## [The array of values for n.]
    n_arr = np.zeros(N)

    for j in range(N):
        ## [Python is zero based indexing.]
        k = 1+j

        ## [The actual value for the number of inner nodes for the solution.]
        n = 10 + 10*(k-1)
        n_arr[j] = n

        ## [Get the spatial grid for this node size.]
        r = Create_Space_Grid(r2, r1, n+2)
        dr = r[1] - r[0]

        ## [Get the M rhs matrix.]
        M = Create_M(n, nu, r, dr)

        ## [solve for the Eigen values.]
        eigvals = np.linalg.eigvals(M)

        ## [The spectral radius is the maximum of the absolute value of the eigenvalues.]
        specr_arr[j] = np.max(np.absolute(eigvals))

    if plot:
        ## [This is the plotting of the results.]
        fig, ax = plt.subplots()
    
        ax.plot(n_arr, specr_arr, 'k')
        ax.plot(n_arr, specr_arr, 'bo', fillstyle='none', markersize=5)
    
        fontsize = 12
        ax.set_ylabel(r'Spectral Radius $\rho (\mathbf{M})$', fontsize = fontsize)
        ax.set_xlabel(r'Number of Inner Nodes [N]', fontsize = fontsize)
        ax.set_title('Spectral Radius of RHS FD Matrix vs Number of Inner Nodes')
        ax.grid()
    
        fig.show()

    return n_arr, specr_arr


def Absolute_Stability(r2, r1, nu): 
    """
    This is the solution to question 2 part c of the final exam.

    It also plots the log(dt_crit) against log(n).
    """

    ## [Run the spectral radius study and get, n_arr and specr_arr.]
    n_arr, specr_arr = Spectral_Radius_M(r2, r1, nu, plot=False)

    ## [The critical time step for absolute stability.]
    dt_crit = (6/11) * (1/specr_arr)

    ## [Plot the results as a log log plot.] 
    fig, ax = plt.subplots()

    ax.loglog(n_arr, dt_crit, 'k')
    ax.loglog(n_arr, dt_crit, 'bo', fillstyle='none', markersize=5)

    fontsize = 12
    ax.set_ylabel(r'log(Critical Time Step $\Delta t ^*$)', fontsize = fontsize)
    ax.set_xlabel(r'log(Number of Inner Nodes [N])', fontsize = fontsize)
    ax.set_title('Critical Time Step vs Number of Inner Nodes')
    ax.grid()

    fig.show()

    return

def Dudt(u, M): 
    """
    The rhs side of the final exam IVBP.

    Parameters
    ----------
    u: 1-D Array [N]
        The vector solution for u.
    M: 2-D Array [N,N]
        The finite difference rhs operator matrix for the problem.
    """

    return 


def U_num(r, t, dr, dt, Nr, Nrp1, Nrp2, Nt, nu, U0):
    """
    This solves for U numerically using finite difference and AB3. It is assumed that 
    t and r are constant grid.

    It is also assumed zero valued boundary conditions in space. 

    It is assumed that r[0] = r1 and r[Np1] = r[-1] = r2, s.t. len(r) = Np2
    """

    ## [The rows are t, the y-axis, and the columns, the x-axis, is r. ]
    U = np.zeros((Nt, Nrp2))
    
    ## [Construct the FD rhs matrix M, size NxN.]
    M = Create_M(Nr, nu, r, dr)

    ## [Must inittialize the first step of U to be U0.]
    U[0,:] = U0(r)

    ## [Must initialize steps k=1,2 for AB3 method using a one step method of the 
    ##  same order, namely RK3.]
    ## [This loop goes through k=0,1, which mean the k+1, k=2 step will also be initialized]
    for k in range(0,2): 
        ## [The three stages of RK3]
        k1 = M@U[k,1:Nrp1]
        k2 = M@(U[k,1:Nrp1] + dt*0.5*k1) 
        k3 = M@(U[k,1:Nrp1] + dt*(-1*k1 + 2*k2)) 

        ## [Solving for the next step in y.]
        U[k+1,1:Nrp1] = U[k,1:Nrp1] + dt*((1/6)*k1 + (2/3)*k2 + (1/6)*k3)

    ## [After the first three steps are initialized the AB3 method can be used for the rest.]
    for k in range(2, Nt-1):
        ## [The AB3 method.]
        U[k+1,1:Nrp1] = U[k,1:Nrp1] + (dt/12) * ( 23*M@U[k,1:Nrp1] - 16*M@U[k-1,1:Nrp1] + 5*M@U[k-2,1:Nrp1] )

    return U


def Plot_Usol(rr, tt, U, title): 
    """
    This plots the resulting U solution as a surface plot
    """

    ## [Plotting.]
    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})

    im = ax.plot_surface(rr, tt, U, cmap ='viridis')
    ax.set_ylabel('t')
    ax.set_xlabel('r')
    ax.set_zlabel('U(r, t)')
    ax.set_title(title)

    fig.show()

    ## [Plot Color map solution.]
    fig, ax = plt.subplots()

    im = ax.pcolormesh(rr, tt, U, cmap='nipy_spectral')
    ax.set_ylabel('t')
    ax.set_xlabel('r')
    ax.set_title(title)

    plt.colorbar(im, ax=ax)

    fig.show()

    return 


def Compute_Error_Allt(Ua, Un): 
    """
    This computes the maximum absolute biad between the analytical and numerical solution for 
    the IVBP on the final exam.

    Ua is the analytical solution and Un is the numerical solution.
    """

    diff = np.absolute(Ua - Un)

    ## [Taking the maximunm across each spatial axis.]
    err = diff.max(axis=1)

    return err


def Error_Study(Nk, t, dt, Nt, r2, r1, nu, U0, load=True): 
    """
    This implements the erro study requested of the final exam Q2 Part d.
    """
    savefile= 'err_arr.p'

    ## [The different Nrs to run the study for.]
    Nr_arr = np.array([25, 50, 100, 150])


    if load == True: 
        err_arr = pickle.load(open(savefile, 'rb'))

    else: 
        ## [The resulting error at each time step for each different Nrs above.]
        err_arr = np.zeros((Nt, len(Nr_arr)))
    
        ## [Loop over the Nr_arr.]
        for j, Nr in enumerate(Nr_arr):
            
            Nrp1 = Nr + 1
            Nrp2 = Nr + 2
    
            ## [Spatial grid for given Nr.]
            r = Create_Space_Grid(r2, r1, Nrp2)
            dr = r[1] - r[0]
    
            ## [The numerical solution for U.]
            Un = U_num(r, t, dr, dt, Nr, Nrp1, Nrp2, Nt, nu, U0)
            
            ## [The meshgrdi for analytical and plotting.]
            rr, tt = np.meshgrid(r,t)
        
            ## [The analytical solution for U.]
            Ua = analytical_sol.U_ana(rr, tt, Nk, r2, r1, U0, nu)
    
            ## [The error at each time step.]
            err_arr[:,j] = Compute_Error_Allt(Ua, Un)

        ## [Save the result to pickle.]
        pickle.dump(err_arr, open(savefile,'wb')) 
    
    ## [Plot the result.]
    fig, ax = plt.subplots()
    ## [Loop over the Nr_arr.]
    for j, Nr in enumerate(Nr_arr):
        ax.semilogy(t, err_arr[:,j], label= f'N = {Nr}')
    
    fontsize = 12
    ax.set_ylabel(r'log($e_N(t)$)')
    ax.set_xlabel(r't [s]')
    ax.set_title('The Error for Different Spatial Resolutions in Time')
    ax.legend()
    ax.grid()

    fig.show()

    return


def Error_Study_Finalt(Nk, t, dt, Nt, r2, r1, nu, U0, load=True):
    """
    This error study is done for with more Nr values and only looks at the final time error.

    This is the solution to Q2 Part e on the final exam.
    """

    savefile= 'err_arr_final.p'

    ## [This N is just the N=20 for the looping of ks for the study.]
    N = 20

    ## [The array of values for n.]
    n_arr = np.zeros(N, dtype=int)

    for j in range(N):
        ## [Python is zero based indexing.]
        k = 1+j
    
        ## [The actual value for the number of inner nodes for the solution.]
        n = 10 + 10*(k-1)
        n_arr[j] = n


    if load == True: 
        err_arr = pickle.load(open(savefile, 'rb'))

    else: 

        ## [The error array, the final e_max for each Nr.]
        err_arr = np.zeros(N)

        for j in range(N):
            
            Nr = n_arr[j]
            
            Nrp1 = Nr + 1
            Nrp2 = Nr + 2
    
            ## [Spatial grid for given Nr.]
            r = Create_Space_Grid(r2, r1, Nrp2)
            dr = r[1] - r[0]
    
            ## [The numerical solution for U.]
            Un = U_num(r, t, dr, dt, Nr, Nrp1, Nrp2, Nt, nu, U0)
            
            ## [The meshgrdi for analytical and plotting.]
            rr, tt = np.meshgrid(r,t)
        
            ## [The analytical solution for U.]
            Ua = analytical_sol.U_ana(rr, tt, Nk, r2, r1, U0, nu)

            ## [The error is only calculated from the final time step.]
            err_arr[j] = np.absolute(Ua[-1,:] - Un[-1,:]).max()

            ## [Save the result.]
            pickle.dump(err_arr, open(savefile, 'wb'))

    ## [Plotting the result.]
    fig, ax = plt.subplots()

    ax.loglog(n_arr, err_arr, 'k')
    ax.loglog(n_arr, err_arr, 'bo', fillstyle='none', markersize=5)

    fontsize = 12
    ax.set_ylabel(r'log($e_N(t=2)$)', fontsize = fontsize)
    ax.set_xlabel(r'log(Number of Inner Nodes [N])', fontsize = fontsize)
    ax.set_title('Error at Final Time Step vs Number of Inner Nodes')
    ax.grid()

    fig.show()



if __name__ == '__main__':

    ## [Problem parameters.]
    nu = 0.5 
    r1 = 1
    r2 = 4
    ## [Temporal resolution.] 
    dt = 5e-5
    ## [Start and End time.]
    t1 = 0
    t2 = 2
    ## [The spatial grid.]
    Nr = 100
    Nrp1 = Nr + 1
    Nrp2 = Nr + 2
    ## [The numer of ks in series to trunctae analytcial solution to.]
    Nk = 50
    ## [The initial condition function.]
    U0 = lambda x: 10*(x-1)*(4-x)**2 * np.exp(- x)

    ## [Q2 Part b {Spectral Radius Study}]
    Spectral_Radius_M(r2, r1, nu, plot=True)

    ## [Q2 Part c {Critical Time Step Study}
    Absolute_Stability(r2, r1, nu)

    ## [The time grid.]
    t =  np.arange(t1, t2, dt)
    Nt = len(t)

    ## [Q2 Part d {Error Study in Time.}]
#    Error_Study(Nk, t, dt, Nt, r2, r1, nu, U0, load=True)

    ## [Q3 Part e {Error Study at Final Time.}]
#    Error_Study_Finalt(Nk, t, dt, Nt, r2, r1, nu, U0, load=False)

    ## [This is just for plotting a single solution.]
    ## [Create the spatial grid.]
    r = Create_Space_Grid(r2, r1, Nrp2)
    ## [Evenly spaced grid.]
    dr = r[1] - r[0]

    ## [The numerical solution for U.]
#    Un = U_num(r, t, dr, dt, Nr, Nrp1, Nrp2, Nt, nu, U0)
    
    ## [The meshgrdi for analytical and plotting.]
    rr, tt = np.meshgrid(r,t)

    ## [The analytical solution for U.]
#    Ua = analytical_sol.U_ana(rr, tt, Nk, r2, r1, U0, nu)

#    e = Compute_Error_Allt(Ua, Un)




    
