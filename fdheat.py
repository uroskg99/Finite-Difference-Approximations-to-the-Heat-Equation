import numpy as np
import matplotlib.pylab as plt
import math

def tridiagLU(a,b,c):
    #tridiagLU Obtain the LU factorization of a tridiagonal matrix

    #Synopsis: (e,f) = tridiag(a,b,c)

    #Input: a,b,c = vectors defining the tridiagonal matrix. a is the
    #          subdiagonal, b is the main diagonal, and c is the superdiagonal

    #Outpu: vectors defining the L and U factors of the tridiagonal matrix

    n = len(a)
    e = np.zeros((n,1),dtype=float)
    f = np.zeros((n,1),dtype=float)
    e[0] = b[0]
    f[0] = c[0] / b[0]

    for i in range(1,n):
        e[i] = b[i] - a[i] * f[i-1]
        f[i] = c[i] / e[i]

    return e, f


def tridiagLUsolve(d,a,e,f,v):

    #tridiagLUsolve Solve (LU)*v = d where L and U are LU factors of a tridiagonal matrix

    #Synopsis: v = tridiagLUsolve(d,a,e,f,v)
    
    #Input: d = right hand side vector of the system of equatoins
    #       e,f = vectors defining the L and U factors of the tridiagonal matrix.
    #       e and f are obtained with the tridiagLU function
    #       v = solution vector. v is used as a scratch vector in the forward solve.
 
    #Output: v = solution vector

    ng = len(d)
    v[0] = d[0] / e[0]

    #Forward substitution to solve L*w = d
    for p in range(1,ng):
        v[p] = (d[p] - a[p]*v[p-1]) / e[p]

    #Backward substitution to solve U*v = w
    for z in range(ng-2, -1,-1):
        v[z] = v[z] - f[z]*v[z+1]

    return v


def heatFTCS(nt=10, nx=20, alpha=0.1, L=1, tmax=0.5, errPlots=1):

    #heatFTCS Solve 1D heat equation with the FTCS scheme

    #Synopsis:  heatFTCS()
    #           heatFTCS(nt)
    #           heatFTCS(nt,nx)
    #           heatFTCS(nt,nx,alpha)
    #           heatFTCS(nt,nx,alpha,L)
    #           heatFTCS(nt,nx,alpha,L,tmax)
    #           heatFTCS(nt,nx,alpha,L,tmax,errPlots)
    #           err = heatFTCS(...)
    
    #Input:   nt = number of steps. Default: nt = 10;
    #         nx = number of mesh points in x direction. Default: nx=20
    #         alpha = diffusivity. Default: alpha = 0.1
    #         L = length of the domain. Default: L = 1;
    #         tmax = maximum time for the simulation. Default: tmax = 0.5
    #         errPlots = flag (1 or 0) to control whether error plots should be shown
    
    #Output: err = L2 norm of error evaluated at the spatial nodes on last time step
    #        x = location of finite difference nodes
    #        t = values of time at which solution is obtained (time nodes)
    #        U = matrix of solutions: U(:,j) is U(x) at t = t(j)

    #Compute mesh spacing and time step
    dx = L/(nx-1)
    dt = tmax/(nt-1)
    r = alpha*dt / dx**2
    r2 = 1 - 2*r
        
    #Create arrays to save data for export
    x = np.linspace(0, L, nx)
    t = np.linspace(0, tmax, nt)
    u = np.zeros((nx,nt), dtype=float)

    #Set IC and BC
    q = np.zeros(nx)
    z = math.pi * x / L

    for i in range(0,nx):
        q[i] = math.sin(z[i])

    u[:,0] = q
    u0 = 0
    uL = 0

    #Loop over time steps
    for m in range(1, nt):
        for i in range(1, nx-1):
            u[i,m] = r*u[i-1, m-1] + r2*u[i, m-1] + r*u[i+1, m-1]

    #Compare with exact solution at end of simulation
    ue = q * math.exp(-t[nt-1] * alpha * (math.pi / L)**2)
    err = np.linalg.norm(u[:, nt-1] - ue)
    errgraph = u[:, nt-1] - ue

    print('Norm of error = ', err, ' at t = ', t[nt-1])
    print('tdt = ', dt, ' dx = ', dx, ' r = ', r)

    plt.title('heatFTCS', fontsize=20)
    plt.xlabel('x', fontsize=15)
    plt.ylabel('u', fontsize=15)
    plt.plot(x, ue, 'r-', linewidth=1, label='Exact')
    plt.plot(x, u[:,nt-1], 'b--o', linewidth=1, label='FTCS')
    plt.legend()
    return plt.show()


def heatBTCS(nt=10, nx=20, alpha=0.1, L=1, tmax=0.5, errPlots=1):

    #heatBTCS Solve 1D heat equation with the BTCS scheme

    #Synopsis:  heatBTCS()
    #           heatBTCS(nt)
    #           heatBTCS(nt,nx)
    #           heatBTCS(nt,nx,alpha)
    #           heatBTCS(nt,nx,alpha,L)
    #           heatBTCS(nt,nx,alpha,L,tmax)
    #           heatBTCS(nt,nx,alpha,L,tmax,errPlots)
    #           err = heatBTCS(...)
    
    #Input:   nt = number of steps. Default: nt = 10;
    #         nx = number of mesh points in x direction. Default: nx=20
    #         alpha = diffusivity. Default: alpha = 0.1
    #         L = length of the domain. Default: L = 1;
    #         tmax = maximum time for the simulation. Default: tmax = 0.5
    #         errPlots = flag (1 or 0) to control whether error plots should be shown
    
    #Output: err = L2 norm of error evaluated at the spatial nodes on last time step
    #        x = location of finite difference nodes
    #        t = values of time at which solution is obtained (time nodes)
    #        U = matrix of solutions: U(:,j) is U(x) at t = t(j)

    #Compute mesh spacing and time step
    dx = L/(nx-1)
    dt = tmax/(nt-1)
    
    #Create arrays to save data for export
    x = np.linspace(0,L,nx)
    t = np.linspace(0,tmax,nt)
    u = np.zeros((nx,nt), dtype=float)

    #Set IC and BC
    q = np.zeros(nx)
    z = math.pi * x / L

    for i in range(0,nx):
        q[i] = math.sin(z[i])

    u[:,0] = q
    u0 = 0
    uL = 0

    #Coefficients of the tridiagonal system

    #subdiagonal a: coefficients of phi(i-1)
    a = (-alpha / dx**2) * np.ones((nx,1), dtype=float)

    #superdiagonal c: coefficients of phi(i+1)
    c = (-alpha / dx**2) * np.ones((nx,1), dtype=float)

    #diagonal b: coefficients of phi(i)
    b = (1/dt) * np.ones((nx,1), dtype=float) - 2*a

    #Fix coefficients of boundary nodes
    b[0] = 1
    c[0] = 0
    b[-1] = 1
    a[-1] = 0

    #Get LU factorization of coefficient matrix
    (e, f) = tridiagLU(a,b,c)

    #Loop over time steps
    for m in range(1,nt):
        d = u[:,m-1] / dt
        d[0] = u0
        d[-1] = uL
        u[:,m] = tridiagLUsolve(d,a,e,f,u[:,m-1])

    #Compare with exact solution at end of simulation
    ue = q * math.exp(-t[nt-1] * alpha * (math.pi / L)**2)
    err = np.linalg.norm(u[:,nt-1] - ue)
    errgraph = u[:, nt-1] - ue

    print('Norm of error = ', err, ' at t = ', t[nt-1])
    print('tdt = ', dt, ' dx = ', dx)

    plt.title('heatBTCS', fontsize=20)
    plt.xlabel('x', fontsize=15)
    plt.ylabel('u', fontsize=15)
    plt.plot(x, ue, 'r-', linewidth=1, label='Exact')
    plt.plot(x, u[:,nt-1], 'b--o', linewidth=1, label='BTCS')
    plt.legend()
    return plt.show()


def heatCN(nt=10, nx=20, alpha=0.1, L=1, tmax=0.5, errPlots=1):

    #heatCN Solve 1D heat equation with the Crank-Nicolson scheme

    #Synopsis:  heatCN()
    #           heatCN(nt)
    #           heatCN(nt,nx)
    #           heatCN(nt,nx,alpha)
    #           heatCN(nt,nx,alpha,L)
    #           heatCN(nt,nx,alpha,L,tmax)
    #           heatCN(nt,nx,alpha,L,tmax,errPlots)
    #           err = heatCN(...)
    
    #Input:   nt = number of steps. Default: nt = 10;
    #         nx = number of mesh points in x direction. Default: nx=20
    #         alpha = diffusivity. Default: alpha = 0.1
    #         L = length of the domain. Default: L = 1;
    #         tmax = maximum time for the simulation. Default: tmax = 0.5
    #         errPlots = flag (1 or 0) to control whether error plots should be shown
    
    #Output: err = L2 norm of error evaluated at the spatial nodes on last time step
    #        x = location of finite difference nodes
    #        t = values of time at which solution is obtained (time nodes)
    #        U = matrix of solutions: U(:,j) is U(x) at t = t(j)
 
    #Compute mesh spacing and time step
    dx = L/(nx-1)
    dt = tmax/(nt-1)

    #Create arrays to save data for export
    x = np.linspace(0,L,nx)
    t = np.linspace(0,tmax,nt)
    u = np.zeros((nx,nt), dtype=float)

    #Set IC and BC
    q = np.zeros(nx)
    z = math.pi * x / L

    for i in range(0,nx):
        q[i] = math.sin(z[i])

    u[:,0] = q
    u0 = 0
    uL = 0

    #Coefficients of the tridiagonal system

    #subdiagonal a: coefficients of phi(i-1)
    a = (-alpha/2/dx**2) * np.ones((nx,1), dtype=float)

    #superdiagonal c: coefficients of phi(i+1)
    c = (-alpha/2/dx**2) * np.ones((nx,1), dtype=float)

    #diagonal b: coefficients of phi(i)
    b = (1/dt) * np.ones((nx, 1), dtype=float) - (a+c)

    #Fix coefficients of boundary nodes
    b[0] = 1
    c[0] = 0
    b[-1] = 1
    a[-1] = 0

    #Get LU factorization of coefficient matrix
    (e, f) = tridiagLU(a,b,c)

    #Loop over time steps
    for z in range(1,nt):
        #Right hand side includes time derivative and CN terms
        d = u[:, z-1] / dt
        m = d[1:-1]

        y = m - a[0] * u[0:-2, z-1] + (a[0] + c[1]) * u[1:-1, z-1] - c[1] * u[2:, z-1]
        d[1:-1] = y

        #overwrite BC values
        d[0] = 0
        d[-1] = 0

        #solve the system
        u[:,z] = tridiagLUsolve(d,a,e,f,u[:,z-1])

    #Compare with exact solution at end of simulation
    ue = q * math.exp(-t[nt-1] * alpha * (math.pi / L)**2)
    err = np.linalg.norm(u[:, nt-1] - ue)
    errgraph = u[:, nt-1] - ue

    print('Norm of error = ', err, ' at t = ', t[nt-1])
    print('tdt = ', dt, ' dx = ', dx)

    plt.title('heatCN', fontsize=20)
    plt.xlabel('x', fontsize=15)
    plt.ylabel('u', fontsize=15)
    plt.plot(x, ue, 'r-', linewidth=1, label='Exact')
    plt.plot(x, u[:,nt-1], 'b--o', linewidth=1, label='CN')
    plt.legend()
    return plt.show()