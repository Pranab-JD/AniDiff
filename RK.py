"""
Created on Wed Sep 04 16:15:14 2020

@author: Pranab JD
"""

def RK2(RHS_function, u, dt):

    k1 = dt * RHS_function(u)
    k2 = dt * RHS_function(u + k1)

    u_rk2 = u + 1./2. * (k1 + k2)

    return u_rk2, 2

def RK4(RHS_function, u, dt, *t):

    k1 = dt * RHS_function(u, *t)
    k2 = dt * RHS_function(u + k1/2, *t + dt/2)
    k3 = dt * RHS_function(u + k2/2, *t + dt/2)
    k4 = dt * RHS_function(u + k3, *t + dt)

    u_rk4 = u + 1./6.*(k1 + 2*k2 + 2*k3 + k4)

    return u_rk4, 4

def RKF45(RHS_function, u, dt):

    k1 = dt * RHS_function(u)
    k2 = dt * RHS_function(u + k1/4)
    k3 = dt * RHS_function(u + 3./32.*k1 + 9./32.*k2)
    k4 = dt * RHS_function(u + 1932./2197.*k1 - 7200./2197.*k2 + 7296./2197.*k3)
    k5 = dt * RHS_function(u + 439./216.*k1 - 8*k2 + 3680./513.*k3 - 845./4104.*k4)
    k6 = dt * RHS_function(u - 8./27.*k1 + 2*k2 - 3544./2565.*k3 + 1859./4140.*k4 - 11./40.*k5)

    # u_rkf4 = u + (25./216.*k1 + 1408./2565.*k3 + 2197./4101.*k4 - 1./5.*k5)
    u_rkf5 = u + (16./135.*k1 + 6656./12825.*k3 + 28561./56430.*k4 - 9./50.*k5 + 2./55.*k6)

    return u_rkf5, 6