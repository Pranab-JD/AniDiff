"""
Created on Mon Aug 22 17:55 2022

@author: Pranab JD
"""

import sys
import numpy as np
from scipy.sparse import kron, identity, linalg

### Import LeXInt
sys.path.insert(1, "./LeXInt/Python/")
from real_Leja_exp import *
from Eigenvalues import *

def mu_mode_Leja(U, dt, A_xx, A_yy, A_x, A_y, D_xx, D_yy, D_xy, D_yx, dx, dy, Leja_X, tol):

    N_x = np.shape(A_xx)[0]
    N_y = np.shape(A_yy)[0]

    U_test = U.reshape(N_x, N_y)

    exp_Axx = linalg.expm(A_xx/dx**2 * D_xx * dt)
    exp_Ayy = linalg.expm(A_yy/dy**2 * D_yy * dt)

    U_sol = exp_Axx.dot(exp_Ayy.dot(U_test).transpose()).transpose()

    U_sol = U_sol.reshape(N_x * N_y)
    
    A_drift_dif = kron(A_x, A_y) * D_xy/(4*dx*dy) + kron(A_y, A_x) * D_yx/(4*dx*dy)
    
    ### Eigenvalues (Remains constant for linear equations)
    eigen_min = 0.0
    eigen_max, eigen_imag = Gershgorin(A_drift_dif)      # Max real, imag eigenvalue
    
    ### Scaling and shifting factors
    c = 0.5 * (eigen_max + eigen_min)
    Gamma = 0.25 * (eigen_min - eigen_max)
    
    print(c, Gamma)
    
    def RHS_drift(u):
        return A_drift_dif.dot(u)
    
    u_sol, num_rhs_calls = real_Leja_exp(U, dt, RHS_drift, c, Gamma, Leja_X, 1e-3)
    
    # num_rhs_calls = 0

    return u_sol, num_rhs_calls