"""
Created on Mon Aug 22 17:55 2022

@author: Pranab JD
"""

import numpy as np
from scipy.sparse import kron, identity, linalg

def mu_mode(U, dt, A_xx, A_yy, D_xx, D_yy, dx, dy):
    
    N_x = np.shape(A_xx)[0]
    N_y = np.shape(A_yy)[0]
    
    # exp_A_xx = linalg.expm(kron(identity(N_y), A_xx/dx**2) * D_xx * dt)
    # exp_A_yy = linalg.expm(kron(A_yy/dy**2, identity(N_x)) * D_yy * dt)
    
    # U_sol = exp_A_yy.dot(exp_A_xx.dot(U))
    
    
    
    U_test = U.reshape(N_x, N_y)

    exp_Axx = linalg.expm(D_xx * A_xx/dx**2 * dt)
    exp_Ayy = linalg.expm(A_yy/dy**2 * D_yy * dt)
    
    U_sol = exp_Axx.dot(exp_Ayy.dot(U_test).transpose()).transpose()
    
    U_sol = U_sol.reshape(N_x * N_y)
    
    return U_sol