"""
Created on Mon Aug 22 17:55 2022

@author: Pranab JD
"""

import numpy as np
from scipy.sparse import linalg

def mu_mode_Leja(U, dt, eigen_B, A_xx, A_yy, D_xx, D_yy, dx, dy):

    N_x = np.shape(A_xx)[0]
    N_y = np.shape(A_yy)[0]

    U_test = U.reshape(N_x, N_y)

    exp_Axx = linalg.expm(A_xx/dx**2 * D_xx * dt * eigen_B)
    exp_Ayy = linalg.expm(A_yy/dy**2 * D_yy * dt * eigen_B)
    
    U_sol = exp_Axx.dot(exp_Ayy.dot(U_test).transpose()).transpose()

    U_sol = U_sol.reshape(N_x * N_y)

    return U_sol