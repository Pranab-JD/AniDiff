"""
Created on Mon Aug 22 17:55 2022

@author: Pranab JD
"""

import numpy as np
from scipy.sparse import kron, identity, linalg

def mu_mode(U, exp_Axx, exp_Ayy):

    N_x = np.shape(exp_Axx)[0]
    N_y = np.shape(exp_Ayy)[0]

    U_test = U.reshape(N_x, N_y)

    U_sol = exp_Axx.dot(exp_Ayy.dot(U_test).transpose()).transpose()

    U_sol = U_sol.reshape(N_x * N_y)

    return U_sol