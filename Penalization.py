"""
Created on Fri Apr  1 14:06:43 2022

@author: Pranab JD
"""

import numpy as np
from scipy.sparse import lil_matrix, kron, identity, diags

# from domain import Computational_Domain_2D

def penalty(u, A):
    
    Nx, Ny = np.shape(u)
    
    Laplacian_x = lil_matrix(diags(np.ones(Nx - 1), -1) + diags(-2*np.ones(Nx), 0) + diags(np.ones(Nx - 1), 1))
    Laplacian_y = lil_matrix(diags(np.ones(Ny - 1), -1) + diags(-2*np.ones(Ny), 0) + diags(np.ones(Ny - 1), 1))
    
    Laplacian_x[0, -1] = 1; Laplacian_x[-1, 0] = 1
    Laplacian_y[-1, 0] = 1; Laplacian_x[0, -1] = 1
    
    Laplacian = kron(np.identity(Ny), Laplacian_x) + kron(Laplacian_y, np.identity(Nx))
    
