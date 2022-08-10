"""
Created on Fri Apr  1 14:06:43 2022

@author: Pranab JD
"""

import numpy as np
from scipy.sparse import lil_matrix, kron, identity, diags

from domain import Computational_Domain_2D

def Laplacian(Nx, Ny):
    
    Laplacian_x = lil_matrix(diags(np.ones(Nx - 1), -1) + diags(-2*np.ones(Nx), 0) + diags(np.ones(Nx - 1), 1))
    Laplacian_y = lil_matrix(diags(np.ones(Ny - 1), -1) + diags(-2*np.ones(Ny), 0) + diags(np.ones(Ny - 1), 1))
    
    Laplacian_x[0, -1] = 1; Laplacian_x[-1, 0] = 1
    Laplacian_y[-1, 0] = 1; Laplacian_x[0, -1] = 1
    
    return kron(identity(Ny), Laplacian_x) + kron(Laplacian_y, identity(Nx))
    
def RHS_Laplacian(u):
    
    print(np.shape(u)[0], np.shape(u)[1])
    
    eigen_B = 2
    Laplacian_matrix = Laplacian(np.shape(u)[0], np.shape(u)[1])
    
    
     
    return (eigen_B * Laplacian_matrix.dot(u)) + (Computational_Domain_2D.self.A_dif.dot(u) - (eigen_B * Laplacian_matrix.dot(u)))