"""
Created on Fri Apr 12 11:46:41 2022

@author: Pranab JD
"""

import numpy as np
from scipy.sparse import lil_matrix, kron, identity, diags, linalg

from domain import Computational_Domain_2D

class counter:
    def __init__(self):
        self.count = 0
    def incr(self,x):
        self.count = self.count + 1

def GMRES(A, b, x0, tol):
    
    c = counter()
    
    return linalg.gmres(A, b, x0 = x0, callback = c.incr, tol = tol)[0], c.count

def Laplacian(N):
    
    Laplacian_x = lil_matrix(diags(np.ones(N - 1), -1) + diags(-2*np.ones(N), 0) + diags(np.ones(N - 1), 1))
    Laplacian_y = lil_matrix(diags(np.ones(N - 1), -1) + diags(-2*np.ones(N), 0) + diags(np.ones(N - 1), 1))
    
    Laplacian_x[0, -1] = 1; Laplacian_x[-1, 0] = 1
    Laplacian_y[-1, 0] = 1; Laplacian_x[0, -1] = 1
    
    return kron(identity(N), Laplacian_x) + kron(Laplacian_y, identity(N))

def ARK2(u, dt, A, tol):
    
    N = A.shape[0]
    eigen_B = 2
    Laplacian_matrix = Laplacian(int(N**0.5))

    U2, c2 = GMRES(identity(N) - eigen_B*dt*Laplacian_matrix, u + 0.5*dt*(eigen_B*Laplacian_matrix.dot(u) + A.dot(u)), u, tol)
    U3, c3 = GMRES(identity(N) - eigen_B*dt*Laplacian_matrix, u + dt*(-eigen_B*Laplacian_matrix.dot(u) + eigen_B*Laplacian_matrix.dot(U2) + A.dot(U2)), u, tol)
    
    u_sol =  u + dt * (-eigen_B*Laplacian_matrix.dot(u) + eigen_B*Laplacian_matrix.dot(U2) - eigen_B*Laplacian_matrix.dot(U3) + A.dot(U2))
    
    return u_sol, c2 + c3

# def ARK2(u, dt, A, tol):
    
#     N = A.shape[0]
#     eigen_B = 2
#     Laplacian_matrix = Laplacian(int(N**0.5))

#     U2, c2 = GMRES(identity(N) - eigen_B*dt*Laplacian_matrix, u + 0.5*dt*(-2*eigen_B*Laplacian_matrix.dot(u) + A.dot(u)), u, tol)
#     U3, c3 = GMRES(identity(N) - eigen_B*dt*Laplacian_matrix, u + dt*(eigen_B*Laplacian_matrix.dot(u) - 2*eigen_B*Laplacian_matrix.dot(U2) + A.dot(U2)), u, tol)
    
#     u_sol = u + dt * (eigen_B*Laplacian_matrix.dot(u) - 2*eigen_B*Laplacian_matrix.dot(U2) + eigen_B*Laplacian_matrix.dot(U3) + A.dot(U2))
    
#     return u_sol, c2 + c3