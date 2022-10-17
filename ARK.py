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
    return linalg.gmres(A, b, callback = c.incr, tol = tol)[0], c.count

def ARK2(u, dt, A, Laplacian, tol):
    
    N = A.shape[0]
    eigen_B = 2
    Laplacian_matrix = Laplacian()
    
    U2, c2 = GMRES(identity(N) - eigen_B*dt*Laplacian_matrix, u + 0.5*dt*(A.dot(u) - 2*eigen_B*Laplacian_matrix.dot(u)), u, tol)
    U3, c3 = GMRES(identity(N) - eigen_B*dt*Laplacian_matrix, u + dt*(eigen_B*Laplacian_matrix.dot(u) - 2*eigen_B*Laplacian_matrix.dot(U2) + A.dot(U2)), U2, tol)
    
    return U3, c2 + c3 + 5

def ARK4(u, dt, A, Laplacian, tol):
    
    N = A.shape[0]
    eigen_B = 10                    # Small value of penalisation parameter leads to instability in ARK4
    Laplacian_matrix = Laplacian()
    
    U2, c2 = GMRES(identity(N) - 0.5*eigen_B*dt*Laplacian_matrix, u + dt*(-1/2*eigen_B*Laplacian_matrix.dot(u) + 1/3*A.dot(u)), u, tol)
    U3, c3 = GMRES(identity(N) - 0.5*eigen_B*dt*Laplacian_matrix, u + dt*(-1/2*eigen_B*Laplacian_matrix.dot(U2) + 1/6*A.dot(u) + 1/6*A.dot(U2)), U2, tol)
    U4, c4 = GMRES(identity(N) - 0.5*eigen_B*dt*Laplacian_matrix, u + dt*(1/4*eigen_B*Laplacian_matrix.dot(u) - 3/8*eigen_B*Laplacian_matrix.dot(U2) - 3/8*eigen_B*Laplacian_matrix.dot(U3) \
                                                                        + 1/8*A.dot(u) + 3/8*A.dot(U3)), U3, tol)
    U5, c5 = GMRES(identity(N) - 0.5*eigen_B*dt*Laplacian_matrix, u + dt*(-1/2*eigen_B*Laplacian_matrix.dot(U4) + 1/8*A.dot(u) + 3/8*A.dot(U3)), U4, tol)
    U6, c6 = GMRES(identity(N) - 0.5*eigen_B*dt*Laplacian_matrix, u + dt*(-eigen_B*Laplacian_matrix.dot(u) + 9/2*eigen_B*Laplacian_matrix.dot(U3) - 3*eigen_B*Laplacian_matrix.dot(U4) - eigen_B*Laplacian_matrix.dot(U5) \
                                                                        + 1/2*A.dot(u) - 3/2*A.dot(U3) + A.dot(U4) + A.dot(U5)), U5, tol)
    U7, c7 = GMRES(identity(N) - 2/3*eigen_B*dt*Laplacian_matrix, u + dt*(-2/3*eigen_B*Laplacian_matrix.dot(U6) + 1/6*A.dot(u) + 2/3*A.dot(U5) + 1/6*A.dot(U6)), U6, tol)
    
    return U7, c2 + c3 + c4 + c5 + c6 + c7 + 25