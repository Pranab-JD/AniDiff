"""
Created on Fri Apr 12 11:46:41 2022

@author: Pranab JD
"""

import numpy as np
from scipy.sparse import lil_matrix, kron, identity, diags, linalg

class counter:
    def __init__(self):
        self.count = 0
    def incr(self,x):
        self.count = self.count + 1

def GMRES(A, b, x0, tol):
    
    c = counter()
    
    return linalg.gmres(A, b, x0 = x0, callback = c.incr, tol = tol)[0], c.count

def Crank_Nicolson(u, dt, A, tol):
    
    rhs = u + 0.5*dt*A.dot(u)
    u, iters = GMRES(identity(A.shape[0])-0.5*dt*A, rhs, u, tol)
    
    return u, iters + 1