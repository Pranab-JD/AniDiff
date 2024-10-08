"""
Created on Fri Apr 12 11:46:41 2022

@author: Pranab JD
"""

from scipy.sparse import identity, linalg

from domain import Computational_Domain_2D

class counter:
    def __init__(self):
        self.count = 0
    def incr(self, x):
        self.count = self.count + 1

def GMRES(A, b, x0, tol):
    c = counter()
    return linalg.gmres(A, b, x0 = x0, callback = c.incr, tol = tol), c.count

def CG(A, b, x0, tol):
    c = counter()
    return linalg.cg(A, b, x0 = x0, callback = c.incr, tol = tol), c.count

def IMEX_Euler(u, dt, A, Laplacian, tol):

    eigen_B = 1
    
    u, iters = GMRES(identity(A.shape[0]) - eigen_B*dt*Laplacian, u + dt*(A.dot(u) - eigen_B*Laplacian.dot(u)), u, tol)
    
    return u[0], iters + 2

def Crank_Nicolson(u, dt, A, tol, *args):
    
    if args:
        u, iters = CG(identity(A.shape[0]) - 0.5*dt*A, u + 0.5*dt*A.dot(u) + 0.5*dt*(args[0] + args[1]), u, tol)
    else:
        u, iters = CG(identity(A.shape[0]) - 0.5*dt*A, u + 0.5*dt*A.dot(u), u, tol)
    
    return u[0], iters + 1