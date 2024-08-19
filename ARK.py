"""
Created on Fri Apr 12 11:46:41 2022

@author: Pranab JD
"""

from scipy.sparse import identity, linalg

class counter:
    def __init__(self):
        self.count = 0
    def incr(self,x): 
        self.count = self.count + 1

def GMRES(A, b, x0, tol):
    c = counter()
    return linalg.gmres(A, b, x0 = x0, callback = c.incr, tol = tol), c.count

def CG(A, b, x0, tol):
    c = counter()
    return linalg.cg(A, b, x0 = x0, callback = c.incr, tol = tol), c.count

def ARK2(u, dt, A, p1, p2, tol, *args):
    
    N = A.shape[0]

    U2, c2 = CG(identity(N) - p1*dt*A, u + dt*0.5*(p2-p1)*A.dot(u) + 0.5*dt*(args[0] + args[1]), u, tol)
    U3, c3 = CG(identity(N) - p1*dt*A, u + dt*(A.dot(u) - p1*A.dot(U2[0])) + 0.5*dt*(args[0] + args[1]), U2[0], tol)
    
    return U3[0], c2 + c3 + 3

def ARK4(u, dt, A, preconditioner, tol):
    
    N = A.shape[0]
    eigen_B = 2.0                   # Small value of penalisation parameter leads to instability in ARK4
    
    U2, c2 = GMRES(identity(N) - 0.5*eigen_B*dt*preconditioner, u + dt*(-1/2*eigen_B*preconditioner.dot(u) + 1/3*A.dot(u)), u, tol)

    U3, c3 = GMRES(identity(N) - 0.5*eigen_B*dt*preconditioner, u + dt*(-1/2*eigen_B*preconditioner.dot(U2[0]) + 1/6*A.dot(u) + 1/6*A.dot(U2[0])), U2[0], tol)

    U4, c4 = GMRES(identity(N) - 0.5*eigen_B*dt*preconditioner, u + dt*(1/4*eigen_B*preconditioner.dot(u) - 3/8*eigen_B*preconditioner.dot(U2[0]) - 3/8*eigen_B*preconditioner.dot(U3[0]) \
                                                                        + 1/8*A.dot(u) + 3/8*A.dot(U3[0])), U3[0], tol)
    
    U5, c5 = GMRES(identity(N) - 0.5*eigen_B*dt*preconditioner, u + dt*(-1/2*eigen_B*preconditioner.dot(U4[0]) + 1/8*A.dot(u) + 3/8*A.dot(U3[0])), U4[0], tol)

    U6, c6 = GMRES(identity(N) - 0.5*eigen_B*dt*preconditioner, u + dt*(-eigen_B*preconditioner.dot(u) + 9/2*eigen_B*preconditioner.dot(U3[0]) - 3*eigen_B*preconditioner.dot(U4[0]) - eigen_B*preconditioner.dot(U5[0]) \
                                                                        + 1/2*A.dot(u) - 3/2*A.dot(U3[0]) + A.dot(U4[0]) + A.dot(U5[0])), U5[0], tol)
    
    U7, c7 = GMRES(identity(N) - 2/3*eigen_B*dt*preconditioner, u + dt*(-2/3*eigen_B*preconditioner.dot(U6[0]) + 1/6*A.dot(u) + 2/3*A.dot(U5[0]) + 1/6*A.dot(U6[0])), U6[0], tol)
    
    return U7[0], c2 + c3 + c4 + c5 + c6 + c7 + 25