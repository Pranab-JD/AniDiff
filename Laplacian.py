"""
Created on Sun Nov 27 22:03:10 2022

@author: PJD
"""

import numpy as np
from scipy.sparse import lil_matrix, kron, identity, diags, linalg

N = 2**7

X = np.linspace(-1, 1, N, endpoint = False)
Y = np.linspace(-1, 1, N, endpoint = False)

dx = X[2] - X[1]
dy = Y[2] - Y[1]

X, Y = np.meshgrid(X, Y)

### Diffusion Coefficients
D_x = 1.0; D_y = 0.5

D_xx = D_x*D_x
D_xy = D_x*D_y
D_yx = D_y*D_x
D_yy = D_y*D_y

### 2nd order centered difference (1, -2, 1) & (-1/2, 0, 1/2)
Dif_xx = lil_matrix(diags(np.ones(N - 1), -1) + diags(-2*np.ones(N), 0) + diags(np.ones(N - 1), 1))
Dif_yy = lil_matrix(diags(np.ones(N - 1), -1) + diags(-2*np.ones(N), 0) + diags(np.ones(N - 1), 1))

Dif_x  = lil_matrix(diags(np.ones(N - 1), -1) + diags(-np.ones(N - 1), 1))
Dif_y  = lil_matrix(diags(np.ones(N - 1), -1) + diags(-np.ones(N - 1), 1))

### Boundary conditions (Periodic)

## Diagonal terms
Dif_xx[0, -1] = 1; Dif_xx[-1, 0] = 1      # (0, -1), (N-1, 0)
Dif_yy[0, -1] = 1; Dif_yy[-1, 0] = 1      # (0, -1), (N-1, 0)

## Off-diagonal terms
Dif_x[0, -1] = 1; Dif_x[-1, 0] = -1       # (0, -1), (N-1, 0)
Dif_y[0, -1] = 1; Dif_y[-1, 0] = -1       # (0, -1), (N-1, 0)

### Diffusion matrices
A_dif = kron(identity(N), Dif_xx/dx**2) * D_xx \
           + kron(Dif_yy/dy**2, identity(N)) * D_yy \
           + kron(Dif_x, Dif_y) * D_xy/(4*dx*dy) \
           + kron(Dif_y, Dif_x) * D_yx/(4*dx*dy)


A_lap = kron(identity(N), Dif_xx/dx**2) \
           + kron(Dif_yy/dy**2, identity(N))

### ----------------------------------------------------------------------- ###

## Constant diffusion on a periodic band (Crouseilles et al. 2015)
radius = (X**2 + Y**2)**0.5
u_init = np.zeros((N, N))

for ii in range(N):
    for jj in range(N):
        if radius[ii, jj] < 2*np.pi/5:
            u_init[ii, jj] = 1.0 + (3*np.exp(-2*radius[ii, jj]**2))
        else:
            u_init[ii, jj] = 1.0

u_init = u_init.reshape(N*N)

### ----------------------------------------------------------------------- ###

class counter:
    def __init__(self):
        self.count = 0
    def incr(self,x):
        self.count = self.count + 1

def GMRES(A, b, x0, tol):
    c = counter()
    return linalg.gmres(A, b, x0 = x0, callback = c.incr, tol = tol)[0], c.count

### ----------------------------------------------------------------------- ###

### Solve systems of linear equations
dt = 0.05
tol = 1e-8

### AniDiff matrix
u_dif = u_init
sol_dif, rhs_dif = GMRES(identity(N*N) - 0.5*dt*A_dif, u_dif + 0.5*dt*A_dif.dot(u_dif), u_dif, tol)

### Laplacian
u_lap = u_init
sol_lap, rhs_lap = GMRES(identity(N*N) - 0.5*dt*A_lap, u_lap + 0.5*dt*A_lap.dot(u_lap), u_lap, tol)

print()
print("RHS calls (AniDiff): ", rhs_dif)
print("RHS calls (Laplacian): ", rhs_lap)
print()
