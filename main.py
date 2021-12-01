"""
Created on Wed Dec  1 15:37:38 2021

@author: Pranab JD

Decription: Create required matrices and initialize parameters
"""


import numpy as np
from scipy.sparse import kron, identity

class Computational_Domain_2D:

    def __init__(self, N_x, N_y, tmax, eta_x, eta_y, error_tol):
        self.N_x = int(N_x)                     # Number of points along X
        self.N_y = int(N_y)                     # Number of points along Y
        self.xmin = 0                           # Min value of X
        self.xmax = 1                           # Max value of X
        self.ymin = 0                           # Min value of Y
        self.ymax = 1                           # Max value of Y
        self.eta_x = eta_x                      # Peclet number along X
        self.eta_y = eta_y                      # Peclet number along Y
        self.tmax = tmax                        # Maximum time
        self.error_tol = error_tol              # Maximum error permitted
        self.initialize_spatial_domain()
        self.initialize_parameters()
        self.initialize_matrices()
        
    ### Discretize the spatial domain
    def initialize_spatial_domain(self):
        self.dx = (self.xmax - self.xmin)/self.N_x
        self.dy = (self.ymax - self.ymin)/self.N_y
		## Periodic boundary conditions on [self.xmin, self.xmax] x [self.ymin, self.ymax]
        self.X = np.linspace(self.xmin, self.xmax, self.N_x, endpoint = False)
        self.Y = np.linspace(self.ymin, self.ymax, self.N_y, endpoint = False)
        self.X, self.Y = np.meshgrid(self.X, self.Y)
              
    ### Parameters  
    def initialize_parameters(self):
        self.dif_cfl = (self.dx**2 * self.dy**2)/(2 *(self.dx**2 + self.dy**2))
        print('Diffusion CFL: ', self.dif_cfl)
        print('Tolerance:', self.error_tol)
        self.dt = 0.9 * self.dif_cfl                      # N * CFL condition
        self.Fx = 1/self.dx**2                            # Fourier mesh number
        self.Fy = 1/self.dy**2                            # Fourier mesh number

    ### Operator matrices
    def initialize_matrices(self):
        self.Dif_x = np.zeros((self.N_x, self.N_x))       # Diffusion (X)
        self.Dif_y = np.zeros((self.N_y, self.N_y))       # Diffusion (Y)

        ## 2nd order centered difference
        self.Dif_x = np.diag(np.ones(self.N_x), 1) \
        + np.diag(np.ones(self.N_x), -1) + np.diag(-2*np.ones(self.N_x + 1), 0)
        
        self.Dif_y = np.diag(np.ones(self.N_y), 1) \
        + np.diag(np.ones(self.N_y), -1) + np.diag(-2*np.ones(self.N_y + 1), 0)
        
        ### Boundary conditions (Periodic)
        


        ### Merge X and Y to get a single matrix         
        self.A_dif = kron(identity(self.N_y), self.Fx * self.Dif_x) \
                   + kron(self.Fy * self.Dif_y, identity(self.N_x))