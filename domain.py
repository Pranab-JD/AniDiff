"""
Created on Wed Dec  1 15:37:38 2021

@author: Pranab JD

Decription: Create required matrices and initialize parameters
"""


import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, kron, identity

class Computational_Domain_2D:

    def __init__(self, N_x, N_y, spatial_order, tmax, error_tol):
        self.N_x = int(N_x)                         # Number of points along X
        self.N_y = int(N_y)                         # Number of points along Y
        self.spatial_order = int(spatial_order)     # Order of discretization
        self.xmin = 0                               # Min value of X
        self.xmax = 1                               # Max value of X
        self.ymin = 0                               # Min value of Y
        self.ymax = 1                               # Max value of Y
        self.tmax = tmax                            # Simulation time
        self.error_tol = error_tol                  # Maximum error permitted
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
        self.dt = 0.9 * self.dif_cfl                # N * CFL condition
        self.Fx = 1/self.dx**2                      # Fourier mesh number
        self.Fy = 1/self.dy**2                      # Fourier mesh number

    ### Operator matrices
    def initialize_matrices(self):

        ## 2nd order centered difference (1, -2, 1)
        if self.spatial_order == 2:
            
            self.Dif_x = csr_matrix(np.diag(np.ones(self.N_x - 1), -1) + np.diag(-2*np.ones(self.N_x), 0) + np.diag(np.ones(self.N_x - 1), 1))
            self.Dif_y = csr_matrix(np.diag(np.ones(self.N_y - 1), -1) + np.diag(-2*np.ones(self.N_y), 0) + np.diag(np.ones(self.N_y - 1), 1))
            
            ### Boundary conditions (Periodic)
            self.Dif_x[0, -1] = 1; self.Dif_x[self.N_x - 1, 0] = 1
            self.Dif_y[0, -1] = 1; self.Dif_y[self.N_y - 1, 0] = 1

            ### Merge X and Y to get a single matrix         
            self.A_dif = kron(identity(self.N_y), self.Fx * self.Dif_x) + kron(self.Fy * self.Dif_y, identity(self.N_x))
            
        ## 4th order centered difference (−1/12, 4/3, −5/2,	4/3, −1/12)
        elif self.spatial_order == 4:
            
            self.Dif_x = lil_matrix(np.diag(-1/12 * np.ones(self.N_x - 2), -2) + np.diag(4/3 * np.ones(self.N_x - 1), -1) \
                                  + np.diag(-5/2 * np.ones(self.N_x), 0) \
                                  + np.diag(4/3 * np.ones(self.N_x - 1), 1) + np.diag(-1/12 * np.ones(self.N_x - 2), 2))
            
            
            self.Dif_y = lil_matrix(np.diag(-1/12 * np.ones(self.N_y - 2), -2) + np.diag(4/3 * np.ones(self.N_y - 1), -1) \
                                  + np.diag(-5/2 * np.ones(self.N_y), 0) \
                                  + np.diag(4/3 * np.ones(self.N_y - 1), 1) + np.diag(-1/12 * np.ones(self.N_y - 2), 2))
            
            ### Boundary conditions (Periodic)
            self.Dif_x[0, -2] = -1/12; self.Dif_x[0, -1] = 4/3; self.Dif_x[1, -1] = -1/12
            self.Dif_x[self.N_x - 1, 0] = 4/3; self.Dif_x[self.N_x - 1, 1] = -1/12; self.Dif_x[self.N_x - 2, 0] = -1/12
            self.Dif_y[0, -2] = -1/12; self.Dif_y[0, -1] = 4/3; self.Dif_y[1, -1] = -1/12
            self.Dif_y[self.N_y - 1, 0] = 4/3; self.Dif_y[self.N_y - 1, 1] = -1/12; self.Dif_y[self.N_y - 2, 0] = -1/12

            ### Merge X and Y to get a single matrix         
            self.A_dif = kron(identity(self.N_y), self.Fx * self.Dif_x) + kron(self.Fy * self.Dif_y, identity(self.N_x))