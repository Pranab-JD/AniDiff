"""
Created on Wed Dec  1 15:37:38 2021

@author: Pranab JD

Description: Create required matrices and initialize parameters
"""

import numpy as np
from scipy.sparse import lil_matrix, kron, identity

### ============================================================================ ###

class Computational_Domain_2D:

    def __init__(self, N_x, N_y, spatial_order, tmax, error_tol):
        self.N_x = int(N_x)                         # Number of points along X
        self.N_y = int(N_y)                         # Number of points along Y
        self.xmin = -1                              # Min value of X
        self.xmax =  1                              # Max value of X
        self.ymin = -1                              # Min value of Y
        self.ymax =  1                              # Max value of Y
        
        self.spatial_order = int(spatial_order)     # Order of discretization
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
        self.dif_cfl = (self.dx**2 * self.dy**2)/(2 * (self.dx**2 + self.dy**2))
        self.dt = 0.1 * self.dif_cfl                # N * CFL condition
        print('Diffusion CFL: ', self.dif_cfl)
        print('Tolerance:', self.error_tol)
        
        ## Diffusion Tensor
        self.D_xx = 1; self.D_yy = 1; self.D_xy = 1; self.D_yx = 1
        self.D_coeff = np.array([(self.D_xx, self.D_xy), (self.D_yx, self.D_yy)])       
        
    ### Operator matrices
    def initialize_matrices(self):

        ## 2nd order centered difference (1, -2, 1)
        if self.spatial_order == 2:
            
            self.Dif_x =  lil_matrix(np.diag(np.ones(self.N_x - 1), -1) + np.diag(-2*np.ones(self.N_x), 0) + np.diag(np.ones(self.N_x - 1), 1))
            self.Dif_y =  lil_matrix(np.diag(np.ones(self.N_y - 1), -1) + np.diag(-2*np.ones(self.N_y), 0) + np.diag(np.ones(self.N_y - 1), 1))
            self.Dif_xy = lil_matrix(-1*np.diag(np.ones(self.N_x - 2), -2) + np.diag(-1*np.ones(self.N_x - 2), 2))
            
            ### Boundary conditions (Periodic)
            self.Dif_x[0, -1] = 1; self.Dif_x[self.N_x - 1, 0] = 1      # (0, -1), (N-1, 0)
            self.Dif_y[0, -1] = 1; self.Dif_y[self.N_y - 1, 0] = 1      # (0, -1), (N-1, 0)

            ### Merge X and Y to get a single matrix         
            self.A_dif = kron(self.Dif_x, identity(self.N_y)) * self.D_coeff[0, 0]/self.dx**2 \
                       + kron(identity(self.N_x), self.Dif_y) * self.D_coeff[1, 1]/self.dy**2 \
                    #    + kron(identity(self.N_x), self.Dif_xy) * (self.D_coeff[0, 1])/(self.dx*self.dy)
                           
            # ### Test matrix written to file
            # file_matrix = open("matrix_data.txt", 'w+')
            # np.savetxt(file_matrix, lil_matrix.todense(self.A_dif), fmt = '%.2f')
            # file_matrix.close()
            
        ### ------------------------------------------------- ###
            
        ## 4th order centered difference (−1/12, 16/12, −30/12,	16/12, −1/12)
        elif self.spatial_order == 4:
            
            self.Dif_x = lil_matrix(np.diag(-1/12 * np.ones(self.N_x - 2), -2) + np.diag(16/12 * np.ones(self.N_x - 1), -1) \
                                  + np.diag(-30/12 * np.ones(self.N_x), 0) \
                                  + np.diag(16/12 * np.ones(self.N_x - 1), 1) + np.diag(-1/12 * np.ones(self.N_x - 2), 2))
            
            self.Dif_y = lil_matrix(np.diag(-1/12 * np.ones(self.N_y - 2), -2) + np.diag(16/12 * np.ones(self.N_y - 1), -1) \
                                  + np.diag(-30/12 * np.ones(self.N_y), 0) \
                                  + np.diag(16/12 * np.ones(self.N_y - 1), 1) + np.diag(-1/12 * np.ones(self.N_y - 2), 2))
            
            ### Boundary conditions (Periodic)
            self.Dif_x[0, -2] = -1/12; self.Dif_x[0, -1] = 16/12                        # (0, -2), (0, -1)
            self.Dif_x[1, -1] = -1/12                                                   # (1, -1)
            self.Dif_x[self.N_x - 2, 0] = -1/12                                         # (N-2, 0)
            self.Dif_x[self.N_x - 1, 0] = 16/12; self.Dif_x[self.N_x - 1, 1] = -1/12    # (N-1, 0), (N-1, 1)
            
            self.Dif_y[0, -2] = -1/12; self.Dif_y[0, -1] = 16/12                        # (0, -2), (0, -1)
            self.Dif_y[1, -1] = -1/12                                                   # (1, -1)
            self.Dif_y[self.N_y - 2, 0] = -1/12                                         # (N-2, 0)
            self.Dif_y[self.N_y - 1, 0] = 16/12; self.Dif_y[self.N_y - 1, 1] = -1/12    # (N-1, 0), (N-1, 1)

            ### Merge X and Y to get a single matrix
            self.A_dif = kron(self.Dif_x, identity(self.N_y)) * self.D_coeff[0, 0]/self.dx**2 \
                       + kron(identity(self.N_x), self.Dif_y) * self.D_coeff[1, 1]/self.dy**2
            
        ### ------------------------------------------------- ###
            
        ## 6th order centered difference (1/90, −3/20, 3/2, −49/18,	3/2, −3/20,	1/90)
        elif self.spatial_order == 6:
            
            self.Dif_x = lil_matrix(np.diag(1/90 * np.ones(self.N_x - 3), -3) + np.diag(-3/20 * np.ones(self.N_x - 2), -2) + np.diag(3/2 * np.ones(self.N_x - 1), -1) \
                                  + np.diag(-49/18 * np.ones(self.N_x), 0) \
                                  + np.diag(3/2 * np.ones(self.N_x - 1), 1) + np.diag(-3/20 * np.ones(self.N_x - 2), 2) + np.diag(1/90 * np.ones(self.N_x - 3), 3))
            
            self.Dif_y = lil_matrix(np.diag(1/90 * np.ones(self.N_y - 3), -3) + np.diag(-3/20 * np.ones(self.N_y - 2), -2) + np.diag(3/2 * np.ones(self.N_y - 1), -1) \
                                  + np.diag(-49/18 * np.ones(self.N_y), 0) \
                                  + np.diag(3/2 * np.ones(self.N_y - 1), 1) + np.diag(-3/20 * np.ones(self.N_y - 2), 2) + np.diag(1/90 * np.ones(self.N_y - 3), 3))
            
            ### Boundary conditions (Periodic)
            self.Dif_x[0, -3] = 1/90; self.Dif_x[0, -2] = -3/20; self.Dif_x[0, -1] = 3/2                                    # (0, -3), (0, -2), (0, -1)
            self.Dif_x[1, -2] = 1/90; self.Dif_x[1, -1] = -3/20                                                             # (1, -2), (1, -1)
            self.Dif_x[2, -1] = 1/90                                                                                        # (2, -1)
            self.Dif_x[self.N_x - 3, 0] = 1/90                                                                              # (N-3, 0)
            self.Dif_x[self.N_x - 2, 0] = -3/20; self.Dif_x[self.N_x - 2, 1] = 1/90                                         # (N-2, 0), (N-2, 1)
            self.Dif_x[self.N_x - 1, 0] = 3/2; self.Dif_x[self.N_x - 1, 1] = -3/20; self.Dif_x[self.N_x - 1, 2] = 1/90      # (N-1, 0), (N-1, 1), (N-1, 2)
            
            self.Dif_y[0, -3] = 1/90; self.Dif_y[0, -2] = -3/20; self.Dif_y[0, -1] = 3/2                                    # (0, -3), (0, -2), (0, -1)
            self.Dif_y[1, -2] = 1/90; self.Dif_y[1, -1] = -3/20                                                             # (1, -2), (1, -1)
            self.Dif_y[2, -1] = 1/90                                                                                        # (2, -1)
            self.Dif_y[self.N_y - 3, 0] = 1/90                                                                              # (N-3, 0)
            self.Dif_y[self.N_y - 2, 0] = -3/20; self.Dif_y[self.N_y - 2, 1] = 1/90                                         # (N-2, 0), (N-2, 1)
            self.Dif_y[self.N_y - 1, 0] = 3/2; self.Dif_y[self.N_y - 1, 1] = -3/20; self.Dif_y[self.N_y - 1, 2] = 1/90      # (N-1, 0), (N-1, 1), (N-1, 2)
            
            ### Merge X and Y to get a single matrix
            self.A_dif = kron(self.Dif_x, identity(self.N_y)) * self.D_coeff[0, 0]/self.dx**2 \
                       + kron(identity(self.N_x), self.Dif_y) * self.D_coeff[1, 1]/self.dy**2
            
        ### ------------------------------------------------- ###
            
        ## 8th order centered difference (−1/560, 8/315, −1/5, 8/5,	−205/72, 8/5, −1/5,	8/315, −1/560)
        elif self.spatial_order == 8:
            
            self.Dif_x = lil_matrix(np.diag(-1/560 * np.ones(self.N_x - 4), -4) + np.diag(8/315 * np.ones(self.N_x - 3), -3) \
                                  + np.diag(-1/5 * np.ones(self.N_x - 2), -2) + np.diag(8/5 * np.ones(self.N_x - 1), -1) \
                                  + np.diag(-205/72 * np.ones(self.N_x), 0) \
                                  + np.diag(8/5 * np.ones(self.N_x - 1), 1) + np.diag(-1/5 * np.ones(self.N_x - 2), 2) \
                                  + np.diag(8/315 * np.ones(self.N_x - 3), 3) + np.diag(-1/560 * np.ones(self.N_x - 4), 4))
            
            self.Dif_y = lil_matrix(np.diag(-1/560 * np.ones(self.N_y - 4), -4) + np.diag(8/315 * np.ones(self.N_y - 3), -3) \
                                  + np.diag(-1/5 * np.ones(self.N_y - 2), -2) + np.diag(8/5 * np.ones(self.N_y - 1), -1) \
                                  + np.diag(-205/72 * np.ones(self.N_y), 0) \
                                  + np.diag(8/5 * np.ones(self.N_y - 1), 1) + np.diag(-1/5 * np.ones(self.N_y - 2), 2) \
                                  + np.diag(8/315 * np.ones(self.N_y - 3), 3) + np.diag(-1/560 * np.ones(self.N_y - 4), 4))
                    
            ### Boundary conditions (Periodic)
            ## (0, -4), (0, -3), (0, -2), (0, -1)
            self.Dif_x[0, -4] = -1/560; self.Dif_x[0, -3] = 8/315; self.Dif_x[0, -2] = -1/5; self.Dif_x[0, -1] = 8/5
            ## (1, -3), (1, -2), (1, -1)           
            self.Dif_x[1, -3] = -1/560; self.Dif_x[1, -2] = 8/315; self.Dif_x[1, -1] = -1/5
            ## (2, -2), (2, -1)
            self.Dif_x[2, -2] = -1/560; self.Dif_x[2, -1] = 8/315
            ## (3, -1)
            self.Dif_x[3, -1] = -1/560
            ## (N-4, 0)
            self.Dif_x[self.N_x - 4, 0] = -1/560
            ## (N-3, 0), (N-3, 1)
            self.Dif_x[self.N_x - 3, 0] = 8/315; self.Dif_x[self.N_x - 3, 1] = -1/560
            ## (N-2, 0), (N-2, 1), (N-2, 2)
            self.Dif_x[self.N_x - 2, 0] = -1/5; self.Dif_x[self.N_x - 2, 1] = 8/315; self.Dif_x[self.N_x - 2, 2] = -1/560
            ## (N-1, 0), (N-1, 1), (N-1, 2), (N-1, 3)
            self.Dif_x[self.N_x - 1, 0] = 8/5; self.Dif_x[self.N_x - 1, 1] = -1/5; self.Dif_x[self.N_x - 1, 2] = 8/315; self.Dif_x[self.N_x - 1, 3] = -1/560
            
            ## (0, -4), (0, -3), (0, -2), (0, -1)
            self.Dif_y[0, -4] = -1/560; self.Dif_y[0, -3] = 8/315; self.Dif_y[0, -2] = -1/5; self.Dif_y[0, -1] = 8/5
            ## (1, -3), (1, -2), (1, -1)           
            self.Dif_y[1, -3] = -1/560; self.Dif_y[1, -2] = 8/315; self.Dif_y[1, -1] = -1/5
            ## (2, -2), (2, -1)
            self.Dif_y[2, -2] = -1/560; self.Dif_y[2, -1] = 8/315
            ## (3, -1)
            self.Dif_y[3, -1] = -1/560
            ## (N-4, 0)
            self.Dif_y[self.N_y - 4, 0] = -1/560
            ## (N-3, 0), (N-3, 1)
            self.Dif_y[self.N_y - 3, 0] = 8/315; self.Dif_y[self.N_y - 3, 1] = -1/560
            ## (N-2, 0), (N-2, 1), (N-2, 2)
            self.Dif_y[self.N_y - 2, 0] = -1/5; self.Dif_y[self.N_y - 2, 1] = 8/315; self.Dif_y[self.N_y - 2, 2] = -1/560
            ## (N-1, 0), (N-1, 1), (N-1, 2), (N-1, 3)
            self.Dif_y[self.N_y - 1, 0] = 8/5; self.Dif_y[self.N_y - 1, 1] = -1/5; self.Dif_y[self.N_y - 1, 2] = 8/315; self.Dif_y[self.N_y - 1, 3] = -1/560
            
            ### Merge X and Y to get a single matrix
            self.A_dif = kron(self.Dif_x, identity(self.N_y)) * self.D_coeff[0, 0]/self.dx**2 \
                       + kron(identity(self.N_x), self.Dif_y) * self.D_coeff[1, 1]/self.dy**2
            
        ### ------------------------------------------------- ###
                
        else:
            print("Incorrect spatial order! Please choose either '2' or '4' or '6' or '8'.")
            
### ============================================================================ ###