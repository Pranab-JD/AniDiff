"""
Created on Wed Dec  1 15:37:38 2021

@author: Pranab JD

Description: Create required matrices and initialize parameters
"""

import numpy as np
from scipy.sparse import lil_matrix, kron, identity, diags

### ============================================================================ ###

class Computational_Domain_2D:

    def __init__(self, N_x, N_y, spatial_order, tmax, N_cfl, error_tol):
        self.N_x = int(N_x)                         # Number of points along X
        self.N_y = int(N_y)                         # Number of points along Y
        self.xmin = -1                              # Min value of X
        self.xmax =  1                              # Max value of X
        self.ymin = -1                              # Min value of Y
        self.ymax =  1                              # Max value of Y
        
        self.spatial_order = int(spatial_order)     # Order of discretization
        self.tmax = tmax                            # Simulation time
        self.N_cfl = N_cfl                          # "N" times CFL limit
        self.error_tol = error_tol                  # Maximum error permitted
        
        self.initialize_spatial_domain()
        self.initialize_parameters()
        self.initialize_matrices()
        
    ### Discretize the spatial domain
    def initialize_spatial_domain(self):

		## Periodic boundary conditions
        self.X = np.linspace(self.xmin, self.xmax, self.N_x, endpoint = False)
        self.Y = np.linspace(self.ymin, self.ymax, self.N_y, endpoint = False)
        
        self.dx = self.X[2] - self.X[1]
        self.dy = self.Y[2] - self.Y[1]
        
        self.X, self.Y = np.meshgrid(self.X, self.Y)
        
        self.radius = (self.X**2 + self.Y**2)**0.5
        
    ### Parameters  
    def initialize_parameters(self):
        self.dif_cfl = (self.dx**2 * self.dy**2)/(2*(self.dx**2 + self.dy**2))
        self.dt = self.N_cfl * self.dif_cfl                # N * CFL condition
        print()
        print("==============================================")
        print("Nx x Ny      :", self.N_x, "x", self.N_y)
        print("CFL time     : {:e}".format(self.dif_cfl))
        print("N times CFL  :", self.N_cfl)
        print("Tolerance    :", self.error_tol)
        print("Spatial Order:", self.spatial_order)
        print()
        
        ### Diffusion Coefficients
        self.D_xx =  1
        self.D_xy =  0
        self.D_yx =  0
        self.D_yy =  0

    ### Operator matrices
    def initialize_matrices(self):

        ### 2nd order centered difference (1, -2, 1) & (-1/2, 0, 1/2)
        if self.spatial_order == 2:
            
            self.Dif_xx = lil_matrix(diags(np.ones(self.N_x - 1), -1) + diags(-2*np.ones(self.N_x), 0) + diags(np.ones(self.N_x - 1), 1))
            self.Dif_yy = lil_matrix(diags(np.ones(self.N_y - 1), -1) + diags(-2*np.ones(self.N_y), 0) + diags(np.ones(self.N_y - 1), 1))
            
            self.Dif_x  = lil_matrix(diags(np.ones(self.N_x - 1), -1) + diags(-np.ones(self.N_x - 1), 1))
            self.Dif_y  = lil_matrix(diags(np.ones(self.N_y - 1), -1) + diags(-np.ones(self.N_y - 1), 1))
            
            ### Boundary conditions (Periodic)
            
            ## Diagonal terms
            self.Dif_xx[0, -1] = 1; self.Dif_xx[-1, 0] = 1      # (0, -1), (N-1, 0)
            self.Dif_yy[0, -1] = 1; self.Dif_yy[-1, 0] = 1      # (0, -1), (N-1, 0)
            
            ## Off-diagonal terms
            self.Dif_x[0, -1] = 1; self.Dif_x[-1, 0] = -1       # (0, -1), (N-1, 0)
            self.Dif_y[0, -1] = 1; self.Dif_y[-1, 0] = -1       # (0, -1), (N-1, 0)
        
            ### Space independent diffusion coefficients               
            self.A_dif = kron(identity(self.N_y), self.Dif_xx/self.dx**2) * self.D_xx \
                       + kron(self.Dif_yy/self.dy**2, identity(self.N_x)) * self.D_yy \
                       + kron(self.Dif_x, self.Dif_y) * self.D_xy/(4*self.dx*self.dy) \
                       + kron(self.Dif_y, self.Dif_x) * self.D_yx/(4*self.dx*self.dy)
            
            ### Space dependent diffusion coefficients
            # self.Dif_x = self.Dif_x.multiply(self.X)
            # self.Dif_y = self.Dif_y.multiply(-self.Y)

            # ### Merge X and Y to get a single matrix         
            # self.A_dif = kron(identity(self.N_y).multiply(self.D_xx.diagonal()), self.Dif_xx/self.dx**2) \
            #            + kron(self.Dif_yy/self.dy**2, identity(self.N_x).multiply(self.D_yy.diagonal())) \
            #            + kron(self.Dif_x, self.Dif_y)/(4*self.dx*self.dy) \
            #            + kron(self.Dif_y, self.Dif_x)/(4*self.dx*self.dy)

            
        ### ------------------------------------------------- ###
            
        ### 4th order centered difference (−1/12, 16/12, −30/12, 16/12, −1/12) & (1/12, -2/3, 0, 2/3, -1/12)
        # elif self.spatial_order == 4:
            
        #     self.Dif_xx = lil_matrix(diags(-1/12 * np.ones(self.N_x - 2), -2) + diags(16/12 * np.ones(self.N_x - 1), -1) \
        #                            + diags(-30/12 * np.ones(self.N_x), 0) \
        #                            + diags(16/12 * np.ones(self.N_x - 1), 1) + diags(-1/12 * np.ones(self.N_x - 2), 2))
            
        #     self.Dif_yy = lil_matrix(diags(-1/12 * np.ones(self.N_y - 2), -2) + diags(16/12 * np.ones(self.N_y - 1), -1) \
        #                            + diags(-30/12 * np.ones(self.N_y), 0) \
        #                            + diags(16/12 * np.ones(self.N_y - 1), 1) + diags(-1/12 * np.ones(self.N_y - 2), 2))
            
        #     self.Dif_x  = lil_matrix(diags(1/12 * np.ones(self.N_x - 2), -2) + diags(-2/3 * np.ones(self.N_x - 1), -1) \
        #                            + diags(2/3 * np.ones(self.N_x - 1), 1) + diags(-1/12 * np.ones(self.N_x - 2), 2))
            
        #     self.Dif_y  = lil_matrix(diags(1/12 * np.ones(self.N_y - 2), -2) + diags(-2/3 * np.ones(self.N_y - 1), -1) \
        #                            + diags(2/3 * np.ones(self.N_y - 1), 1) + diags(-1/12 * np.ones(self.N_y - 2), 2))
            
            
        #     ### Boundary conditions (Periodic)
            
        #     ## Diagonal terms
        #     self.Dif_xx[0, -2] = -1/12; self.Dif_xx[0, -1] = 16/12                        # (0, -2), (0, -1)
        #     self.Dif_xx[1, -1] = -1/12                                                    # (1, -1)
        #     self.Dif_xx[self.N_x - 2, 0] = -1/12                                          # (N-2, 0)
        #     self.Dif_xx[self.N_x - 1, 0] = 16/12; self.Dif_xx[self.N_x - 1, 1] = -1/12    # (N-1, 0), (N-1, 1)
            
        #     self.Dif_yy[0, -2] = -1/12; self.Dif_yy[0, -1] = 16/12                        # (0, -2), (0, -1)
        #     self.Dif_yy[1, -1] = -1/12                                                    # (1, -1)
        #     self.Dif_yy[self.N_y - 2, 0] = -1/12                                          # (N-2, 0)
        #     self.Dif_yy[self.N_y - 1, 0] = 16/12; self.Dif_yy[self.N_y - 1, 1] = -1/12    # (N-1, 0), (N-1, 1)
            
        #     ## Off-diagonal terms
        #     self.Dif_x[0, -2] = 1/12; self.Dif_x[0, -1] = -2/3                            # (0, -2), (0, -1)
        #     self.Dif_x[1, -1] = 1/12                                                      # (1, -1)
        #     self.Dif_x[self.N_x - 2, 0] = -1/12                                           # (N-2, 0)
        #     self.Dif_x[self.N_x - 1, 0] = 2/3; self.Dif_x[self.N_x - 1, 1] = -1/12        # (N-1, 0), (N-1, 1)
            
        #     self.Dif_y[0, -2] = 1/12; self.Dif_y[0, -1] = -2/3                            # (0, -2), (0, -1)
        #     self.Dif_y[1, -1] = 1/12                                                      # (1, -1)
        #     self.Dif_y[self.N_y - 2, 0] = -1/12                                           # (N-2, 0)
        #     self.Dif_y[self.N_y - 1, 0] = 2/3; self.Dif_y[self.N_y - 1, 1] = -1/12        # (N-1, 0), (N-1, 1)
            
        # #     ### Merge X and Y to get a single matrix
        #     self.A_dif = kron(identity(self.N_y), self.Dif_xx) * self.D_coeff[0, 0]/self.dx**2 \
        #                + kron(self.Dif_yy, identity(self.N_x)) * self.D_coeff[1, 1]/self.dy**2 \
        #                + kron(self.Dif_y, self.Dif_x) * (self.D_coeff[0, 1]+self.D_coeff[1, 0])/(self.dx*self.dy)
            
        # ### ------------------------------------------------- ###
            
        # ### 6th order centered difference (1/90, −3/20, 3/2, −49/18, 3/2, −3/20, 1/90) & (-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60)
        # elif self.spatial_order == 6:
            
        #     self.Dif_xx = lil_matrix(diags(1/90 * np.ones(self.N_x - 3), -3) + diags(-3/20 * np.ones(self.N_x - 2), -2) + diags(3/2 * np.ones(self.N_x - 1), -1) \
        #                            + diags(-49/18 * np.ones(self.N_x), 0) \
        #                            + diags(3/2 * np.ones(self.N_x - 1), 1) + diags(-3/20 * np.ones(self.N_x - 2), 2) + diags(1/90 * np.ones(self.N_x - 3), 3))
            
        #     self.Dif_yy = lil_matrix(diags(1/90 * np.ones(self.N_y - 3), -3) + diags(-3/20 * np.ones(self.N_y - 2), -2) + diags(3/2 * np.ones(self.N_y - 1), -1) \
        #                            + diags(-49/18 * np.ones(self.N_y), 0) \
        #                            + diags(3/2 * np.ones(self.N_y - 1), 1) + diags(-3/20 * np.ones(self.N_y - 2), 2) + diags(1/90 * np.ones(self.N_y - 3), 3))
            
        #     self.Dif_x  = lil_matrix(diags(-1/60 * np.ones(self.N_x - 3), -3) + diags(3/20 * np.ones(self.N_x - 2), -2) + diags(-3/4 * np.ones(self.N_x - 1), -1) \
        #                            + diags(3/4 * np.ones(self.N_x - 1), 1) + diags(-3/20 * np.ones(self.N_x - 2), 2) + diags(1/60 * np.ones(self.N_x - 3), 3))
            
        #     self.Dif_y  = lil_matrix(diags(-1/60 * np.ones(self.N_y - 3), -3) + diags(3/20 * np.ones(self.N_y - 2), -2) + diags(-3/4 * np.ones(self.N_y - 1), -1) \
        #                            + diags(3/4 * np.ones(self.N_y - 1), 1) + diags(-3/20 * np.ones(self.N_y - 2), 2) + diags(1/60 * np.ones(self.N_y - 3), 3))
            
            
        #     ### Boundary conditions (Periodic)
            
        #     ## Diagonal terms
        #     self.Dif_xx[0, -3] = 1/90; self.Dif_xx[0, -2] = -3/20; self.Dif_xx[0, -1] = 3/2                                    # (0, -3), (0, -2), (0, -1)
        #     self.Dif_xx[1, -2] = 1/90; self.Dif_xx[1, -1] = -3/20                                                              # (1, -2), (1, -1)
        #     self.Dif_xx[2, -1] = 1/90                                                                                          # (2, -1)
        #     self.Dif_xx[self.N_x - 3, 0] = 1/90                                                                                # (N-3, 0)
        #     self.Dif_xx[self.N_x - 2, 0] = -3/20; self.Dif_xx[self.N_x - 2, 1] = 1/90                                          # (N-2, 0), (N-2, 1)
        #     self.Dif_xx[self.N_x - 1, 0] = 3/2; self.Dif_xx[self.N_x - 1, 1] = -3/20; self.Dif_xx[self.N_x - 1, 2] = 1/90      # (N-1, 0), (N-1, 1), (N-1, 2)
            
        #     self.Dif_yy[0, -3] = 1/90; self.Dif_yy[0, -2] = -3/20; self.Dif_yy[0, -1] = 3/2                                    # (0, -3), (0, -2), (0, -1)
        #     self.Dif_yy[1, -2] = 1/90; self.Dif_yy[1, -1] = -3/20                                                              # (1, -2), (1, -1)
        #     self.Dif_yy[2, -1] = 1/90                                                                                          # (2, -1)
        #     self.Dif_yy[self.N_y - 3, 0] = 1/90                                                                                # (N-3, 0)
        #     self.Dif_yy[self.N_y - 2, 0] = -3/20; self.Dif_yy[self.N_y - 2, 1] = 1/90                                          # (N-2, 0), (N-2, 1)
        #     self.Dif_yy[self.N_y - 1, 0] = 3/2; self.Dif_yy[self.N_y - 1, 1] = -3/20; self.Dif_yy[self.N_y - 1, 2] = 1/90      # (N-1, 0), (N-1, 1), (N-1, 2)
            
        #     ## Off-diagonal terms
        #     self.Dif_x[0, -3] = -1/60; self.Dif_x[0, -2] = 3/20; self.Dif_x[0, -1] = -3/4                                      # (0, -3), (0, -2), (0, -1)
        #     self.Dif_x[1, -2] = -1/60; self.Dif_x[1, -1] = 3/20                                                                # (1, -2), (1, -1)
        #     self.Dif_x[2, -1] = -1/60                                                                                          # (2, -1)
        #     self.Dif_x[self.N_x - 3, 0] = 1/60                                                                                 # (N-3, 0)
        #     self.Dif_x[self.N_x - 2, 0] = -3/20; self.Dif_x[self.N_x - 2, 1] = 1/60                                            # (N-2, 0), (N-2, 1)
        #     self.Dif_x[self.N_x - 1, 0] = 3/4; self.Dif_x[self.N_x - 1, 1] = -3/20; self.Dif_x[self.N_x - 1, 2] = 1/60         # (N-1, 0), (N-1, 1), (N-1, 2)
            
        #     self.Dif_y[0, -3] = -1/60; self.Dif_y[0, -2] = 3/20; self.Dif_y[0, -1] = -3/4                                      # (0, -3), (0, -2), (0, -1)
        #     self.Dif_y[1, -2] = -1/60; self.Dif_y[1, -1] = 3/20                                                                # (1, -2), (1, -1)
        #     self.Dif_y[2, -1] = -1/60                                                                                          # (2, -1)
        #     self.Dif_y[self.N_y - 3, 0] = 1/60                                                                                 # (N-3, 0)
        #     self.Dif_y[self.N_y - 2, 0] = -3/20; self.Dif_y[self.N_y - 2, 1] = 1/60                                            # (N-2, 0), (N-2, 1)
        #     self.Dif_y[self.N_y - 1, 0] = 3/4; self.Dif_y[self.N_y - 1, 1] = -3/20; self.Dif_y[self.N_y - 1, 2] = 1/60         # (N-1, 0), (N-1, 1), (N-1, 2)
            
        #     ### Merge X and Y to get a single matrix
        #     self.A_dif = kron(identity(self.N_y), self.Dif_xx) * self.D_coeff[0, 0]/self.dx**2 \
        #                + kron(self.Dif_yy, identity(self.N_x)) * self.D_coeff[1, 1]/self.dy**2 \
        #                + kron(self.Dif_y, self.Dif_x) * (self.D_coeff[0, 1]+self.D_coeff[1, 0])/(self.dx*self.dy)
            
        # ### ------------------------------------------------- ###
            
        # ### 8th order centered difference (−1/560, 8/315, −1/5, 8/5, −205/72, 8/5, −1/5, 8/315, −1/560) & (1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280)
        # elif self.spatial_order == 8:
            
        #     self.Dif_xx = lil_matrix(diags(-1/560 * np.ones(self.N_x - 4), -4) + diags(8/315 * np.ones(self.N_x - 3), -3) \
        #                            + diags(-1/5 * np.ones(self.N_x - 2), -2) + diags(8/5 * np.ones(self.N_x - 1), -1) \
        #                            + diags(-205/72 * np.ones(self.N_x), 0) \
        #                            + diags(8/5 * np.ones(self.N_x - 1), 1) + diags(-1/5 * np.ones(self.N_x - 2), 2) \
        #                            + diags(8/315 * np.ones(self.N_x - 3), 3) + diags(-1/560 * np.ones(self.N_x - 4), 4))
            
        #     self.Dif_yy = lil_matrix(diags(-1/560 * np.ones(self.N_y - 4), -4) + diags(8/315 * np.ones(self.N_y - 3), -3) \
        #                            + diags(-1/5 * np.ones(self.N_y - 2), -2) + diags(8/5 * np.ones(self.N_y - 1), -1) \
        #                            + diags(-205/72 * np.ones(self.N_y), 0) \
        #                            + diags(8/5 * np.ones(self.N_y - 1), 1) + diags(-1/5 * np.ones(self.N_y - 2), 2) \
        #                            + diags(8/315 * np.ones(self.N_y - 3), 3) + diags(-1/560 * np.ones(self.N_y - 4), 4))
            
        #     self.Dif_x  = lil_matrix(diags(1/280 * np.ones(self.N_x - 4), -4) + diags(-4/105 * np.ones(self.N_x - 3), -3) \
        #                            + diags(1/5 * np.ones(self.N_x - 2), -2) + diags(-4/5 * np.ones(self.N_x - 1), -1) \
        #                            + diags(4/5 * np.ones(self.N_x - 1), 1) + diags(-1/5 * np.ones(self.N_x - 2), 2) \
        #                            + diags(4/105 * np.ones(self.N_x - 3), 3) + diags(-1/280 * np.ones(self.N_x - 4), 4))
            
        #     self.Dif_y  = lil_matrix(diags(1/280 * np.ones(self.N_y - 4), -4) + diags(-4/105 * np.ones(self.N_y - 3), -3) \
        #                            + diags(1/5 * np.ones(self.N_y - 2), -2) + diags(-4/5 * np.ones(self.N_y - 1), -1) \
        #                            + diags(4/5 * np.ones(self.N_y - 1), 1) + diags(-1/5 * np.ones(self.N_y - 2), 2) \
        #                            + diags(4/105 * np.ones(self.N_y - 3), 3) + diags(-1/280 * np.ones(self.N_y - 4), 4))
                    
                    
        #     ### Boundary conditions (Periodic)
            
        #     ## Diagonal terms
        #     ## (0, -4), (0, -3), (0, -2), (0, -1)
        #     self.Dif_x[0, -4] = -1/560; self.Dif_x[0, -3] = 8/315; self.Dif_x[0, -2] = -1/5; self.Dif_x[0, -1] = 8/5
        #     ## (1, -3), (1, -2), (1, -1)           
        #     self.Dif_x[1, -3] = -1/560; self.Dif_x[1, -2] = 8/315; self.Dif_x[1, -1] = -1/5
        #     ## (2, -2), (2, -1)
        #     self.Dif_x[2, -2] = -1/560; self.Dif_x[2, -1] = 8/315
        #     ## (3, -1)
        #     self.Dif_x[3, -1] = -1/560
        #     ## (N-4, 0)
        #     self.Dif_x[self.N_x - 4, 0] = -1/560
        #     ## (N-3, 0), (N-3, 1)
        #     self.Dif_x[self.N_x - 3, 0] = 8/315; self.Dif_x[self.N_x - 3, 1] = -1/560
        #     ## (N-2, 0), (N-2, 1), (N-2, 2)
        #     self.Dif_x[self.N_x - 2, 0] = -1/5; self.Dif_x[self.N_x - 2, 1] = 8/315; self.Dif_x[self.N_x - 2, 2] = -1/560
        #     ## (N-1, 0), (N-1, 1), (N-1, 2), (N-1, 3)
        #     self.Dif_x[self.N_x - 1, 0] = 8/5; self.Dif_x[self.N_x - 1, 1] = -1/5; self.Dif_x[self.N_x - 1, 2] = 8/315; self.Dif_x[self.N_x - 1, 3] = -1/560
            
        #     ## (0, -4), (0, -3), (0, -2), (0, -1)
        #     self.Dif_yy[0, -4] = -1/560; self.Dif_yy[0, -3] = 8/315; self.Dif_yy[0, -2] = -1/5; self.Dif_yy[0, -1] = 8/5
        #     ## (1, -3), (1, -2), (1, -1)           
        #     self.Dif_yy[1, -3] = -1/560; self.Dif_yy[1, -2] = 8/315; self.Dif_yy[1, -1] = -1/5
        #     ## (2, -2), (2, -1)
        #     self.Dif_yy[2, -2] = -1/560; self.Dif_yy[2, -1] = 8/315
        #     ## (3, -1)
        #     self.Dif_yy[3, -1] = -1/560
        #     ## (N-4, 0)
        #     self.Dif_yy[self.N_y - 4, 0] = -1/560
        #     ## (N-3, 0), (N-3, 1)
        #     self.Dif_yy[self.N_y - 3, 0] = 8/315; self.Dif_yy[self.N_y - 3, 1] = -1/560
        #     ## (N-2, 0), (N-2, 1), (N-2, 2)
        #     self.Dif_yy[self.N_y - 2, 0] = -1/5; self.Dif_yy[self.N_y - 2, 1] = 8/315; self.Dif_yy[self.N_y - 2, 2] = -1/560
        #     ## (N-1, 0), (N-1, 1), (N-1, 2), (N-1, 3)
        #     self.Dif_yy[self.N_y - 1, 0] = 8/5; self.Dif_yy[self.N_y - 1, 1] = -1/5; self.Dif_yy[self.N_y - 1, 2] = 8/315; self.Dif_yy[self.N_y - 1, 3] = -1/560
            
        #     ## Off-diagonal terms
        #     ## (0, -4), (0, -3), (0, -2), (0, -1)
        #     self.Dif_x[0, -4] = 1/280; self.Dif_x[0, -3] = -4/105; self.Dif_x[0, -2] = 1/5; self.Dif_x[0, -1] = -4/5
        #     ## (1, -3), (1, -2), (1, -1)           
        #     self.Dif_x[1, -3] = 1/280; self.Dif_x[1, -2] = -4/105; self.Dif_x[1, -1] = 1/5
        #     ## (2, -2), (2, -1)
        #     self.Dif_x[2, -2] = 1/280; self.Dif_x[2, -1] = -4/105
        #     ## (3, -1)
        #     self.Dif_x[3, -1] = 1/280
        #     ## (N-4, 0)
        #     self.Dif_x[self.N_x - 4, 0] = -1/280
        #     ## (N-3, 0), (N-3, 1)
        #     self.Dif_x[self.N_x - 3, 0] = 4/105; self.Dif_x[self.N_x - 3, 1] = -1/280
        #     ## (N-2, 0), (N-2, 1), (N-2, 2)
        #     self.Dif_x[self.N_x - 2, 0] = -1/5; self.Dif_x[self.N_x - 2, 1] = 4/105; self.Dif_x[self.N_x - 2, 2] = -1/280
        #     ## (N-1, 0), (N-1, 1), (N-1, 2), (N-1, 3)
        #     self.Dif_x[self.N_x - 1, 0] = 4/5; self.Dif_x[self.N_x - 1, 1] = -1/5; self.Dif_x[self.N_x - 1, 2] = 4/105; self.Dif_x[self.N_x - 1, 3] = -1/280
            
        #     ## (0, -4), (0, -3), (0, -2), (0, -1)
        #     self.Dif_y[0, -4] = 1/280; self.Dif_y[0, -3] = -4/105; self.Dif_y[0, -2] = 1/5; self.Dif_y[0, -1] = -4/5
        #     ## (1, -3), (1, -2), (1, -1)           
        #     self.Dif_y[1, -3] = 1/280; self.Dif_y[1, -2] = -4/105; self.Dif_y[1, -1] = 1/5
        #     ## (2, -2), (2, -1)
        #     self.Dif_y[2, -2] = 1/280; self.Dif_y[2, -1] = -4/105
        #     ## (3, -1)
        #     self.Dif_y[3, -1] = 1/280
        #     ## (N-4, 0)
        #     self.Dif_y[self.N_y - 4, 0] = -1/280
        #     ## (N-3, 0), (N-3, 1)
        #     self.Dif_y[self.N_y - 3, 0] = 4/105; self.Dif_y[self.N_y - 3, 1] = -1/280
        #     ## (N-2, 0), (N-2, 1), (N-2, 2)
        #     self.Dif_y[self.N_y - 2, 0] = -1/5; self.Dif_y[self.N_y - 2, 1] = 4/105; self.Dif_y[self.N_y - 2, 2] = -1/280
        #     ## (N-1, 0), (N-1, 1), (N-1, 2), (N-1, 3)
        #     self.Dif_y[self.N_y - 1, 0] = 4/5; self.Dif_y[self.N_y - 1, 1] = -1/5; self.Dif_y[self.N_y - 1, 2] = 4/105; self.Dif_y[self.N_y - 1, 3] = -1/280
            
        #     ### Merge X and Y to get a single matrix
        #     self.A_dif = kron(identity(self.N_y), self.Dif_xx) * self.D_coeff[0, 0]/self.dx**2 \
        #                + kron(self.Dif_yy, identity(self.N_x)) * self.D_coeff[1, 1]/self.dy**2 \
        #                + kron(self.Dif_y, self.Dif_x) * (self.D_coeff[0, 1]+self.D_coeff[1, 0])/(self.dx*self.dy)
            
        ### ------------------------------------------------- ###
                
        else:
            print("Incorrect spatial order! Please choose either '2' or '4' or '6' or '8'.")
            
### ============================================================================ ###
