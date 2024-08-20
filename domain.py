"""
Created on Wed Dec 1 15:37:38 2021

@author: Pranab JD

Description: Create required matrices and initialize parameters
"""

import numpy as np
from scipy.sparse import lil_matrix, kron, identity, diags

### ============================================================================ ###

class Computational_Domain_2D:

    def __init__(self, N_x, N_y, tmax, N_cfl, error_tol, case):
        self.N_x = int(N_x)                         # Number of points along X
        self.N_y = int(N_y)                         # Number of points along Y
        self.xmin = -1                              # Min value of X
        self.xmax =  1                              # Max value of X
        self.ymin = -1                              # Min value of Y
        self.ymax =  1                              # Max value of Y
        
        self.tmax = tmax                            # Simulation time
        self.N_cfl = N_cfl                          # "N" times CFL limit
        self.error_tol = error_tol                  # Maximum error permitted

        self.case = case                            # Problem: B field configuration
        
        self.initialize_spatial_domain()
        self.initialize_parameters()
        self.initialize_matrices()

        
    def initialize_spatial_domain(self):

		###? Periodic boundary conditions
        self.X = np.linspace(self.xmin, self.xmax, self.N_x, endpoint=False)
        self.Y = np.linspace(self.ymin, self.ymax, self.N_y, endpoint=False)

        self.dx = self.X[2] - self.X[1]
        self.dy = self.Y[2] - self.Y[1]

        self.X, self.Y = np.meshgrid(self.X, self.Y)


    def initialize_parameters(self):

        ###? ------------------------------------- ?###

        ###! Anisotropic Diffusion Coefficients
        
        if self.case == "2D Band":
            
            self.D_x = 1.0
            self.D_y = 0.5

        elif self.case == "1D Band":
            
            self.D_x = 1.0
            self.D_y = 0.0

        elif self.case == "Ring":
            
            self.D_x =  self.Y
            self.D_y = -self.X

        elif self.case == "Spiral 1":

            self.D_x =  self.X + 4*self.Y
            self.D_y = -self.X

        elif self.case == "Spiral 2":

            self.D_x =  self.X + self.Y
            self.D_y = -self.X

        else:
            print("Incorrect case!")

        ###? ------------------------------------- ?###
        
        ###? Advection velocity
        self.velocity_x = 0.1
        self.velocity_y = 0.2
        
        ###? CFL constraints
        self.dif_cfl = (self.dx**2 * self.dy**2)/(2*(self.dx**2 + self.dy**2))
        self.adv_cfl = (self.dx * self.dy)/(abs(self.velocity_y)*self.dx + abs(self.velocity_x)*self.dy)
        self.dt = self.N_cfl * min(self.dif_cfl, self.adv_cfl)
        
        print()
        print("==============================================")
        print("Nx x Ny          :", self.N_x, "x", self.N_y)
        print("Dif CFL time     : {:e}".format(self.dif_cfl))
        print("Adv CFL time     : {:e}".format(self.adv_cfl))
        print("N times CFL      :", self.N_cfl)
        print("Tolerance        :", self.error_tol)
        print("Simulation time  :", self.tmax)
        print()
        

    def initialize_matrices(self):

        ###? 2nd order centered difference for diffusion (1, -2, 1) & (-1/2, 0, 1/2)
        self.Dif_xx = lil_matrix(diags(np.ones(self.N_x - 1), -1) + diags(-2*np.ones(self.N_x), 0) + diags(np.ones(self.N_x - 1), 1))
        self.Dif_yy = lil_matrix(diags(np.ones(self.N_y - 1), -1) + diags(-2*np.ones(self.N_y), 0) + diags(np.ones(self.N_y - 1), 1))
        
        self.Dif_x  = lil_matrix(diags(-1/2*np.ones(self.N_x - 1), 1) + diags(1/2*np.ones(self.N_x - 1), -1))
        self.Dif_y  = lil_matrix(diags(-1/2*np.ones(self.N_y - 1), 1) + diags(1/2*np.ones(self.N_y - 1), -1))

        if self.case == "1D Band" or "2D Band":

            ###? Periodic Boundary conditions - Diagonal terms
            self.Dif_xx[0, -1] = 1; self.Dif_xx[-1, 0] = 1      # (0, -1), (N-1, 0)
            self.Dif_yy[0, -1] = 1; self.Dif_yy[-1, 0] = 1      # (0, -1), (N-1, 0)
            
            ###? Off-diagonal terms
            self.Dif_x[0, -1] = 1/2; self.Dif_x[-1, 0] = -1/2       # (0, -1), (N-1, 0)
            self.Dif_y[0, -1] = 1/2; self.Dif_y[-1, 0] = -1/2       # (0, -1), (N-1, 0)

            ###! 2D Anisotropic Diffusion Matrix
            self.A_dif = kron(identity(self.N_y), self.D_x*self.Dif_xx/self.dx**2)   \
                       + kron(self.D_y*self.Dif_yy/self.dy**2, identity(self.N_x))   \
                       + kron(self.Dif_x*self.D_x, self.Dif_y*self.D_x)/(self.dx*self.dy)     \
                       + kron(self.Dif_y*self.D_y, self.Dif_x*self.D_x)/(self.dx*self.dy)

        elif self.case == "Ring" or "Spiral 1" or "Spiral 2":

            ###! 2D Anisotropic Diffusion Matrix
            self.A_dif = kron(identity(self.N_y).multiply((self.D_x**2).diagonal()), self.Dif_xx/self.dx**2)    \
                       + kron(self.Dif_yy/self.dy**2, identity(self.N_x).multiply((self.D_y**2).diagonal()))    \
                       + kron(self.Dif_x.multiply(self.D_x), self.Dif_y.multiply(self.D_y))/(self.dx*self.dy)   \
                       + kron(self.Dif_y.multiply(self.D_y), self.Dif_x.multiply(self.D_x))/(self.dx*self.dy)
            
        else:
            print("Incorrect case!")


        ###? 3rd order upwind for advection (-2/6, -3/6, 1, -1/6)
        self.Adv_x = lil_matrix(diags(-2/6*np.ones(self.N_x - 1), -1) + diags(-3/6*np.ones(self.N_x), 0) + diags(np.ones(self.N_x - 1), 1) + diags(-1/6*np.ones(self.N_x - 2), 2))
        self.Adv_y = lil_matrix(diags(-2/6*np.ones(self.N_y - 1), -1) + diags(-3/6*np.ones(self.N_y), 0) + diags(np.ones(self.N_y - 1), 1) + diags(-1/6*np.ones(self.N_y - 2), 2))
        
        ###? Boundary conditions (Periodic)
        self.Adv_x[-2, 0] = -1/6; self.Adv_x[-1, 0] = 1; self.Adv_x[-1, 1] = -1/6; self.Adv_x[0, -1] = -2/6
        self.Adv_y[-2, 0] = -1/6; self.Adv_y[-1, 0] = 1; self.Adv_y[-1, 1] = -1/6; self.Adv_y[0, -1] = -2/6

        ###! 2D Advection Matrix
        self.A_adv = kron(identity(self.N_y), self.Adv_x*self.velocity_x/self.dx) + kron(self.Adv_y*self.velocity_y/self.dy, identity(self.N_x))
                    
        
        ###? Laplacian matrix for the penalisation approach
        self.Laplacian = kron(identity(self.N_y), self.Dif_xx/self.dx**2) + kron(self.Dif_yy/self.dy**2, identity(self.N_x))