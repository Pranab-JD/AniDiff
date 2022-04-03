"""
Created on Wed Dec  3 15:46:29 2021

@author: Pranab JD

Description: Consists of different initial conditions
"""

import numpy as np

from domain import Computational_Domain_2D

### ============================================================================ ###

class initial_distribution(Computational_Domain_2D):
    
    def __init__(self, N_x, N_y, spatial_order, tmax, N_cfl, error_tol):
        super().__init__(N_x, N_y, spatial_order, tmax, N_cfl, error_tol)
        
    def initial_u(self):
        
        ### Ring
        
        ### ==================================================== ###
        
        ### Crouseilles et al. 2015
        u_init = 0.1 + 10*np.exp(-((self.X + 0.6)**2 + self.Y**2 )/(0.002))
        
        ### ==================================================== ###
        
        ### Sharma & Hammett 2011
        # u_init = np.zeros((self.N_x, self.N_y))
        # theta = np.zeros((self.N_x, self.N_y))
        
        # for ii in range(self.N_x):
        #     for jj in range(self.N_y):
        #         theta[ii, jj] = np.math.atan(self.Y[ii, jj]/self.X[ii, jj])    
                    
        # for ii in range(self.N_x):
        #     for jj in range(self.N_y):
        #         if ((0.5 < self.radius[ii, jj] < 0.7) and (-0.05026187426480691 < theta[ii, jj] < 0.05026187426480691)):
        #             u_init[ii, jj] = 12
        #         else:
        #             u_init[ii, jj] = 0.1
        
        ### ==================================================== ###

        return u_init
        
### ============================================================================ ###