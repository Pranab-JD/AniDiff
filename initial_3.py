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
        
    def initialize_parameters(self):
        super().initialize_parameters()
        self.D_xx = 1.0
        self.D_xy = 1.0
        self.D_yx = 1.0
        self.D_yy = 1.0
    
    def initial_u(self):
        
        ## Analytic test case
        u_init = np.sin(np.pi * self.X) * np.sin(np.pi * self.Y)
        
        return u_init
        
### ============================================================================ ###