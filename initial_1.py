"""
Created on Wed Dec  3 15:46:29 2021

@author: Pranab JD

Description: Consists of different initial conditions
"""

import numpy as np

from domain import Computational_Domain_2D

### ============================================================================ ###

class initial_distribution(Computational_Domain_2D):
    
    def __init__(self, N_x, N_y, spatial_order, tmax, error_tol):
        super().__init__(N_x, N_y, spatial_order, tmax, error_tol)
        
    def initialize_parameters(self):
        super().initialize_parameters()
        self.D_xx = 1.0
        self.D_xy = 0.5
        self.D_yx = 0.5
        self.D_yy = 0.25
        
    def initial_u(self):
        
        ## Diffusion on a ring
        u_init = 0.1 + 10 * np.exp(-((self.X - 0.6)**2 + self.Y**2)/0.02)
        
        return u_init
    
### ============================================================================ ###