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
        self.xmin = 0                              # Min value of X
        self.ymin = 0                              # Min value of Y
        
    def initial_u(self):
                    
        eps = 0.05
        
        ## Gaussian pulse (Isotropic)
        u_init = (2*np.pi)**(-3/2)/(eps**2 + (2*self.D_xx))**(3/2) * np.exp(-(1/2) * ((self.X**2 + self.Y**2)/(eps**2 + (2*self.D_xx))))
        
        return u_init
        
### ============================================================================ ###