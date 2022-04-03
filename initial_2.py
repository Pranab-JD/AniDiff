"""
Created on Wed Dec  3 15:46:29 2021

@author: Pranab JD

Description: Consists of different initial conditions
"""

import numpy as np

from domain import Computational_Domain_2D

### ============================================================================ ###

class initial_distribution(Computational_Domain_2D):
    
    def initial_u(self):
        
        ## Constant diffusion on a periodic band (Crouseilles et al. 2015)
        radius = (self.X**2 + self.Y**2)**0.5
        u_init = np.zeros((self.N_x, self.N_y))
        
        for ii in range(self.N_x):
            for jj in range(self.N_y):
                if radius[ii, jj] < 2*np.pi/5:    
                    u_init[ii, jj] = 1.0 + (3*np.exp(-2*radius[ii, jj]**2))
                else:
                    u_init[ii, jj] = 1.0
        
        return u_init
    
### ============================================================================ ###