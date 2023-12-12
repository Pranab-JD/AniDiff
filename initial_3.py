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
        
        u_init = np.zeros((self.N_x, self.N_y))
        
        for ii in range(self.N_x):
            for jj in range(self.N_y):
                if self.X[ii, jj] > 0 and self.Y[ii, jj] > 0:    
                    u_init[ii, jj] = 1000.0
                else:
                    u_init[ii, jj] = 0.0
                    
        return u_init
        
### ============================================================================ ###