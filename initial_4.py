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
                    
        eps = 0.05
        
        ### Gaussian pulse (Hopkins 2017)
        u_init = (2*np.pi)**(-3/2)/eps**3 * np.exp(-0.5 * ((self.X**2 + self.Y**2)/eps**2))
        
        return u_init
        
### ============================================================================ ###