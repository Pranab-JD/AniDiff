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
        
        ## Analytic test case (Crouseilles et al. 2015)
        u_init = np.sin(np.pi * self.X) * np.sin(np.pi * self.Y)
        
        return u_init
        
### ============================================================================ ###