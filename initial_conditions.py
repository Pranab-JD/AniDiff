"""
Created on Wed Dec  3 15:46:29 2021

@author: Pranab JD

Decription: Consists of different initial conditions
"""


import numpy as np

from domain import Computational_Domain_2D

class initial_distribution(Computational_Domain_2D):

    def initial_u_1(self):
        
        x0 = 0.9; y0 = 0.9
        sigma = 0.02
        
        np.seterr(divide = 'ignore')
        u_init_1 =  1 + (np.exp(1 - (1/(1 - (2 * self.X - 1)**2)) - (1/(1 - (2 * self.Y - 1)**2)))) \
                      + 1./2. * np.exp(-((self.X - x0)**2 + (self.Y - y0)**2)/(2 * sigma**2))
        
        return u_init_1
    
    def initial_u_2(self):
        
        x1 = 0.25; x2 = 0.6
        y1 = 0.25; y2 = 0.6
        
        u_init_2 = 1 + np.heaviside(x1 - self.X, x1) + np.heaviside(self.X - x2, x2) \
                     + np.heaviside(y1 - self.Y, y1) + np.heaviside(self.Y - y2, y2)
               
        return u_init_2