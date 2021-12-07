"""
Created on Wed Dec  1 15:37:38 2021

@author: Pranab JD

Decription:
"""


import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

from initial_conditions import initial_distribution
from integrate import *

### ============================================================== ###

### Parameters
N_x = 500 
N_y = 500
spatial_order = 4
tmax = 1
error_tol = 1e-5



run = initial_distribution(N_x, N_y, spatial_order, tmax, error_tol)
init_value = run.initial_u_2()

plt.imshow(init_value, origin = 'lower', cmap = cm.plasma, extent = [0, 1, 0, 1], aspect = 'equal')
plt.colorbar()
plt.show()
