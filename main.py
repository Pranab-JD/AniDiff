"""
Created on Wed Dec  1 15:37:38 2021

@author: Pranab JD

Description:
"""

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

# from integrate_adaptive import *
from integrate_constant import *

from datetime import datetime

startTime = datetime.now()

### ============================================================================ ###

### Parameters
N_x = 75
N_y = 75
spatial_order = 2
tmax = 5.0
error_tol = 1e-5

### Object initialization
run = Integrate(N_x, N_y, spatial_order, tmax, error_tol)

def main():
    run.run_code(tmax)
    
if __name__ == "__main__":
    main()
    
### ============================================================================ ###

print('Total Time Elapsed = ', datetime.now() - startTime)