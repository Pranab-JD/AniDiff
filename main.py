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
spatial_order = 2
tmax_list = [0.1, 0.3, 15.0]
N_list = [2**6, 2**7, 2**8, 2**9]
# cfl_list = [1000, 10000, 25000, 50000]
cfl_list = [10, 50, 100]

tol_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]

for tmax in tmax_list:
    for N in N_list:
        for n_cfl in cfl_list:
            for tol in tol_list:

                ### Object initialization
                run = Integrate(N, N, spatial_order, tmax, n_cfl, tol)

                def main():
                    run.run_code(tmax)
                    
                if __name__ == "__main__":
                    main()


### ============================================================================ ###

print()
print("==============================================")
print()
print('Total Time Elapsed = ', datetime.now() - startTime)
print()