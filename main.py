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
tmax_list = [0.50]
N_list = [2**8]
# N_list = [2**8]

cfl_list = [20]
# cfl_list = [10]

# tol_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
# tol_list = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8]

# tol_list = [1e-12, 1e-2]
# tol_list = [1e-9, 1e-8, 1e-7]
# tol_list = [1e-10, 1e-4, 1e-6]
tol_list = [1e-10]

for N in N_list:
    for tmax in tmax_list:
        for n_cfl in cfl_list:
            for tol in tol_list:

                ### Object initialization
                run = Integrate(N, N, tmax, n_cfl, tol)

                def main():
                    run.run_code(tmax)
                    
                if __name__ == "__main__":
                    main()


### Test
# run = Integrate(2**8, 2**8, 2, 5.75, 1000000, 1e-12)

# def main():
#     run.run_code(5.75)
    
# if __name__ == "__main__":
#     main()


### ============================================================================ ###

print()
print("==============================================")
print()
print('Total Time Elapsed = ', datetime.now() - startTime)
print()