import numpy as np
from Divided_Difference import Divided_Difference

def real_Leja_phi(u, dt, RHS_function, interp_function, c, Gamma, Leja_X, phi_function, tol):
    """
    Parameters
    ----------
    u                       : 1D vector u (input)
    dt                      : Step size
    RHS_function            : RHS function
    interp_function         : Function to be interpolated
    c                       : Shifting factor
    Gamma                   : Scaling factor
    Leja_X                  : Array of Leja points
    phi_function            : phi function
    tol                     : Accuracy of the polynomial so formed

    Returns
    ----------
    polynomial_array        : Polynomial interpolation of 'interp_function' 
                              multiplied by 'phi_function' at real Leja points
    2*ii                    : # of RHS calls

    """

    ### Initialize paramters and arrays
    convergence = 0                                                             # 0 -> did not converge, 1 -> converged
    max_Leja_pts = len(Leja_X)                                                  # Max number of Leja points  
    y = interp_function.copy()                                                  # To avoid changing 'interp_function'
        
    ### Phi function applied to 'interp_function' (scaled and shifted); scaling down of c and Gamma (i.e. largest and smallest eigenvalue) by dt
    phi_function_array = phi_function((c + Gamma*Leja_X)*dt)
    
    ### Compute polynomial coefficients
    poly_coeffs = Divided_Difference(Leja_X, phi_function_array) 
    
    ### p_0 term
    polynomial = interp_function * poly_coeffs[0]
    
    ### p_1, p_2, ...., p_n terms; iterate until converges
    for ii in range(1, max_Leja_pts):

        ### y = y * ((z - c)/Gamma - Leja_X)
        y = (RHS_function(y)/Gamma) + (y * (-c/Gamma - Leja_X[ii - 1]))

        ### Error estimate
        poly_error = np.linalg.norm(y) * abs(poly_coeffs[ii])
        
        ### To prevent diverging, restart simulations with smaller dt
        if poly_error > 1e17:
            convergence = 0
            print("Did not converge")
            return u, ii+1, convergence

        ### Add the new term to the polynomial
        polynomial = polynomial + (poly_coeffs[ii] * y)
        
        ### If new term to be added < tol, break loop; safety factor = 0.1
        if  poly_error < 0.1*tol:
            convergence = 1
            # print("# Leja points (phi): ", ii)
            break
        
        if ii == max_Leja_pts - 1:
            print("Warning!! Max. # of Leja points reached!!")
            break

    return polynomial, ii, convergence