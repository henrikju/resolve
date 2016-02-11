from nifty import *
from grid_cython import grid_complex_cython, grid_abs_squared_cython

def grid_function(codomain=None, inpoints=None, u=None, v=None, precision=30, abs_squared=False):
    
    if(codomain is None):
        raise TypeError("Error: insufficient input.")    
    elif(not isinstance(codomain,rg_space)):
        raise TypeError("Error: invalid input.")
    elif(codomain.naxes() != 2):
        raise ValueError("Error: only 2D spaces supported.")
    
    if(u is None):
        u = np.empty((0,), dtype=np.float64)
    else:
        u = np.array(u, dtype=np.float64)
    if(len(u.shape) != 1):
        raise ValueError("Error: u needs to be a one-dimensional array.")

    if(v is None):
        v = np.empty((0,), dtype=np.float64)
    else:
        v = np.array(v, dtype=np.float64)
    if(v.shape != u.shape):
        raise ValueError("Error: u and v incompatible.")

    if(abs_squared):
        inpoint_dtype = np.float64
    else:
        inpoint_dtype = np.complex128

    if(inpoints is None):
        inpoints = np.ones(u.shape, dtype=inpoint_dtype)
    else:
        inpoints = np.array(inpoints, dtype=inpoint_dtype)
    if(inpoints.shape != u.shape):
        raise ValueError("Error: inpoints and u array not compatible.")

    Nk, Nq = codomain.para[:2]

    vk, vq = codomain.vol

    precision = int(precision)

    if(abs_squared):
        grid = grid_abs_squared_cython(Nk, Nq, vk, vq, inpoints, u, v, precision)
    else:
        grid = grid_complex_cython(Nk, Nq, vk, vq, inpoints, u, v, precision)

    return grid
