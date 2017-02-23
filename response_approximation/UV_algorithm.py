"""
resolve.py

Written by Henrik Junklewitz based on scipts by Maksim Greiner

Resolve.py defines the main function that runs RESOLVE on a measurement
set with radio interferometric data.

Copyright 2016 Henrik Junklewitz

RESOLVE is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RESOLVE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with RESOLVE. If not, see <http://www.gnu.org/licenses/>.
"""

from nifty import *
from UV_classes import UV_quantities, lognormal_Hamiltonian
import time as ttt

def initialize_fastresolve(s_space, k_space, R, d, var):
    
    return UV_quantities(domain=s_space, codomain=k_space, u=R.u,\
        v=R.v, d=d, varis=var, A=R.A) 

def resolve_to_fastresolve(fastresolve_operators):
    """
    """

    return fastresolve_operators.get_dd(), fastresolve_operators.get_NN(),\
        fastresolve_operators.get_RR()
        
        
def fastresolve_to_resolve(dorg, Rorg, Norg):
    """
    """
    
    return dorg, Norg, Rorg
