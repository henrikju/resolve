"""
operators.py
Written by Henrik Junklewitz

operators.py is an auxiliary file for resolve.py and belongs to the RESOLVE
package. It provides all the needed operators for the inference code except
the response operator defined in response.py.

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
from nifty import nifty_tools as nt
from utility_functions import exp,log
from scipy.optimize import minimize as scipyminimize

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
 
#-------------------------single band operators--------------------------------

#class N_operator(operator):
#    """
#    Wrapper around a standard Noise operator. Handles radio astronomical flags.
#    """
#    
#    def _multiply(self, x):
#        
#        variance = self.para[0]
#        
#        mask = variance>0
#        variance[variance==0] = 1.
#
#        Ntemp = diagonal_operator(domain=self.domain, diag = variance)
#
#        return mask * Ntemp(x)
#
#    def _inverse_multiply(self, x):
#        
#        variance = self.para[0]
#        
#        mask = variance>0
#        variance[variance==0] = 1.
#
#        Ntemp = diagonal_operator(domain=self.domain, diag = variance)
#
#        return mask * Ntemp.inverse_times(x)

        
class M_operator(operator):
    """
    """

    def _multiply(self, x):
        """
        """
        N = self.para[0]
        R = self.para[1]

        return R.adjoint_times(N.inverse_times(R.times(x)))


class D_operator(operator):
    """
    The inverse of the D operator is effectively equivalent with the second
    derivitave of the lognormal_Hamiltonian. Since it is only used together
    with this operator, this is not made explicitly clear and, thus,the
    D_operator is not a direct part of the lognormal Hamiltonian.
    """

    def _inverse_multiply(self, x):
        """
        """

        S = self.para[0]
        M = self.para[1]
        m = self.para[2]
        j = self.para[3]
        M0 = self.para[4]
        rho0 = self.para[5]

        nondiagpart = M_part_operator(M.domain, imp=True, para=[M, m, rho0])

        diagpartval = (-1. * j * rho0 * exp(m) + rho0 * exp(m) * M(rho0 * exp(m))).hat()
        
        diag = diagonal_operator(domain = S.domain, diag = 1. * M0)

        part1 = S.inverse_times(x)
        part2 = diagpartval(x)
        part3 = nondiagpart(x)
        part4 = diag(x)

        return part1 + part2 + part3 + part4 

    _matvec = (lambda self, x: self.inverse_times(x).val.flatten())
    

    def _multiply(self, x):
        """
        the operator is defined by its inverse, so multiplication has to be
        done by inverting the inverse numerically using the conjugate gradient
        method.
        """
        convergence = 0
        params = self.para[6]

        if params.pspec_algo == 'cg':
            x_,convergence = nt.conjugate_gradient(self._matvec, x, \
                note=True)(tol=params.pspec_tol,clevel=params.pspec_clevel,\
                limii=params.pspec_iter)
                
        elif params.pspec_algo == 'sd':
            x_,convergence = nt.steepest_descent(self._matvec, x, \
                note=True)(tol=params.pspec_tol,clevel=params.pspec_clevel,\
                limii=params.pspec_iter)
                    
        return x_
        


class M_part_operator(operator):
    """
    """

    def _multiply(self, x):
        """
        """
        M = self.para[0]
        m = self.para[1]
        rho0 = self.para[2]

        return 1. * rho0 * exp(m) * M(rho0 * exp(m) * x)



class lognormal_Hamiltonian_m(object):
    
    def __init__(self, args):
        self.j = args[0]
        self.S = args[1]
        self.M = args[2]
        self.rho0 = args[3]
        self.d = args[4]
        
    def H(self, x):
        """
        """
        # For H it is much faster to calculate the full Hamiltonian in a
        # compact way like this. The derivitaves need to be fully worked out

        N = self.M.para[0]
        R = self.M.para[1]
        n = self.d - R(exp(x))
        return 0.5 * x.dot(self.S.inverse_times(x)) +\
            0.5 * n.dot(N.inverse_times(n))


    def gradH(self, x):
        """
        """
        
        expx = field(domain = x.domain, val = self.rho0 * exp(x))
    
        Sx = self.S.inverse_times(x)
        expxMexpx = expx * self.M(expx)      
        full = -self.j * expx + expxMexpx + Sx
    
        return full
    
    def egg_s(self, x):
        
        E = self.H(x)
        g = self.gradH(x)
        
        return E,g
#-----------------------POINT-RESOLVE-operators------------------------------------        

class lognormal_Hamiltonian_u(object):
    
    def __init__(self, args):
        self.j = args[0]
        self.S = args[1]
        self.M = args[2]
        self.A = args[3]
        self.B = args[4]
        self.NU = args[5]
        self.b = self.B-1

    def H(self, u):
        """
        """
        part2 = self.j.dot(self.A * exp(u))
        part3 = (self.A * exp(u)).dot(self.M(self.A * exp(u))) / 2 
        part4 = (u).dot(self.b)+(exp(-u)).dot(self.NU)
        return part3 + part4 - part2 

    def gradH_u(self,u):
        """
        """
    
        temp1 = self.B - 1 - self.NU * exp(-u)
        temp = -self.j * self.A * exp(u) + self.A* exp(u) * \
            self.M(self.A * (exp(u))) + temp1
    
        return temp
        
    def egg_u(self, u):
        
        E = self.H(u)
        gu = self.gradH_u(u)        
        return E,gu        

        
class lognormal_Hamiltonian_mu(object):
    
    def __init__(self, args):
        self.j = args[0]
        self.S = args[1]
        self.M = args[2]
        self.rho0 = args[3]
        self.B = args[4]
        self.NU = args[5]
        self.seff = args[6]
        self.ueff = args[7]
        self.b = self.B-1

    def H(self,x,u):
        """
        """
        I = field(domain = x.domain, val = self.rho0 * (exp(x)+exp(u)))
        
        part1 = x.dot(self.S.inverse_times(x.weight(power = 0)))  / 2
        part2 = self.j.dot(I)
        part3 = I.dot(self.M(I)) / 2
        part4 = -u.dot(self.b)-(exp(-u)).dot(self.NU) 

        return part1 - part2 + part3 - part4
    
    def gradH_s(self, x,u):
        """
        """
        
        I = field(domain = x.domain, val = self.rho0 * (exp(x)+exp(u)))
        expx = field(domain = x.domain, val = self.rho0 * exp(x)) 
        
        temp1 = self.S.inverse_times(x)
        temp = -self.j * expx + expx * \
            self.M(I) + temp1
    
        return temp

    def gradH_u(self, x,u):
        """
        """
        I = field(domain = x.domain, val = self.rho0 * (exp(x)+exp(u)))
        expu = field(domain = x.domain, val = self.rho0 * exp(u))

        temp1 = self.b - self.NU * exp(-u)
        temp = -self.j * expu+ expu * \
            self.M(I) + temp1
    
        return temp
    
    def egg_s(self, x):
        
        E = self.H(x,self.ueff)
        gs = self.gradH_s(x,self.ueff)
        return E,gs

    def egg_u(self, u):
        
        E = self.H(self.seff,u)
        gu = self.gradH_u(self.seff,u)        
        return E,gu
        
class Dmm_operator(operator):
    """
    The Dmm operator is the second derivative of the full extended_point_Hamil-
    tonian only with respect to m. This is NOT the same as the D operator, only
    valid for the case of a pure extended prior.    
    
    The inverse of the D operator is effectively equivalent with the second
    derivitave of the lognormal_Hamiltonian. Since it is only used together
    with this operator, this is not made explicitly clear and, thus,the
    D_operator is not a direct part of the lognormal Hamiltonian.
    
    Nur params uebergeben oder nur [params.eta ,params.M0, params.rho0] uebergeben
    beides zu uebergeben ist unnoetig
    """

    def _inverse_multiply(self, x):
        """
        """

        S = self.para[0]
        M = self.para[1]
        m = self.para[2]
        j = self.para[3]
        M0 = self.para[4]
        rho0 = self.para[5]
        u =self.para[6]
        
        I = field(domain = x.domain, val = rho0 * (exp(m)+exp(u)))

        nondiagpart = M_part_operator(M.domain, imp=True, para=[M, m, rho0])
        
        diagpartval = (-1. * j * rho0 * exp(m) + rho0 * exp(m) * M(I)).hat()  
        
        diag = diagonal_operator(domain = S.domain, diag = 1. * M0)

        part1 = S.inverse_times(x)
        part2 = diagpartval(x)
        part3 = nondiagpart(x)
        part4 = diag(x)

        return part1 + part2 + part3 + part4 

    _matvec = (lambda self, x: self.inverse_times(x).val.flatten())
    

    def _multiply(self, x):
        """
        the operator is defined by its inverse, so multiplication has to be
        done by inverting the inverse numerically using the conjugate gradient
        method.
        """
        convergence = 0
        params = self.para[7]

        if params.pspec_algo == 'cg':
            x_,convergence = nt.conjugate_gradient(self._matvec, x, \
                note=True)(tol=params.pspec_tol,clevel=params.pspec_clevel,\
                limii=params.pspec_iter)
                
        elif params.pspec_algo == 'sd':
            x_,convergence = nt.steepest_descent(self._matvec, x, \
                note=True)(tol=params.pspec_tol,clevel=params.pspec_clevel,\
                limii=params.pspec_iter)
                    
        return x_

class Duu_operator(operator):
    """
    Umbenennen um klarer zu machen das dies die 2fache ableitung
    von H nach u ist?
    
    Nur params uebergeben oder nur [params.eta ,params.M0, params.rho0] uebergeben
    beides zu uebergeben ist unnoetig
    """

    def _inverse_multiply(self, x):
        """
        """

        S = self.para[0]
        M = self.para[1]
        m = self.para[2]
        j = self.para[3]
        M0 = self.para[4]
        rho0 = self.para[5]
        u =self.para[6]
        params = self.para[7]
        
        I = field(domain = x.domain, val = rho0 * (exp(m)+exp(u)))

        nondiagpart = M_part_operator(M.domain, imp=True, para=[M, u, rho0])
        
        diagpartval = (-1. * j * rho0 * exp(u) + params.eta * exp(-u) + exp(u) * M(I)).hat()  

        part1 = diagpartval(x)
        part2 = nondiagpart(x)

        return part1 + part2 

    _matvec = (lambda self, x: self.inverse_times(x).val.flatten())
    

    def _multiply(self, x):
        """
        the operator is defined by its inverse, so multiplication has to be
        done by inverting the inverse numerically using the conjugate gradient
        method.
        """
        convergence = 0
        params = self.para[7]

        if params.pspec_algo == 'cg':
            x_,convergence = nt.conjugate_gradient(self._matvec, x, \
                note=True)(tol=params.pspec_tol,clevel=params.pspec_clevel,\
                limii=params.pspec_iter)
                
        elif params.pspec_algo == 'sd':
            x_,convergence = nt.steepest_descent(self._matvec, x, \
                note=True)(tol=params.pspec_tol,clevel=params.pspec_clevel,\
                limii=params.pspec_iter)
                    
        return x_


#-----------------------multi frequency operators------------------------------------

class MI_operator(operator):
    """
    """

    def _multiply(self, x):
        """
        """
        N = self.para[0]
        R = self.para[1]
        a = self.para[2]

        return R.adjoint_times(N.inverse_times(R.times(x, a = a)), a = a)


class Ma_part_operator(operator):
    """
    """

    def _multiply(self, x):
        """
        """
        R = self.para[0]
        N = self.para[1]
        m_I = self.para[2]
        a = self.para[3]
        rho0 = self.para[4]

        return rho0 * exp(m_I) * R.adjoint_times(N.inverse_times( \
            R(rho0 * exp(m_I), a = a)), a = a, mode = 'D')        

        
class Da_operator(operator):

    def _inverse_multiply(self,x):

        S = self.para[0]
        R = self.para[1]
        N = self.para[2]
        a = self.para[3]
        M0 = self.para[4]
        rho0 = self.para[5]
        m_I = self.para[6]
        d = self.para[7]
        

        nondiagpart = Ma_part_operator(R.domain, para = [R,N,m_I,a,rho0])

        diagpartval = (-1.*R.adjoint_times(N.inverse_times(d), a = a, \
            mode = 'D') * rho0 * exp(m_I) + rho0 * exp(m_I) * \
            (R.adjoint_times(N.inverse_times( \
            R(rho0 * exp(m_I), a = a)), a = a, mode = 'D'))).hat()

        diag = diagonal_operator(domain = S.domain, diag = 1. * M0)

        part1 = S.inverse_times(x)

        part2 = diagpartval(x)

        part3 = nondiagpart(x)

        part4 = diag(x)     
                                                                                                                               

        return part1 + part2 + part3 + part4 
        
    _matvec = (lambda self,x: self.inverse_times(x).val.flatten())
        
    def _multiply(self,x):
        
        convergence = 0
        params = self.para[8]

        if params.pspec_algo == 'cg':
            x_,convergence = nt.conjugate_gradient(self._matvec, x, \
                note=True)(tol=params.pspec_tol_a,clevel=params.pspec_clevel_a,\
                limii=params.pspec_iter_a)
                
        elif params.pspec_algo == 'sd':
            x_,convergence = nt.steepest_descent(self._matvec, x, \
                note=True)(tol=params.pspec_tol_a,clevel=params.pspec_clevel_a,\
                limii=params.pspec_iter_a)
                    
        return x_
        
class lognormal_Hamiltonian_a(object):

    
    
    def __init__(self, args):
        self.d = args[0]
        self.S = args[1]
        self.N = args[2]
        self.R = args[3]
        self.m_I = args[4]
        
    def Ha(self,a):
        """
        """
        
        expI = exp(self.m_I)

        part1 = a.dot(self.S.inverse_times(a))/2

        part2 = self.R.adjoint_times(self.N.inverse_times(self.d), a = a).dot(\
            expI)

        Mexp = self.R.adjoint_times(self.N.inverse_times(\
            self.R(expI, a = a)), a = a)

        part3 = expI.dot(Mexp)/2.

        return part1 - part2 + part3
    
    def gradHa(self, a):
        """
        """
        
        expI = exp(self.m_I)

        temp = - self.R.adjoint_times(self.N.inverse_times(self.d), a = a, \
            mode = 'grad') * expI + expI * \
            self.R.adjoint_times(self.N.inverse_times(self.R.times( \
            expI, a = a)), a = a, mode = 'grad') + \
            self.S.inverse_times(a)

        return temp
    
    def egg(self, x):
        
        E = self.Ha(x)
        g = self.gradHa(x)
        
        return E,g
        
#------------------------- Minimizer functions --------------------------------


def minimize_Hamiltonian(ham, mapguess, runlist_element, map_algo, call,
                         map_alpha, map_tol, map_clevel, map_iter, params,
                         uguess=None):

    """
    Minimizes a Hamiltonian object for all possible Resolve versions. It either
    uses the steepest descent from nifty or any algorithm available in the
    scipy.optimize module. Choices are: 'TNC','COBYLA','SLSQP','dogleg',\
    'trust-ncg','CG','BFGS','L-BFGS-B','Nelder-Mead','Powell','Newton-CG').
    """

    if runlist_element == 'resolve_map' or runlist_element == \
                          'point_resolve_map_m' or runlist_element ==\
                          'point_resolve_map_u' or runlist_element ==\
                          'mf_resolve_map_alpha' or runlist_element ==\
                          'point_resolve_pure_point':

        if map_algo == 'sd':

            # Check which Hamiltonian to use
            if runlist_element == 'resolve_map' or runlist_element ==\
                                  'point_resolve_map_m':
                fullham_hamgrad = ham.egg_s

            if runlist_element == 'point_resolve_map_u' or runlist_element ==\
                                  'point_resolve_pure_point':
                fullham_hamgrad = ham.egg_u

            minimize = nt.steepest_descent(fullham_hamgrad, spam=
                                           call.callbacknifty, note=True)
            reconstruction = minimize(x0=mapguess, alpha=map_alpha,
                                      tol=map_tol, clevel=map_clevel,
                                      limii=map_iter)[0]

            return reconstruction

        else:

            x0 = mapguess
            reconstruction = scipyminimize(Hamiltonian_to_scipy,
                             (x0).val.flatten(), args=(ham, x0.domain, params,
                             runlist_element), method = map_algo, jac = True,
                             options={"maxiter": map_iter, "eps": map_alpha,
                             "gtol": 1e4*map_tol, "disp": True},
                             callback=call.callbackscipy)

            return field(x0.domain, target=x0.target, val=reconstruction.x)




def Hamiltonian_to_scipy(x0,ham,xdomain,params,runlist_element,\
    mode='single_optimization', mid=0,end=0):

    
    if mode == 'single_optimization': 
        
        # Check which Hamiltonian to use
        if runlist_element == 'resolve_map' or runlist_element ==\
            'point_resolve_map_m':
            fullham_grad = ham.egg_s
            
        if runlist_element == 'point_resolve_map_u' or runlist_element ==\
            'point_resolve_pure_point':
            fullham_grad = ham.egg_u       
        
        x = field(xdomain,val=x0)
        E,g = fullham_grad(x)
        
        return E, g.val.flatten() * xdomain.vol.prod()

        
    elif mode == 'double_optimization':        
             
        sval =np.array(mid)  
        uval =np.array(mid)  
        sval = x0[0:mid]    
        uval = x0[mid:end]
        s = field(xdomain, val = sval)
        u = field(xdomain, val = uval)
  
        gs = ham.gradH_s(s,u)
        gu = ham.gradH_u(s,u)
        E = en.H(s,u)
        
        g=np.ones(end)    
        gsval = gs.val.flatten()* xdomain.vol.prod()
        guval = gu.val.flatten()* xdomain.vol.prod()
        g[0:mid] =gsval
        g[mid:end] =guval

        return E,g
