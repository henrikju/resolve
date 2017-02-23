"""
algorithms.py
Written by Henrik Junklewitz

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
import utility_functions as utils
from operators import *
import response_approximation.UV_algorithm as ra

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#-------------------------Map reconstructions----------------------------------

def resolve_map(m, j, S, M, d, runlist_element, params, logger):
    """
    Standard Resolve extended source map reconstruction.
    """
    
    logger.header1("Run total intensity standard RESOLVE extended"\
        +" sources iteration cycle.")
    
    # Check whether a fastresolve run with explicit crude Hessian approximation
    # and fastresolve-Hamiltonian implementation was requested 
    # (see Greiner et al 2016)
    if not params.crude_Hessian:
        
        # iteration parameters 
        call = utils.callbackclass(params.save,params.map_algo,\
            runlist_element, params.callbackfrequency, params.imsize)
        
        args = (j, S, M, params.rho0, d)
        ham = lognormal_Hamiltonian_m(args)
        
        m = minimize_Hamiltonian(ham, m, runlist_element, params.map_algo,\
        call,params.map_alpha, params.map_tol, params.map_clevel,\
        params.map_iter, params)
        
    else:
        # Fastresolve-Hamiltonian is only working with LBFGS right now
        H = ra.lognormal_Hamiltonian(S=S, N=M.para[0], R=M.para[1], d=d)
        m = H.min_BFGS(m, limii=params.map_iter, note=False, tol=\
            params.map_tol)

    return m

def point_resolve_map_m(m, u, j, S, M, runlist_element, params, logger):
    """
    Point-Resolve extended source map reconstruction.
    """

    logger.header1("Run total intensity Point-RESOLVE extended"\
    +" sources iteration cycle.")
    
    # iteration parameters 
    call = utils.callbackclass(params.save,params.map_algo,\
        runlist_element, params.callbackfrequency, params.imsize)

    args = (j, S, M, params.rho0, params.beta, params.eta,m,u)
    ham = lognormal_Hamiltonian_mu(args)
    m = minimize_Hamiltonian(ham, m, runlist_element, params.map_algo,\
    call,params.map_alpha, params.map_tol, params.map_clevel,\
    params.map_iter, params)

    return m
    
def point_resolve_map_u(m, u, j, S, M, runlist_element, params, logger):
    """
    Point-Resolve point source map reconstruction.
    """
    
    logger.header1("Run total intensity Point-RESOLVE point"\
    +" sources iteration cycle.")
    
    # iteration parameters 
    call = utils.callbackclass(params.save,params.map_algo_u,\
        runlist_element, params.callbackfrequency, params.imsize)

    args = (j, S, M, params.rho0, params.beta, params.eta,m,u)
    ham = lognormal_Hamiltonian_mu(args)
    u = minimize_Hamiltonian(ham, u, runlist_element, params.map_algo_u,\
    call,params.map_alpha_u, params.map_tol_u, params.map_clevel_u,\
    params.map_iter_u, params)

    return u    
    
def point_resolve_pure_point(u, j, S, M, runlist_element, params, logger):
    """
    Point-Resolve pure point source map reconstruction. No extended source 
    model is assumed. A pure point Hamiltonian is assumed.
    """   
    
    logger.header1("Run total intensity Point-RESOLVE pure point"\
        +" sources iteration cycle.")
    
    # iteration parameters 
    call = utils.callbackclass(params.save,params.map_algo_u,\
        runlist_element, params.callbackfrequency, params.imsize)

    args = (j, S, M, params.rho0, params.beta, params.eta)
    ham = lognormal_Hamiltonian_u(args)
    u = minimize_Hamiltonian(ham, u, runlist_element, params.map_algo_u,\
    call,params.map_alpha_u, params.map_tol_u, params.map_clevel_u,\
    params.map_iter_u, params)

    return u

     
def mf_resolve_map_alpha(a, m, j, S, M, runlist_element, params, logger):
    """
    MF-Resolve spectral index map reconstruction.
    """

    logger.header1("Run total intensity MF-RESOLVE spectral"\
        +" index iteration cycle.")
    
    # iteration parameters 
    call = utils.callbackclass(params.save,params.map_algo_a,\
        runlist_element, params.callbackfrequency, params.imsize)

    args = args = (d,S, N, R,m)
    ham = lognormal_Hamiltonian_a(args)
    a = minimize_Hamiltonian(ham, a, runlist_element, params.map_algo,\
    call,params.map_alpha, params.map_tol, params.map_clevel,\
    params.map_iter, params)

    return a    
    
def wienerfilter_map(j, D):
    """
    Wiener Filter extended source map reconstruction. No Point- or MF-Resolve.
    """

    logger.header1("Run total intensity Wienerfilter extended"\
        +" sources iteration cycle.")

    m = D(j)

    return m
    
def maximum_likelihood(m, j, M, D, params, logger):
    """
    Maximum Likelihood extended source map reconstruction.
    """

    logger.header1("Run total intensity Maximum Likelihood extended"\
        +" sources iteration cycle.")

    D.para = [diagonal_operator(m.domain,diag=1.), M, m, j, params.M0_start, params.rho0, params,]

    m = D(j)

    return m


#-------------------------Pspec reconstructions--------------------------------

def resolve_pspec(m, D, S, N, R, d, k_space, alpha_prior, params, logger):
    """
    Power spectrum reconstruction for the extended source field.
    General enough to be used together with all map reconstructions just by
    changing the Response.
    """
    
    logger.header1("Run total standard RESOLVE extended"\
        +" sources power spectrum iteration cycle.")

    # Check whether a fastresolve run with explicit crude Hessian approximation
    # and fastresolve-Hamiltonian implementation was requested 
    # (see Greiner et al 2016)
    if not params.crude_Hessian:

        #extra loop to take care of possible nans in PS calculation
        psloop = True
        M0 = params.M0_start
              
        while psloop:
        
            Sk = projection_operator(domain=k_space)
            if params.ftmode == 'wsclean':
                #bare=True?
                #right now WSclean is not compatible with parallel probing
                #due to its internal parallelization
                logger.message('Calculating Dhat for pspec reconstruction.')
                D_hathat = D.hathat(domain=k_space,loop=True)
                logger.message('Success.')
            else:
               #bare=True?
                logger.message('Calculating Dhat for pspec reconstruction.')
                D_hathat = D.hathat(domain=k_space,\
                    ncpu=params.ncpu,nrun=params.nrun)
                logger.message('Success.')
            
            pspec = infer_power(m,domain=k_space,Sk=Sk,D=D_hathat,\
                q=1E-42,alpha=alpha_prior,perception=(1,0),smoothness=True,\
                    var=params.smoothing, bare=True)
    
            if np.any(pspec == False):
                print 'D not positive definite, try increasing M0.'
                if M0 == 0:
                    M0 += 0.1
                M0 *= 1e6
                D.para = [D.para[0], D.para[1], D.para[2], D.para[3],\
                    M0, D.para[5], D.para[6], D.para[6]]
                logger.message("    Current M0:  " + str(D.para[4])+ '\n.')
            else:
                psloop = False
    
    # Use fastresolve's crude Hessian approximation            
    else:

        Sk = projection_operator(domain=k_space)
        H = ra.lognormal_Hamiltonian(S=S, N=N, R=R, d=d)
        D = H.crude_Hessian(m)
        pspec = infer_power(m, Sk=Sk, D=D, alpha=alpha_prior, q=1e-42,
                            smoothness=True, var=params.smoothing)
    
    return pspec
    
def mf_resolve_pspec_a(a, Da, k_space, alpha_prior, params, logger):
    """
    Power spectrum reconstruction for the spectral index field.
    """

    logger.header1("Run total MF-RESOLVE spectral"\
        +" index power spectrum iteration cycle.")

    #extra loop to take care of possible nans in PS calculation
    psloop = True
    M0 = params.M0_start_a
          
    while psloop:
    
        Sk = projection_operator(domain=k_space)
        if params.ftmode == 'wsclean':
            #bare=True?
            #right now WSclean is not compatible with parallel probing
            #due to its internal parallelization
            logger.message('Calculating Dhat for pspec reconstruction.')
            D_hathat = Da.hathat(domain=k_space,loop=True)
            logger.message('Success.')
        else:
           #bare=True?
            logger.message('Calculating Dhat for pspec reconstruction.')
            D_hathat = Da.hathat(domain=k_space,\
                ncpu=params.ncpu,nrun=params.nrun)
            logger.message('Success.')

        pspec = infer_power(a,domain=k_space,Sk=Sk,D=D_hathat,\
            q=1E-42,alpha=alpha_prior,perception=(1,0),smoothness=True,var=\
            params.smoothing, bare=True)

        if np.any(pspec == False):
            logger.message('D not positive definite, try increasing M0.')
            if M0 == 0:
                M0 += 0.1
            M0 *= 1e6
            Da.para = [Da.para[0], Da.para[1], Da.para[2], Da.para[3],\
                M0, Da.para[5], Da.para[6], Da.para[6]]
            logger.message("    Current M0:  " + str(Da.para[4])+ '\n.')
        else:
            psloop = False
            
    return pspec
    

#-----------------------Uncertainty reconstructions----------------------------

def uncertainty_m(D, params, logger):
    """
    Uncertainty reconstruction for standard Resolve extended source.
    """

    logger.header1('Calculating extended sources uncertainty map.')

    if not params.crude_Hessian:

        if params.ftmode == 'wsclean':
            D_hat = D.hat(domain=D.domain, loop=True)
        else:
            D_hat = D.hat(domain=D.domain,
                          ncpu=params.ncpu, nrun=params.nrun)

    else:
        Sk = projection_operator(domain=D.domain.get_codomain())
        H = ra.lognormal_Hamiltonian(S=S, N=N, R=R, d=d)
        D = H.crude_Hessian(m)

    return D_hat


def uncertainty_a(D, params, logger):
    """
    Uncertainty reconstruction for MF-Resolve spectral index.
    """
    
    logger.header1('Calculating MF-Resolve spectral index'\
        + 'uncertainty map.')
    D_hat = D.hat(domain=s_space,\
        ncpu=params.ncpu,nrun=params.nrun)
    return D_hat
    
#----------------------Others--------------------------------------------------
    
def noise_estimation(mapguess, d, R, params, logger):
    """
    Simple noise estimation using the intermediate residual akin to a reduced
    extended critical filter without the full D.
    """
    
    logger.header1("Run total standard RESOLVE extended"\
        +" sources power spectrum iteration cycle.")

    # Do a "poor-man's" extended critical filter step using residual
    logger.header2("Trying simple noise estimate without any D.")
    est_var = (R(mapguess) - d).val
    est_var = np.abs(est_var)**2
    est_var = params.reg_var * est_var + (1-params.reg_var) * est_var.mean()
    
    return est_var
         
