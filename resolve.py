"""
resolve.py
Written by Henrik Junklewitz

Resolve.py defines the main function that runs RESOLVE on a measurement
set with radio interferometric data.

Copyright 2014 Henrik Junklewitz

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

#import standard python modules
from __future__ import division
from time import time
import argparse

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

print '\nModule loading information:'

#import necessary scientific modules
import matplotlib as mpl
mpl.use('Agg')
import pylab as pl
import numpy as np
from nifty import *
from scipy.ndimage.interpolation import zoom

#import RESOLVE-package modules
import utility_functions as utils
import simulation.resolve_simulation as sim
import response_approximation.UV_algorithm as ra
from operators import *
import response as r
import Messenger as Mess
from algorithms import *
import resolve_parser as pars

#conditionally import wsclean module; might not be available
#due to complicated casacore dependencies
try:
    import pywsclean as w
    wsclean_available = True
except ImportError:
    print 'No wsclean support available'
    wsclean_available = False
    



#a few global constants
q = 1e-15
C = 299792458
asec2rad = 4.813681e-6
gsave = ''
gcallback = 3

# Version number of resolve code
from version import __version__ 

#-----------------------Resolve main function----------------------------------

def resolve(runlist, changeplist, params):

    """
    Performs a RESOLVE-reconstruction.
    """

    # Turn off nifty warnings to avoid "infinite value" warnings
    about.warnings.off()
    
    # Make sure to not interactively plot. Produces a mess.
    pl.ion()
    
    # Set up save directories                                                
    if not os.path.exists('resolve_output_' + str(params.save)+'/general'):
        os.makedirs('resolve_output_' + str(params.save)+'/general')
    if not os.path.exists('resolve_output_' + str(params.save)+\
        '/last_iterations'):
            os.makedirs('resolve_output_' + str(params.save)+\
            '/last_iterations')
    if not os.path.exists('resolve_output_' + str(params.save)+\
        '/m_reconstructions'):
            os.makedirs('resolve_output_' + str(params.save)+\
            '/m_reconstructions')
    if not os.path.exists('resolve_output_' + str(params.save)+\
        '/u_reconstructions') and params.pointresolve:
            os.makedirs('resolve_output_' + str(params.save)+\
            '/u_reconstructions')  
    if not os.path.exists('resolve_output_' + str(params.save)+\
        '/mu_reconstructions') and params.pointresolve:
            os.makedirs('resolve_output_' + str(params.save)+\
            '/mu_reconstructions')                
    if not os.path.exists('resolve_output_' + str(params.save)+\
        '/p_reconstructions'):
            os.makedirs('resolve_output_' + str(params.save)+\
            '/p_reconstructions')
    if not os.path.exists('resolve_output_' + str(params.save)+\
        '/D_reconstructions'):
            os.makedirs('resolve_output_' + str(params.save)+\
            '/D_reconstructions') 

    # Set up message logger            
    logfile = 'resolve_output_'+str(params.save)+'/general/' + params.save + \
        '.log'
    logger = Mess.Messenger(verbosity=params.verbosity, add_timestamp=False, \
        logfile=logfile)
    
    # Ensure to correctly initialize standard, point or MF-Resolve and pspec
    if  any('ms' in string for string in runlist):
        params.multifrequency = True
    else:
        params.multifrequency = False
    if  any('point_resolve' in string for string in runlist):
        params.pointresolve = True
    else:
        params.pointresolve = False
    if  any('pspec' in string for string in runlist):
        params.pspec = True
    else:
        params.pspec = False
    if  any('uncertainty' in string for string in runlist):
        params.uncertainty = True
    else:
        params.uncertainty = False
    if  any('simulating' in string for string in changeplist):
        params.simulating = True
    else:
        params.simulating = False
    if  any('simple_point' in string for string in changeplist):
        Ipoint = 0

    # Basic start messages, parameter log
    logger.header1('\nStarting RESOLVE package.')
    
    # Read in first chunk of paramter changes for output
    params = pars.update_parameterclass(params, changeplist[0])
    logger.message('\n The first set of parameters are:\n')
    for par in vars(params).items():
        logger.message(str(par[0])+' = '+str(par[1]),verb_level=2)

#-- Data setup ----------------------------------------------------------------

    # Simulation setup    
    if params.simulating:
        if params.multifrequency:
            raise NotImplementedError('No multifrequency simulations'\
            + ' implemented')
        # This is needed in case the specific simulation setup is
        # deviating from the default-settings
        params = pars.update_parameterclass(params, changeplist[0])
        d, N, R, di, d_space, s_space, expI, n, powspec_sim  = sim.simulate\
            (params, logger)
        np.save('resolve_output_' + str(params.save)+'/general/data',d.val)
        
    # Real data setup    
    else:
        if params.multifrequency:
            d, N, R, di, d_space, s_space = datasetup_MF(params, logger)
        else:
            d, N, R, di, d_space, s_space = datasetup(params, logger)

    # Check if uvcuts or fastresolve are needed and then save standard settings
    if any('uvcut' in string for string in changeplist) or any('fastresolve' 
           in string for string in changeplist):
        dorgval = d.val
        Norgdiag = N.diag()
        if not params.ftmode == 'wsclean':
            uorg = R.u
            vorg = R.v
            Aorg = R.A


#-- Starting guesses setup ----------------------------------------------------

    # All runs are standard Resolve reconstructions        
    if ((not params.multifrequency) and (not params.pointresolve)):
        m, pspec, params, k_space = starting_guess_setup(params, logger, \
            s_space, d_space, di)
    
    # All runs incorporate Point-Resolve reconstructions but no MF-Resolve
    elif ((not params.multifrequency) and (params.pointresolve)):
        m, pspec, u, params, k_space = starting_guess_setup(params, logger\
            , s_space, d_space, di)

    # All runs incorporate MF-Resolve reconstructions but no Point-Resolve
    elif ((params.multifrequency) and (not params.pointresolve)):
        m, pspec, a, pspec_a, params, k_space, k_space_a =\
            starting_guess_setup(params, logger, s_space, d_space, di)

    # All runs incorporate both Point- and MF-Resolve reconstructions
    elif ((params.multifrequency) and (params.pointresolve)):
        m, pspec, u, a, pspec_a, params, k_space, k_space_a =\
            starting_guess_setup(params, logger, s_space, d_space, di)
        

#-- Operators setup -----------------------------------------------------------

    # Attention: D has the same structure in pointresolve or non-pointresolve
    # mode. There is no need to save separate D and Du operators and thus
    # only D occurs in the code.

    # All runs are without pspec, uncertainty and MF 
    if ((not params.multifrequency) and (not params.pspec)\
        and (not params.uncertainty)):
         j, M, S = operator_setup(d, N, R, pspec, m, logger, k_space, params)

    # All runs are with pspec and/or uncertainty but without MF and without
    # Pointresolve
    elif ((not params.multifrequency) and (params.pspec \
        or params.uncertainty)):
        j, M, D, S, alpha_prior = operator_setup(d, N, R, pspec, m, logger,\
            k_space, params)
        
    # All runs are with pspec and/or uncertainty but without MF and with
    # pointresolve
    elif ((not params.multifrequency) and (params.pspec \
        or params.uncertainty) and (params.pointresolve)):
        j, M, D, S, alpha_prior = operator_setup(d, N, R, pspec, m, logger,\
            k_space, params,u=u)
        
    # All runs are with pspec and/or uncertainty and with MF and without
    # Pointresolve
    elif (params.multifrequency and (params.pspec \
        or params.uncertainty)):
        j, M, D, S, alpha_prior, Da, Sa, alpha_prior_a = operator_setup(d, N,\
            R, pspec, m, logger, k_space, params, a=a, pspec_a=pspec_a,\
            k_space_a=k_space_a)
            
    # All runs are with pspec and/or uncertainty and with MF and with
    # pointresolve
    elif (params.multifrequency and (params.pspec \
        or params.uncertainty) and (params.pointresolve)):
        j, M, D, S, alpha_prior, Da, Sa, alpha_prior_a = operator_setup(d, N,\
            R, pspec, m, logger, k_space, params, u=u, a=a, pspec_a=pspec_a,\
            k_space_a=k_space_a)
            
    # Check whether the fastresolve response is used and than initialize
    # needed operators once
    if any('fastresolve' in string for string in changeplist):
        fr_operators = ra.initialize_fastresolve(s_space,
                                                 k_space, R, d, N.diag())

#-- Start reconstruction runs with different algorithms defined in runfile ----

    if params.stokes != 'I':
        logger.failure('Polarization-RESOLVE not yet implemented.')
        raise NotImplementedError('Polarization-RESOLVE not yet implemented.')

    # Preparation to start fundamental Resolve loop over different algorithms
    # listed in algorithms.py
    global_iteration = 0
    convergence = 0
    plist = [pspec]
    if params.multifrequency:
        p_a_list = [pspec_a]

    # Start basic loop over different algorithm elements of Resolve run
    globalt1 = time()
    for (runlist_element, changep_element) in zip(runlist, changeplist):

        logger.header2('Resolve is at global iteration: ' +
                       str(global_iteration+1))

        global_iteration += 1
        # Insert the specific set of parameters for this runlist_element
        params = pars.update_parameterclass(params, changep_element)

        # Standard Resolve extended sources run
        if runlist_element == 'resolve_map':

            if (params.multifrequency and params.fastresolve):
                raise NotImplementedError("No fastresolve available yet"
                                          + " for multifrequency mode.")

            if params.uvcut:

                d, N, R, d_space = adjust_uvcut(d, N, R, params.uvcut,
                                                params.uvcutval)
                M = M_operator(domain=s_space, sym=True, imp=True, para=[N, R])
                j = R.adjoint_times(N.inverse_times(d))

                if params.fastresolve:
                    # re-initialize fastresolve operators to new uv-space
                    fr_operators = ra.initialize_fastresolve(s_space,
                                                             k_space, R, d,
                                                             N.diag())

            if params.fastresolve:
                # Update effected operators for fastresolve mode
                d, N, R = ra.resolve_to_fastresolve(fr_operators)
                M = M_operator(domain=s_space, sym=True, imp=True, para=[N, R])
                j = R.adjoint_times(N.inverse_times(d))

            mold = m
            t1 = time()
            m = resolve_map(m, j, S, M, d, runlist_element, params, logger)
            t2 = time()
            logger.success("Completed standard RESOLVE extended sources"
                           + " iteration cycle.")
            logger.message("Time to complete: " + str((t2 - t1) / 3600.)
                           + ' hours.')

            # Intermediate saves of iteration

            # Save m
            utils.save_m(m, global_iteration, params)
            pars.write_output_to_fits(np.transpose(exp(m.val)*params.rho0),
                                      params, notifier=str(global_iteration),
                                      mode='I')

            # Save residual
            # The adjointfactor only exitst for the full response
            if not params.fastresolve:
                tempsave = R.adjointfactor
                R.adjointfactor = 1
            utils.save_results(R.adjoint_times((R(exp(m)) - d)).val,
                               'residual, iter #' + str(global_iteration),
                               'resolve_output_' + str(params.save) +
                               '/m_reconstructions/' + params.save +
                               '_residual' + str(global_iteration),
                               rho0=params.rho0)
            if not params.fastresolve:
                R.adjointfactor = tempsave

            # Update D if necessary
            if params.pspec or params.uncertainty:
                D.para[2] = m
                if params.multifrequency:
                    Da.para[6] = m

            # Check convergence progress
            if np.max(np.abs(m - mold)) < params.map_conv:
                logger.message('Image seems converged, increase convergence'
                               + ' measure by one.')
                convergence += 1
            if check_global_convergence(convergence, runlist_element, params,
                                        logger):
                break

            if params.uvcut or params.fastresolve:

                # change back data operator definitions to original
                d, N, R, d_space = readjust_uv(dorgval, Norgdiag, s_space,
                                               uorg, vorg, Aorg)

                M = M_operator(domain=s_space, sym=True, imp=True,
                               para=[N, R])
                j = R.adjoint_times(N.inverse_times(d))

        # Point-Resolve extended sources run        
        elif runlist_element == 'point_resolve_map_m':
            
            if (params.multifrequency and params.fastresolve):
                raise NotImplementedError("No fastresolve available yet"\
                + " for multifrequency mode.")
                
            if params.uvcut:

                d, N, R, d_space = adjust_uvcut(d, N, R, params.uvcut,
                                                params.uvcutval)
                if params.fastresolve:
                    # re-initialize fastresolve operators to new uv-space
                    fr_operators = ra.initialize_fastresolve(s_space,
                                                             k_space, R, d,
                                                             N.diag())

            if params.fastresolve:
                # Update effected operators for fastresolve mode
                d, N, R = ra.resolve_to_fastresolve(fr_operators)
                M = M_operator(domain=s_space, sym=True, imp=True, para=[N, R])
                j = R.adjoint_times(N.inverse_times(d))

            mold = m
            t1 = time()
            m = point_resolve_map_m(m, u, j, S, M, runlist_element, params,\
                logger)
            t2 = time()
            logger.success("Completed Point-RESOLVE extended sources"\
                +" iteration cycle.")
            logger.message("Time to complete: " + str((t2 - t1) / 3600.)\
                + ' hours.')
                
            # Intermediate saves of iteration
                
            # Save m and u
            utils.save_m(m, global_iteration, params)
            pars.write_output_to_fits(np.transpose(exp(m.val)*params.rho0),params,\
                notifier = str(global_iteration), mode='I')
            utils.save_mu(m, u, global_iteration, params)
            pars.write_output_to_fits(np.transpose(exp(m.val+u.val)*params.rho0),\
                params,notifier = str(global_iteration), mode='I')
                
            # Save residual
            if not params.fastresolve: 
                tempsave = R.adjointfactor
                R.adjointfactor = 1
            utils.save_results(R.adjoint_times((R(exp(m)+exp(u)) - d)).val,\
                'residual, iter #' + str(global_iteration), \
                'resolve_output_' + str(params.save) +\
                '/m_reconstructions/' + params.save + '_residual' +\
                str(global_iteration), rho0 = params.rho0)
            if not params.fastresolve:
                R.adjointfactor = tempsave
            
            # Update D if necessary
            if params.pspec or params.uncertainty:
                D.para[2] = m
                if params.multifrequency:
                    Da.para[6] = m
            
            # Check convergence progress
            if np.max(np.abs(m - mold)) < params.map_conv:
                logger.message('Image seems converged, increase convergence'\
                    + ' measure by one.')
                convergence += 1
            if check_global_convergence(convergence, runlist_element, params,\
                logger):
                break

            if params.uvcut or params.fastresolve:

                # change back data operator definitions to original
                d, N, R, d_space = readjust_uvcut(dorgval, Norgdiag, s_space,
                                                  uorg, vorg, Aorg)

                M = M_operator(domain=s_space, sym=True, imp=True,
                               para=[N, R])
                j = R.adjoint_times(N.inverse_times(d))
            
        # Point-Resolve point sources run        
        elif runlist_element == 'point_resolve_map_u':

            if (params.multifrequency and params.fastresolve):
                raise NotImplementedError("No fastresolve available yet"\
                + " for multifrequency mode.")
                
            if params.uvcut:

                d, N, R, d_space = adjust_uvcut(d, N, R, params.uvcut,
                                                params.uvcutval)
                if params.fastresolve:
                    # re-initialize fastresolve operators to new uv-space
                    fr_operators = ra.initialize_fastresolve(s_space,
                                                             k_space, R, d,
                                                             N.diag())

            if params.fastresolve:
                # Update effected operators for fastresolve mode
                d, N, R = ra.resolve_to_fastresolve(fr_operators)
                M = M_operator(domain=s_space, sym=True, imp=True, para=[N, R])
                j = R.adjoint_times(N.inverse_times(d))

            uold = u
            t1 = time()
            u = point_resolve_map_u(m, u, j, S, M, runlist_element, params,\
                logger)
            t2 = time()
            logger.success("Completed Point-RESOLVE point sources"\
                +" iteration cycle.")
            logger.message("Time to complete: " + str((t2 - t1) / 3600.)\
                + ' hours.')
                
            # Intermediate saves of iteration
                
            # Save m and u
            utils.save_u(u, global_iteration, params)
            pars.write_output_to_fits(np.transpose(exp(u.val)*params.rho0),params,\
                notifier = str(global_iteration), mode='I')
            utils.save_mu(m, u, global_iteration, params)
            pars.write_output_to_fits(np.transpose(exp(m.val+u.val)*params.rho0),\
                params,notifier = str(global_iteration), mode='I')
                
            # Save residual
            if not params.fastresolve:
                tempsave = R.adjointfactor
                R.adjointfactor = 1
            utils.save_results(R.adjoint_times((R(exp(m)+exp(u)) - d)).val,\
                'residual, iter #' + str(global_iteration), \
                'resolve_output_' + str(params.save) +\
                '/m_reconstructions/' + params.save + '_residual' +\
                str(global_iteration), rho0 = params.rho0)
            if not params.fastresolve:
                R.adjointfactor = tempsave
            
            # Update D if necessary
            if params.pspec or params.uncertainty:
                D.para[6] = u
                if params.multifrequency:
                    Da.para[6] = m
            
            # Check convergence progress
            if np.max(np.abs(u - uold)) < params.map_conv_u:
                logger.message('Image seems converged, increase convergence'\
                    + ' measure by one.')
                convergence += 1
            if check_global_convergence(convergence, runlist_element, params,\
                logger):
                break

            if params.uvcut or params.fastresolve:

                # change back data operator definitions to original
                d, N, R, d_space = readjust_uvcut(dorgval, Norgdiag, s_space,
                                                  uorg, vorg, Aorg)

                M = M_operator(domain=s_space, sym=True, imp=True,
                               para=[N, R])
                j = R.adjoint_times(N.inverse_times(d))

        # Point-Resolve pure point sources run        
        elif runlist_element == 'point_resolve_pure_point':

            if params.uvcut:

                d, N, R, d_space = adjust_uvcut(d, N, R, params.uvcut,
                                                params.uvcutval)
                if params.fastresolve:
                    # re-initialize fastresolve operators to new uv-space
                    fr_operators = ra.initialize_fastresolve(s_space,
                                                             k_space, R, d,
                                                             N.diag())

            if (params.multifrequency and params.fastresolve):
                raise NotImplementedError("No fastresolve available yet"\
                + " for multifrequency mode.")

            if params.fastresolve:
                # Update effected operators for fastresolve mode
                d, N, R = ra.resolve_to_fastresolve(fr_operators)
                M = M_operator(domain=s_space, sym=True, imp=True, para=[N, R])
                j = R.adjoint_times(N.inverse_times(d))

            uold = u
            t1 = time()
            u = point_resolve_pure_point(u, j, S, M, runlist_element, params,\
                logger)
            t2 = time()
            logger.success("Completed Point-RESOLVE PURE point sources"\
                +" iteration cycle.")
            logger.message("Time to complete: " + str((t2 - t1) / 3600.)\
                + ' hours.')
                
            # Intermediate saves of iteration
                
            # Save u
            utils.save_u(u, global_iteration, params)
            pars.write_output_to_fits(np.transpose(exp(u.val)*params.rho0),params,\
                notifier = str(global_iteration), mode='I')
                
            # Save residual
            if not params.fastresolve:
                tempsave = R.adjointfactor
                R.adjointfactor = 1
            utils.save_results(R.adjoint_times((R(exp(u)) - d)).val,\
                'residual, iter #' + str(global_iteration), \
                'resolve_output_' + str(params.save) +\
                '/m_reconstructions/' + params.save + '_residual' +\
                str(global_iteration), rho0 = params.rho0)
            if not params.fastresolve:
                R.adjointfactor = tempsave
            
            # Update D if necessary
            if params.pspec or params.uncertainty:
                D.para[6] = u
                if params.multifrequency:
                    Da.para[6] = m
            
            # Check convergence progress
            if np.max(np.abs(u - uold)) < params.map_conv_u:
                logger.message('Image seems converged, increase convergence'\
                    + ' measure by one.')
                convergence += 1
            if check_global_convergence(convergence, runlist_element, params,\
                logger):
                break

            if params.uvcut or params.fastresolve:

                # change back data operator definitions to original
                d, N, R, d_space = readjust_uvcut(dorgval, Norgdiag, s_space,
                                                  uorg, vorg, Aorg)

                M = M_operator(domain=s_space, sym=True, imp=True,
                               para=[N, R])
                j = R.adjoint_times(N.inverse_times(d))
                
        # Resolve spectral index run        
        if runlist_element == 'mf_resolve_map_alpha':
            
            mold = m
            t1 = time()
            a = mf_resolve_map_alpha(a, m, j, S, M, runlist_element, params,\
                logger)
            t2 = time()
            logger.success("Completed RESOLVE spectral index"\
                +" iteration cycle.")
            logger.message("Time to complete: " + str((t2 - t1) / 3600.)\
                + ' hours.')
                
            # Intermediate saves of iteration
                
            # Save m
            utils.save_results(a.val,\
                'spectral index, iter #' + str(global_iteration), \
                'resolve_output_' + str(params.save) +\
                '/a_reconstructions/' + params.save + '_a' +\
                str(global_iteration))
            pars.write_output_to_fits(np.transpose(a.val),params,\
                notifier = str(global_iteration), mode='I')
                
            # Update D if necessary
            if params.pspec or params.uncertainty:
                Da.para[3] = a
            
            # Check convergence progress
            if np.max(np.abs(a - aold)) < params.map_conv_a:
                logger.message('Image seems converged, increase convergence'\
                    + ' measure by one.')
                convergence += 1
            if check_global_convergence(convergence, runlist_element, params,\
                logger):
                break
        
        # Resolve crude Wienerfilter run        
        if runlist_element == 'wienerfilter_map':
            
            if params.uvcut:
                
                d, N, R, d_space = adjust_uvcut(d, N, R, params.uvcut,\
                    params.uvcutval)
                if params.fastresolve:
                    # re-initialize fastresolve operators to new uv-space
                     fr_operators = ra.initialize_fastresolve(s_space,
                                                              k_space, R, d, 
                                                              N.diag())
                        
            if params.fastresolve:
                # Update effected operators for fastresolve mode
                d, N, R = ra.resolve_to_fastresolve(fr_operators)
                M = M_operator(domain=s_space, sym=True, imp=True, para=[N, R])
                j = R.adjoint_times(N.inverse_times(d))
            
            mold = m
            t1 = time()
            m = wienerfilter_map(j, S, M, logger)
            t2 = time()
            logger.success("Completed standard RESOLVE extended sources"\
                +" iteration cycle.")
            logger.message("Time to complete: " + str((t2 - t1) / 3600.)\
                + ' hours.')
                
            # Intermediate saves of iteration
                
            # Save m
            utils.save_m(m, global_iteration, params)
            pars.write_output_to_fits(np.transpose(exp(m.val)*params.rho0),params,\
                notifier = str(global_iteration), mode='I')
                
            # Save residual 
            if not params.fastresolve:
                tempsave = R.adjointfactor
                R.adjointfactor = 1
            utils.save_results(R.adjoint_times((R(exp(m)) - d)).val,\
                'residual, iter #' + str(global_iteration), \
                'resolve_output_' + str(params.save) +\
                '/m_reconstructions/' + params.save + '_residual' +\
                str(global_iteration), rho0 = params.rho0)
            if not params.fastresolve:
                R.adjointfactor = tempsave
            
            # Update D if necessary
            if params.pspec or params.uncertainty:
                D.para[2] = m
                if params.multifrequency:
                    Da.para[6] = m
            
            # Check convergence progress
            if np.max(np.abs(m - mold)) < params.map_conv:
                logger.message('Image seems converged, increase convergence'\
                    + ' measure by one.')
                convergence += 1
            if check_global_convergence(convergence, runlist_element, params,\
                logger):
                break
            
            if params.uvcut or params.fastresolve:

                # change back data operator definitions to original
                d, N, R, d_space = readjust_uvcut(dorgval, Norgdiag, s_space,
                                                  uorg, vorg, Aorg)

                M = M_operator(domain=s_space, sym=True, imp=True,
                               para=[N, R])
                j = R.adjoint_times(N.inverse_times(d))
            
        # Resolve crude Maximum-Likelihood run        
        if runlist_element == 'maximum_likelihood':
            
            mold = m
            t1 = time()
            m = maximum_likelihood(m, j, M, D, params, logger)
            t2 = time()
            logger.success("Completed standard RESOLVE extended sources"\
                +" iteration cycle.")
            logger.message("Time to complete: " + str((t2 - t1) / 3600.)\
                + ' hours.')
                
            # Intermediate saves of iteration
                
            # Save m
            utils.save_m(m, global_iteration, params)
            pars.write_output_to_fits(np.transpose(exp(m.val)*params.rho0),params,\
                notifier = str(global_iteration), mode='I')
                
            # Save residual 
            tempsave = R.adjointfactor
            R.adjointfactor = 1
            utils.save_results(R.adjoint_times((R(exp(m)) - d)).val,\
                'residual, iter #' + str(global_iteration), \
                'resolve_output_' + str(params.save) +\
                '/m_reconstructions/' + params.save + '_residual' +\
                str(global_iteration), rho0 = params.rho0)
            R.adjointfactor = tempsave
            
            # Update D if necessary
            if params.pspec or params.uncertainty:
                D.para[2] = m
                if params.multifrequency:
                    Da.para[6] = m
            
            # Check convergence progress
            if np.max(np.abs(m - mold)) < params.map_conv:
                logger.message('Image seems converged, increase convergence'\
                    + ' measure by one.')
                convergence += 1
            if check_global_convergence(convergence, runlist_element, params,\
                logger):
                break
            
        # Resolve power spectrum run for the extended field        
        if runlist_element == 'resolve_pspec':

            if params.uvcut:

                d, N, R, d_space = adjust_uvcut(d, N, R, params.uvcut,
                                                params.uvcutval)
                M = M_operator(domain=s_space, sym=True, imp=True, para=[N, R])
                j = R.adjoint_times(N.inverse_times(d))
                if params.pointresolve:
                    D = Dmm_operator(domain=s_space, sym=True, imp=True,
                                     para=[S, M, m, j, params.M0_start, 
                                     params.rho0, u, params])
                else:
                    D = D_operator(domain=s_space, sym=True, imp=True, 
                                   para=[S, M, m, j, params.M0_start, 
                                   params.rho0, params])
        

                if params.fastresolve:
                    # re-initialize fastresolve operators to new uv-space
                    fr_operators = ra.initialize_fastresolve(s_space,
                                                             k_space, R, d,
                                                             N.diag())

            if params.fastresolve:
                # Update effected operators for fastresolve mode
                d, N, R = ra.resolve_to_fastresolve(fr_operators)
                M = M_operator(domain=s_space, sym=True, imp=True, para=[N, R])
                j = R.adjoint_times(N.inverse_times(d))
                if params.pointresolve:
                    D = Dmm_operator(domain=s_space, sym=True, imp=True,
                                     para=[S, M, m, j, params.M0_start, 
                                     params.rho0, u, params])
                else:
                    D = D_operator(domain=s_space, sym=True, imp=True, 
                                   para=[S, M, m, j, params.M0_start, 
                                   params.rho0, params])

            pold = pspec
            t1 = time()
            pspec = resolve_pspec(m, D, S, N, R, d, k_space, alpha_prior,\
                params, logger)
            t2 = time()
            logger.success("Completed RESOLVE power spectrum"\
                +" iteration cycle.")
            logger.message("Time to complete: " + str((t2 - t1) / 3600.)\
                + ' hours.')
                
            # Intermediate saves of iteration
            
            # Save pspec
            kindex = k_space.get_power_indices()[0]
            plist.append(pspec)
            utils.save_results(kindex,"ps, iter #" + str(global_iteration), \
                    'resolve_output_' + str(params.save) +\
                    "/p_reconstructions/" + params.save + "_p" +\
                    str(global_iteration), value2=pspec,log='loglog')
            # powevol plot 
            pl.figure()
            for i in range(len(plist)):
                pl.loglog(kindex, plist[i], label="iter" + str(i))
            pl.title("Global iteration pspec progress")
            pl.legend()
            pl.savefig('resolve_output_' + str(params.save) + \
                "/p_reconstructions/" + params.save + "_powevol.png")
            pl.close()
                    
            # Update S and D
            S.set_power(newspec=pspec,bare=True)
            D.para[0] = S
            
            # Check convergence progress
            if np.max(np.abs(utils.log(pspec)/utils.log(pold))) < np.log(1e-1):
                logger.message('Power spectrum seems converged, increase'\
                    + ' convergence measure by one.')
                convergence += 1
            if check_global_convergence(convergence, runlist_element, params,\
                logger):
                break

            if params.uvcut or params.fastresolve:

                # change back data operator definitions to original
                d, N, R, d_space = readjust_uv(dorgval, Norgdiag, s_space,
                                               uorg, vorg, Aorg)

                M = M_operator(domain=s_space, sym=True, imp=True,
                               para=[N, R])
                j = R.adjoint_times(N.inverse_times(d))

        # Resolve power spectrum run for the extended field        
        if runlist_element == 'mf_resolve_pspec_a':
            
            pold_a = pspec_a
            t1 = time()
            pspec_a = mf_resolve_pspec_a(a, Da, k_space, alpha_prior,\
                params, logger)
            t2 = time()
            logger.success("Completed RESOLVE spectral index power spectrum"\
                +" iteration cycle.")
            logger.message("Time to complete: " + str((t2 - t1) / 3600.)\
                + ' hours.')
                
            # Intermediate saves of iteration
            
            # Save pspec_a
            kindex = k_space_a.get_power_indices()[0]
            p_a_list.append(pspec_a)
            utils.save_results(kindex,"ps, iter #" + str(global_iteration), \
                'resolve_output_' + str(params.save) +\
                "/p_reconstructions/" + params.save + "_pa" + \
                "_" + str(global_iteration), value2=pspec,log='loglog')
            # powevol plot 
            pl.figure()
            for i in range(len(p_a_list)):
                pl.loglog(kindex, plist[i], label="iter" + str(i))
            pl.title("Global iteration pspec progress")
            pl.legend()
            pl.savefig('resolve_output_' + str(params.save) + \
                "/p_reconstructions/" + params.save + "_powevol_a.png")
            pl.close()
                    
            # Update S and D
            Sa.set_power(newspec=pspec,bare=True)
            Da.para[0] = S
            
            # Check convergence progress
            if np.max(np.abs(utils.log(pspec_a)/utils.log(pold_a)))\
                < np.log(1e-1):
                    logger.message('Power spectrum seems converged, increase'\
                        + ' convergence measure by one.')
                    convergence += 1
            if check_global_convergence(convergence, runlist_element, params,\
                logger):
                break

        # Resolve uncertainty approximation for the extended field
        if runlist_element == 'uncertainty_m':

            t1 = time()
            Dhat = uncertainty_m(D, params, logger)
            t2 = time()
            logger.success("Completed RESOLVE uncertainty"
                           + " calculation.")
            logger.message("Time to complete: " + str((t2 - t1) / 3600.)
                           + ' hours.')

            # Intermediate saves of iteration
            utils.uncertainty_functions(m, Dhat)

        # Resolve uncertainty approximation for the spectral index
        if runlist_element == 'uncertainty_a':

            t1 = time()
            Dhat_a = uncertainty_m(D, params, logger)
            t2 = time()
            logger.success("Completed RESOLVE uncertainty"
                           + " calculation.")
            logger.message("Time to complete: " + str((t2 - t1) / 3600.)
                           + ' hours.')

            # Intermediate saves of iteration
            utils.uncertainty_functions(a, Dhat_a)
            
                
        # Resolve simple noise estimation        
        if runlist_element == 'noise_estimation':
            
            if (params.pointresolve and not params.multifrequency):
                mapguess = exp(m+u)
            elif (not params.pointresolve and not params.multifrequency):
                mapguess = exp(m)
            else:
                raise NotImplementedError('Noise estimation not yet\
                implemented for multifrequency settings')

            t1 = time()
            est_var = noise_estimation(mapguess, d, R, params, logger)
            t2 = time()
            logger.success("Completed RESOLVE noise estimation cycle.")
            logger.message("Time to complete: " + str((t2 - t1) / 3600.)\
                + ' hours.')
            logger.message('Old variance iteration '+\
                str(global_iteration-1)+':' + str(N.diag()))
            logger.message('New variance iteration '+\
                str(global_iteration)+':' + str(est_var))
                
            # Save est_var
            np.save('resolve_output_' + str(params.save)\
                + '/general/oldvar_'+str(global_iteration),N.diag())
            np.save('resolve_output_' + str(params.save)\
                +'/general/newvar_'+str(global_iteration),est_var)
            np.save('resolve_output_' + str(params.save) +'/general/absdmean_'\
                +str(global_iteration),abs(d.val).mean())
            np.save('resolve_output_' + str(params.save) +\
                '/general/absRmmean_' + str(global_iteration),\
                abs(R(exp(m)).val*R.target.num()).mean())
            
            # Update N, M, and j
            # This needs be done to ensure that a change back from fastresolve
            # to resolve translates the updated noise variance
            if  any('fastresolve' in string for string in changeplist):           
                Norg.set_diag(est_var*np.ones(np.shape(Norg.diag())))
            N.set_diag(est_var*np.ones(np.shape(N.diag())))
            M = M_operator(domain=s_space, sym=True, imp=True, para=[N, R])
            j = R.adjoint_times(N.inverse_times(d))

    globalt2 = time()
    logger.success("Completed full RESOLVE cycle.")
    logger.message("Time to complete: " + str((globalt2 - globalt1) / 3600.)\
        + ' hours.')
        
    # final plotting
        
    # All runs are standard Resolve reconstructions        
    if ((not params.multifrequency) and (not params.pointresolve)):
        finalplots(params, m, pspec, k_space)
    
    # All runs are simulated standard Resolve reconstructions        
    if ((not params.multifrequency) and (not params.pointresolve) and 
         params.simulating):
        finalplots(params, m, pspec, k_space, powspec_sim=powspec_sim)
    
    # All runs incorporate Point-Resolve reconstructions but no MF-Resolve
    elif ((not params.multifrequency) and (params.pointresolve)):
        finalplots(params, m, pspec, k_space, u=u)
        
    # All runs are simulated and incorporate Point-Resolve reconstructions 
    # but no MF-Resolve      
    if ((not params.multifrequency) and (not params.pointresolve) and 
         params.simulating):
        finalplots(params, m, pspec, k_space, u=u, powspec_sim=powspec_sim)

    # All runs incorporate MF-Resolve reconstructions but no Point-Resolve
    elif ((params.multifrequency) and (not params.pointresolve)):
        finalplots(params, m, pspec, k_space, a=a,pspec_a=pspec_a, k_space_a=
        k_space_a)

    # All runs incorporate both Point- and MF-Resolve reconstructions
    elif ((params.multifrequency) and (params.pointresolve)):
        finalplots(params, m, pspec, k_space, u=u, a=a,pspec_a=pspec_a, 
        k_space_a=k_space_a)
    


#--------------------------Resolve functions-----------------------------------


def datasetup(params, logger):
    """
        
    Data setup, feasible for the nifty framework.
        
    """
    
    logger.header2("\n Running data setup.")   
        
    #Somewhat inelegant solution, but WSclean needs its own I/O 
    if params.ftmode == 'wsclean':
        
        wscleanpars = w.ImagingParameters()
        wscleanpars.msPath = str(params.datafn)
        wscleanpars.imageWidth = int(params.imsize)
        wscleanpars.imageHeight = int(params.imsize)
        wscleanpars.pixelScaleX = str(params.cellsize)+'rad'
        wscleanpars.pixelScaleY = str(params.cellsize)+'rad'
        wscleanpars.extraParameters = '-weight natural -nwlayers 1 -j 4'\
        +' -channelrange 0 1'
      
        wscOP = w.Operator(wscleanpars)
        vis = wscOP.read()[0]
        if params.noise_est == 'full':
            logger.warn('Full noise estimate needs to be done manually on the'\
                +' measurement set before using WSclean routines.')
        sigma = np.zeros(len(vis))
        try:
            logger.warn('Trying to load in MS header as numpy file while'\
                + ' using WSclean. If not available, FITS image output will'\
                + ' not work.')
            params.summary = np.load(params.datafn+'_summary.npy')
        except IOError:
            logger.warn('No numpy MS header file found. FITS output will'\
                +' be deactivated.')
            params.summary = None
        
    # data and noise settings. Inelegant if-not statement needed
    # because wsclean routines don't explicitly read out these things
    if not params.ftmode == 'wsclean':

        if params.python_casacore:
            vis, sigma, u, v, flags, freqs, nchan, nspw, nvis,\
                params.summary = utils.read_data_from_ms_in_python(\
                    params.datafn, logger)
        else:
            vis, sigma, u, v, flags, freqs, nchan, nspw, nvis,\
                params.summary = utils.load_numpy_data(params.datafn, logger)

        u = np.array(u)
        v = np.array(v)
    
        sspw,schan = params.freq[0], params.freq[1]
        vis = vis[sspw][schan]
        sigma = sigma[sspw]
        u = u[sspw][schan]
        v = v[sspw][schan] 
        flags = flags[sspw][schan]
        
        
        # cut away flagged data
        vis = np.delete(vis,np.where(flags==0))
        u = np.delete(u,np.where(flags==0))
        v = np.delete(v,np.where(flags==0))
        sigma = np.delete(sigma,np.where(flags==0))
        

    # Dumb simple estimate can be done now after reading in the data.
    # No time information needed.
    if params.noise_est == 'simple':
        
        variance = np.var(np.array(vis))*np.ones(np.shape(np.array(vis)))\
            .flatten()
    
    elif params.noise_est == 'simple_from_ecf':

        variance = np.var(np.array(vis))*np.ones(np.shape(np.array(vis)))\
            .flatten() /5.
    
    elif params.noise_est == 'from_file':  
        
        variance = np.ones(np.shape(np.array(vis)))\
            *np.load('/vol/henrikj/data2/henrikju/data/' +\
            +'cygnus/images/B23JAN84_images/resolve/Mar16_Restart/'+\
            +'resolve_output_REDOFLAGS_cygnus_cleansg_SD_simpleest_ecf/'+
            +'general/newvar_5.npy')

    elif params.noise_est == 'SNR_assumed':
        
        variance = np.ones(np.shape(vis))*np.mean(np.abs(vis*vis))/ \
            (1.+params.SNR_assumed)
    else:
        variance = (np.array(sigma)**2).flatten()

    if params.ftmode != 'wsclean':
        # basic diagnostics                                                         
        # maximum k-mode and resolution of data                                      
        uvrange = np.array([np.sqrt(u[i]**2 + v[i]**2) for i in range(len(u))])
        dx_real_rad = (np.max(uvrange))**-1
        logger.message('\nMax. instrumental resolution\n' + 'rad ' \
            + str(dx_real_rad) + '\n' + 'asec ' + str(dx_real_rad/asec2rad))

        utils.save_results(u,'UV', 'resolve_output_' + str(params.save) +\
            '/general/' + params.save + "_uvcov", plotpar='o', value2 = v)

        d_space = point_space(len(u), datatype = np.complex128)
    else:
        d_space = point_space(len(vis), datatype = np.complex128)

    s_space = rg_space(params.imsize, naxes=2, dist = params.cellsize, \
        zerocenter=True)
    
    # primary beam function
    if params.pbeam:
        try:
            logger.message('Attempting to load primary beam from'\
            + ' file' + str(params.pbeam))
            A = np.load(params.pbeam)
        except IOError:
            logger.warn('Could not find primary beam file. Set pb to 1.')
            A = 1.
    else:
        A = 1.
        
    # response operator
    if params.ftmode == 'wsclean':
        # uv arbitrarily set to 0, not needed
        R = r.response(s_space, d_space, 0, 0, A, mode = params.ftmode,\
                       wscOP=wscOP)
    else:
        R = r.response(s_space, d_space, u, v, A, mode = params.ftmode)

    d = field(d_space, val=vis)

    # N_operator deprecated but left for later reference
    #N = N_operator(domain=d_space,imp=True,para=[variance])
    variance[variance<=0]=1e-10
    N = diagonal_operator(domain=d_space,diag=variance)
    
    # dirty image from CASA or Resolve for comparison
    if params.ftmode == 'wsclean':
        di = R.adjoint_times(d)
    else:
        R.adjointfactor = 1
        di = R.adjoint_times(d) #* s_space.vol[0] * s_space.vol[1]
        R = r.response(s_space, d_space, u, v, A)   
    
    if params.ftmode != 'wsclean':
        # more diagnostics 
        # plot the dirty beam
        uvcov = field(d_space,val=np.ones(len(u), dtype = np.complex128))
        db = R.adjoint_times(uvcov) * s_space.vol[0] * s_space.vol[1]       
        utils.save_results(db,"dirty beam", 'resolve_output_' +str(params.save)+\
            '/general/' + params.save + "_db")
    
    # plot the dirty image
    utils.save_results(di,"dirty image",'resolve_output_' +str(params.save)+\
        '/general/' + params.save + "_di")

    return  d, N, R, di, d_space, s_space


def datasetup_MF(params, logger):
    """
        
    Data setup, feasible for the nifty framework.
        
    """    

    logger.header2("\nRunning multifrequency data setup.")

    if params.ftmode == 'wsclean':
        raise NotImplementedError('For wideband mode, only gfft support is'\
        + ' available. Consider using ftmode=gfft.')
    
    vis, sigma, u, v, flags, freqs, nchan, nspw, nvis, params.summary = \
        utils.load_numpy_data(params.datafn, logger)
    
    u = np.array(u)
    v = np.array(v)
    freqs = np.array(freqs)
    nspw = len(nspw)
    nchan = nchan[0]
    nvis = nvis[0]
    
    # Dumb simple estimate can be done now after reading in the data.
    # No time information needed.
    if params.noise_est == 'simple':
        
        variance = np.var(np.array(vis))*np.ones(np.shape(np.array(vis)))\
            .flatten()
    elif params.noise_est == 'SNR_assumed':
        
        variance = np.ones(np.shape(sigma))*np.mean(np.abs(vis*vis))/ \
        (1.+params.SNR_assumed)
    else:
        
        variance = (np.array(sigma)**2).flatten()
        
    # Fix of possible problems in noise estimation
    if np.any(variance) < 1e-10:
        variance[variance<1e-10] = np.mean(variance[variance>1e-10])

    # basic diagnostics                                                         
    # maximum k-mode and resolution of data
    uflat = u.flatten()
    vflat = v.flatten()
    uvrange = np.array([np.sqrt(uflat[i]**2 + vflat[i]**2) \
        for i in range(len(u))])
    dx_real_rad = (np.max(uvrange))**-1
    logger.message('\nMax. instrumental resolution over all spw\n' + 'rad ' \
        + str(dx_real_rad) + '\n' + 'asec ' + str(dx_real_rad/asec2rad))

    utils.save_results(uflat,'UV_allspw', 'resolve_output_' + str(params.save) +\
        '/general/' + params.save + "_uvcov", \
        plotpar='o', value2 = vflat)
    
    d_space = point_space(nspw*nchan*nvis, datatype = np.complex128)
    s_space = rg_space(params.imsize, naxes=2, dist = params.cellsize,\
        zerocenter=True)
    
    # primary beam function
    if params.pbeam:
        try:
            logger.message('Attempting to load primary beam from'\
            + 'file' + str(params.pbeam))
            A = np.load(params.pbeam)
        except IOError:
            logger.warn('Could not find primary beam file. Set pb to 1.')
            A = np.array([[np.ones((int(params.imsize),int(params.imsize)))\
            for k in range(nchan)] for j in range(nspw)])
    else:
        
        A = np.array([[np.ones((int(params.imsize),int(params.imsize)))\
            for k in range(nchan)] for j in range(nspw)])
        
    # response operator
    R = r.response_mfs(s_space, d_space, \
                       u,v,A,nspw,nchan,nvis,freqs,params.reffreq[0],\
                       params.reffreq[1],params.ftmode)
    
    d = field(d_space, val=np.array(vis).flatten())

    # N_operator deprecated but left for later reference
    #N = N_operator(domain=d_space,imp=True,para=[variance])
    N = diagonal_operator(domain=d_space,diag=variance)

    tempsave = R.adjointfactor
    R.adjointfactor = np.ones((nspw,nchan))
    di = R.adjoint_times(d) * s_space.vol[0] * s_space.vol[1]
    R.adjointfactor = tempsave
        
    # more diagnostics 
    # plot the dirty beam
    uvcov = field(d_space,val=np.ones(np.shape(np.array(vis).flatten()), \
         dtype = np.complex128))            
    db = R.adjoint_times(uvcov) * s_space.vol[0] * s_space.vol[1]       
    utils.save_results(db,"dirty beam",'resolve_output_' + str(params.save)+\
        '/general/' + params.save + "_db")
        
    # plot the dirty image
    utils.save_results(di,"dirty image",'resolve_output_' +str(params.save)+\
        '/general/' + params.save + "_di")


    return  d, N, R, di, d_space, s_space
 


def starting_guess_setup(params, logger, s_space, d_space, di):
    
    # Starting guesses for m_s
    
    if params.init_type_m == 'const':
        m = field(s_space, val = params.m_start)

    elif params.init_type_m == 'dirty':
        m = field(s_space, target=s_space.get_codomain(), val=np.mean(log(di-di.min()+1e-12)))   
        
    else:
        if params.sg_logim:
            if params.sg_meanim:
                m = field(s_space, target=s_space.get_codomain(), \
                val=zoom(np.load(params.init_type_m),zoom=\
                    params.zoomfactor).mean())
            else:    
                m = field(s_space, target=s_space.get_codomain(), \
                    val=zoom(np.load(params.init_type_m),zoom=\
                        params.zoomfactor))
        
        else:
            
            expm_val = np.abs(zoom(np.load(params.init_type_m),zoom=\
                params.zoomfactor))
            expm_val[expm_val==0] = 1e-12
            if params.sg_meanim:
                m = field(s_space, target=s_space.get_codomain(), \
                val=log(expm_val.mean()))
            else:
                m = field(s_space, target=s_space.get_codomain(), \
                    val=log(expm_val))    
    
    # Optional starting guesses for m_u
            
    if params.pointresolve:
        
        
        if params.init_type_u == 'const':
            u = field(s_space, val = params.u_start)
        
        elif params.init_type_u == 'dirty':
            u = field(s_space, target=s_space.get_codomain(), val=di)   
            
        else:
            if params.sg_logim:
                if params.sg_meanim:
                    u = field(s_space, target=s_space.get_codomain(), \
                    val=zoom(np.load(params.init_type_u),zoom=\
                        params.zoomfactor).mean())
                else:    
                    u = field(s_space, target=s_space.get_codomain(), \
                        val=zoom(np.load(params.init_type_u),zoom=\
                            params.zoomfactor))
            
            else:
                
                expu_val = np.abs(zoom(np.load(params.init_type_u),zoom=\
                    params.zoomfactor))
                expu_val[expu_val==0] = 1e-12
                if params.sg_meanim:
                    u = field(s_space, target=s_space.get_codomain(), \
                    val=log(expu_val.mean()))
                else:
                    u = field(s_space, target=s_space.get_codomain(), \
                        val=log(expu_val))
                
    if params.rho0 == 'from_sg':
        
        if params.pointresolve:
            params.rho0 = np.mean(exp(m.val[np.where(exp(m).val>=\
                np.max(exp(m).val)/ 10)] + exp(u.val[np.where(exp(u).val>=\
                np.max(exp(u).val)/ 10)])))
        else:
             params.rho0 = np.mean(exp(m.val[np.where(exp(m).val>=\
                np.max(exp(m).val)/ 10)]))
        logger.message('rho0 was calculated from sg as: ' + str(params.rho0)) 
        
    if not params.rho0 == 1.:
        
        m -= log(params.rho0)
        if params.pointresolve:
            u -= log(params.rho0)

    np.save('resolve_output_' + str(params.save)+'/general/rho0',params.rho0)
    if params.rho0 < 0:
        logger.warn('Monopole level of starting guess negative. Probably due \
            to too many imaging artifcts in userimage')
        
            
    # Starting guesses for pspec 

    # Basic k-space
    k_space = s_space.get_codomain()
        
    #Adapts the k-space properties if binning is activated.
    if params.bins:
        k_space.set_power_indices(log=True, nbins=params.bins)
    
    # k-space properties    
    kindex,rho_k,pindex,pundex = k_space.get_power_indices()

    # Simple k^2 power spectrum with p0 from numpars and a fixed monopole from
    # the m starting guess
    if params.init_type_p == 'k^2_mon':
        pspec = np.array((1+kindex)**-2 * params.p_start)
        pspec_temp = m.power(pindex=pindex, kindex=kindex, rho=rho_k)
        #see notes, use average power in dirty map to constrain monopole
        if params.highmonopole:
            pspec[0] = 1.e7
        else:
            pspec[0] = (np.prod(k_space.vol)**(-2) * utils.log(\
                np.sqrt(pspec_temp[0]) *  np.prod(k_space.vol))**2) / 2.
    # default simple k^2 spectrum with free monopole
    elif params.init_type_p == 'k^2':
        pspec = np.array((1+kindex)**-2 * params.p_start)    
    # constant power spectrum guess 
    elif params.init_type_p == 'const':
        pspec = np.ones(len(kindex)) * params.p_start
    # maksim-fastresolve style starting guess
    elif params.init_type_p == 'structure':
        pspec = (kindex+k_space.vol[0])**(-3.)
        fac = 0.15/(4*pspec[1]*k_space.vol.prod()**2)
        pspec *= fac
        pspec[0] *= 1.e7
    # power spectrum file 
    else:
        try:
            logger.message('Using pspec from file '+params.init_type_p)
            pspec = np.load(params.init_type_p)
        except IOError:
            logger.failure('No pspec file found.')
            raise IOError
        
    # check validity of starting pspec guesses
    if np.any(pspec) == 0:
        pspec[pspec==0] = 1e-25

    # Wideband mode starting guesses
    if params.multifrequency:
        
        # Starting guesses for m_a

        if params.init_type_a == 'const':
            a = field(s_space, val = params.a_start)
            
        else:

            if params.sg_meanim:
                a = field(s_space, target=s_space.get_codomain(), \
                val=zoom(np.load(params.init_type_a),zoom=\
                    params.zoomfactor).mean())
            else:    
                a = field(s_space, target=s_space.get_codomain(), \
                    val=zoom(np.load(params.init_type_a),zoom=\
                        params.zoomfactor))
            
        
        # Spectral index pspec starting guesses
        
        # Basic k-space
        k_space_a = s_space.get_codomain()
        
        #Adapts the k-space properties if binning is activated.
        if params.bins_a:
            k_space_a.set_power_indices(log=True, nbins=params.bins)
    
        # k-space properties    
        kindex_a,rho_k,pindex,pundex = k_space_a.get_power_indices()
        
        # default simple k^2 spectrum with free monopole
        if params.init_type_p_a == 'k^2':
            pspec_a = np.array((1+kindex_a)**-2 * params.p_start_a)
            pspec_temp = a.power(pindex=pindex, kindex=kindex, rho=rho_k)
        # k^2 spectrum with fixed monopole
        if params.init_type_p_a == 'k^2_mon':
            pspec_a = np.array((1+kindex)**-2 * params.p_start)
            #see notes, use average power in dirty map to constrain monopole
            if params.highmonopole_a:
                pspec_a[0] = 1e7
            else:
                pspec_a[0] = (np.prod(k_space.vol)**(-2) * utils.log(\
                    np.sqrt(pspec_temp[0]) *  np.prod(k_space.vol))**2) / 2.        
        # constant power spectrum guess 
        elif params.init_type_p_a == 'const':
            pspec_a = params.p_start_a
        # power spectrum from file 
        else:
            try:
                logger.message('Using pspec from file '+params.init_type_p_a)
                pspec_a = np.load(params.init_type_p_a)
            except IOError:
                logger.failure('No pspec file found.')
                raise IOError

        if np.any(pspec_a) == 0:
            pspec_a[pspec_a==0] = 1e-25
 
    # diagnostic plot of m starting guess
    utils.save_results(exp(m.val),"TI exp(Starting guess)",\
        'resolve_output_' + str(params.save) + '/m_reconstructions/' +\
        params.save + "_expm0", rho0 = params.rho0)
    pars.write_output_to_fits(np.transpose(exp(m.val)*params.rho0),params, \
        notifier='0', mode='I')
    if params.multifrequency:
        utils.save_results(a.val,"Alpha Starting guess",\
            'resolve_output_' + str(params.save) + '/m_reconstructions/' +\
            params.save + "_ma0", rho0 = params.rho0) 
        pars.write_output_to_fits(np.transpose(a.val),params, notifier='0', \
               mode='a') 
    if params.pointresolve:
        utils.save_results(exp(u.val),"TI exp(Starting guess)",\
            'resolve_output_' + str(params.save) + '/u_reconstructions/' +\
            params.save + "_expu0", rho0 = params.rho0)    
        pars.write_output_to_fits(np.transpose(exp(u.val)*params.rho0),params, \
            notifier='0', mode='I_u')       
                
        utils.save_results(exp(m.val)+exp(u.val),"TI exp(Starting guess)",\
            'resolve_output_' + str(params.save) + '/mu_reconstructions/'+\
            params.save + "_expmu0", rho0 = params.rho0)    
        pars.write_output_to_fits(np.transpose(exp(m.val)+exp(u.val)*\
            params.rho0), params, notifier='0', mode='I_mu')         
    
    # All runs are standard Resolve reconstructions
    if ((not params.multifrequency) and (not params.pointresolve)):
        return m, pspec, params, k_space
    
    # All runs incorporate Point-Resolve reconstructions but no MF-Resolve
    elif ((not params.multifrequency) and (params.pointresolve)):
        return m, pspec, u, params, k_space

    # All runs incorporate MF-Resolve reconstructions but no Point-Resolve
    elif ((params.multifrequency) and (not params.pointresolve)):
        return m, pspec, a, pspec_a, params, k_space, k_space_a

    # All runs incorporate both Point- and MF-Resolve reconstructions
    elif ((params.multifrequency) and (params.pointresolve)):
        return m, pspec, u, a, pspec_a, params, k_space, k_space_a


def operator_setup(d, N, R, pspec, m, logger, k_space, \
    params, a=0., u=0., pspec_a=0., k_space_a=0.):
    """
    """
    
    s_space = R.domain
    kindex = k_space.get_power_indices()[0]
    
    if params.multifrequency:
        aconst = args[0]
         
    # Sets the alpha prior parameter for all modes
    if params.pspec:
        if params.alpha_prior:
            alpha_prior = np.ones(np.shape(kindex)) * params.alpha_prior
        else:
            alpha_prior = np.ones(np.shape(kindex))
        if params.multifrequency:
            # Sets the spectral index alpha prior parameter for all modes
            if params.alpha_prior_a:
                alpha_prior_a =  np.ones(np.shape(kindex)) *\
                    params.alpha_prior_a
            else:
                alpha_prior_a = np.ones(np.shape(kindex))
        
    # Defines operators
    S = power_operator(k_space, spec=pspec, bare=True)
    if params.multifrequency:
        M = MI_operator(domain=s_space, sym=True, imp=True,\
            para=[N, R, aconst])
        j = R.adjoint_times(N.inverse_times(d), a = aconst)
    else:    
        M = M_operator(domain=s_space, sym=True, imp=True, para=[N, R])
        j = R.adjoint_times(N.inverse_times(d))
    if (params.pspec or params.uncertainty):
        if params.pointresolve:
            D = Dmm_operator(domain=s_space, sym=True, imp=True,\
                para=[S, M, m, j, params.M0_start, params.rho0, u, params])
        else:
            D = D_operator(domain=s_space, sym=True, imp=True, para=[S, M, m,\
                j, params.M0_start, params.rho0, params])
        if params.multifrequency:
            Sa = power_operator(k_space_a, spec=pspec_a, bare=True)
            Da = Da_operator(domain=s_space, sym=True, imp=True,\
                para=[Sa, R, N, a, params.M0_start_a, m, d,\
                params.rho0, params])
        
    
    # diagnostic plots of information source j, aka the dirty image
    if params.multifrequency:
        utils.save_results(j,"j",'resolve_output_' + str(params.save) +\
            '/general/' + params.save + '_j' + '_mf',rho0 = params.rho0)
    else:
        utils.save_results(j,"j",'resolve_output_' + str(params.save) +\
            '/general/' + params.save + '_j',rho0 = params.rho0)

    
    # All runs are without pspec, uncertainty and MF 
    if ((not params.multifrequency) and (not params.pspec)\
        and (not params.uncertainty)):
        return j, M, S

    # All runs are with pspec and/or uncertainty but without MF
    elif ((not params.multifrequency) and (params.pspec \
        or params.uncertainty)):
        return j, M, D, S, alpha_prior
        
    # All runs are with pspec and/or uncertainty and with MF
    elif (params.multifrequency and (params.pspec \
        or params.uncertainty)):
        return j, M, D, S, alpha_prior, Da, Sa, alpha_prior_a
        
def check_global_convergence(convergence, runlist_element, params, logger):
    
    if convergence >= params.final_convlevel:
        logger.message('Global convergence achieved at runlist step'\
            + runlist_element + '.')
        return True
    else:
        return False

# contains all the final plotting routines to keep them out of main
def finalplots(params, m, pspec, k_space, u=None, a=None, pspec_a=None, 
               k_space_a=None, powspec_sim=None):
    
    utils.plot_figure_with_axes(exp(m.val), "exp(Solution m)", 
                          'resolve_output_' + str(params.save) +
                           '/m_reconstructions/' + params.save + "_expmfinal",
                          params, cmap='Greys', rho0=params.rho0)
    pars.write_output_to_fits(np.transpose(exp(m.val)*params.rho0),\
        params, notifier='final',mode='I')
    
    if params.pspec:
        utils.save_results(k_space.get_power_indices()[0],\
                     "final power spectrum", 'resolve_output_' +\
                     str(params.save) + '/p_reconstructions/' + params.save +\
                     "_powfinal", value2 = pspec, log='loglog')
    
    if params.multifrequency:
        utils.plot_figure_with_axes(a.val, "Solution a", 'resolve_output_' + 
                          str(params.save) + '/m_reconstructions/' +
                          params.save + "_mafinal", params,
                          cmap='Greys', rho0=params.rho0)
        pars.write_output_to_fits(np.transpose(a.val),params,\
        notifier ='final', mode='a')
        
        utils.save_results(k_space_a.get_power_indices()[0],\
                     "final spectral index power spectrum",\
                     'resolve_output_' + str(params.save) +\
                     '/p_reconstructions/' + params.save +\
                     "_powfinal_a", value2 = pspec_a, log='loglog')
    
    
    if params.pointresolve:     
        utils.save_results(exp(u.val),"exp(Solution u)",\
            'resolve_output_' + str(params.save) + '/u_reconstructions/' + \
            params.save + "_expufinal", rho0 = params.rho0)
        utils.plot_figure_with_axes(exp(u.val), "exp(Solution u)",  
                           'resolve_output_' + str(params.save) + 
                          '/u_reconstructions/' + params.save + "_expufinal", 
                          params, cmap='Greys', rho0=params.rho0)
        pars.write_output_to_fits(np.transpose(exp(u.val)*params.rho0),\
            params, notifier='final',mode='I_u')
    
    if params.simulating:
        
        pl.figure()
        pl.loglog(k_space.get_power_indices()[0], pspec, label="final")
        pl.loglog(k_space.get_power_indices()[0],powspec_sim,\
            label="simulated") 
        pl.title("Compare final and simulated power spectrum")
        pl.legend()
        pl.savefig("resolve_output_" + str(params.save) +"/p_reconstructions/"\
            + params.save + "_compare.png")
        pl.close()
    
    
    

# helper function that adjusts a requested uvcut during a specific runfile
# command        
def adjust_uvcut(d, N, R, uvcut, uvcutval):

    u = R.u
    v = R.v
    dval = d.val
    
    if uvcut == 'smaller':
        mask = (np.sqrt(u**2+v**2) < uvcutval)
    elif uvcut == 'larger':
        mask = (np.sqrt(u**2+v**2) > uvcutval)
        
    u = np.delete(u,np.where(mask==0))
    R.u = u
    v = np.delete(v,np.where(mask==0))
    R.v = v
    
    dval = np.delete(dval,np.where(mask==0))
    d_space = point_space(len(dval), datatype = np.complex128)
    d = field(d_space, val=dval)
    R.target = d_space
    
    Ndiag = N.diag()
    Ndiag = np.delete(Ndiag,np.where(mask==0))
    N = diagonal_operator(domain=d_space,diag=Ndiag)
    
    return d, N, R, d_space       

# readjusts all data/uv operators after a runfile command that requested
# a uvcut and or fastresolve mode 
def readjust_uv(dorgval, Norgdiag, s_space, u, v, A):
    
    d_space = point_space(len(dorgval), datatype = np.complex128)
    d = field(d_space, val=dorgval)
    N = diagonal_operator(domain=d_space,diag=Norgdiag)
    if params.ftmode == 'wsclean':
        # uv arbitrarily set to 0, not needed
        R = r.response(s_space, d_space, 0, 0, A, mode = params.ftmode,
                          wscOP=wscOP)
    else:
        R = r.response(s_space, d_space, u, v, A, mode=
                          params.ftmode)
    
    return d, N, R, d_space

#------------------- Command Line Parsing ------------------------------------- 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("datafn",type=str,
                        help="directory of numpy-saved visibility data files.")
    parser.add_argument("cell",type=float,
                        help="cellsize in arcsec.")
    parser.add_argument("imsize",type=int, 
                        help="Imsize in number of pixel.")
    parser.add_argument("resolve_mode",type=str,
                        help="Indicates the mode of package that should be"\
                        + " used, see documentation.")
    parser.add_argument("-s","--save", type=str, default='save',
                        help="output string for save directories.")
    parser.add_argument("-p","--python_casacore", type=str, default=False,
                        help="Uses the python-casacore bindings to read "
                             + "data directly from the measurement set. Use "
                             + " the datafn argument to indicate the ms.")                    
    parser.add_argument("-v","--verbosity", type=int, default=2,
                        help="Reset verbosity level of code output. Default"\
                            +' is 2/5.')                    
    args = parser.parse_args()
    
    # Load the runfiles and parameter-change-files for the requested resolve 
    # mode
    if not '.run' in args.resolve_mode:
        args.resolve_mode = args.resolve_mode + '.run'
    runlist, changeplist = pars.get_parameter_and_runscript_from_runfile(\
        args.resolve_mode)
    # Load the default parameter set
    params = pars.parameters(pars.load_default_parameters(), args.datafn, \
                            args.imsize, args.cell, args.save, 
                            args.python_casacore, args.verbosity)
    
    resolve(runlist, changeplist, params)
