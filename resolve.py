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
import sys
import datetime
import argparse
import csv


print '\nModule loading information:'

#import necessary standard scientific modules
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
from nifty import *
from nifty import nifty_tools as nt
import pyfits

#import RESOLVE-package modules
import utility_functions as utils
import simulation.resolve_simulation as sim
import response_approximation.UV_algorithm as ra
from operators import *
import response as r
import Messenger as M

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
asec2rad = 4.84813681e-6
gsave = ''
gcallback = 3

def resolve(params, numparams):

    """
        A RESOLVE-reconstruction.
    
        Args:
            ms: Measurement set that holds the data to be imaged.
            imsize: Size of the image.
            cellsize: Size of one grid cell.
            algorithm: Which algorithm to be used. 
                1) 'ln-map': standard-resolve 
                2) 'wf': simple Wiener Filter 
                3) 'Gibbs': Gibbs-Energy minimizer
                4) 'samp': Sampling
                5) 'ln-map_u': point-resolve
            init_type_s: What input should be used to fix the monopole of the map
                by effectively estimating rho_0.
                1) 'dirty'
                2) '<user-defined input image pathname>'
            use_init_s: Defines what is used as a starting guess 
                1) False (default): Use a constant field close to zero
                2) 'starting_guess': Use the field defined in init_type_s
                3) '<lastit path/filename>': Load from previous iteration
            init_type_p: Starting guess for the power spectrum.
                1) 'k^2': Simple k^2 power spectrum.
                2) 'k^2_mon': Simple k^2 power spectrum with fixed monopole.
                3) 'constant': constant power spectrum using p0 from the 
                    numparameters.
                4) '<lastit path/filename>': Load from previous iteration
            init_type_p_a: Starting guess for the spectral index power spectrum.
                1) 'k^2': Simple k^2 power spectrum.
                3) 'constant': constant power spectrum using p0_a from the 
                    numparameters.
                4) 4) '<lastit path/filename>': Load from previous iteration
            freq: Whether to perform single band or wide band RESOLVE.
                1) [spw,cha]: single band
                2) 'wideband'
            pbeam: user-povided primary beam pattern.
            uncertainty: Whether to attempt calculating an uncertainty map \
                (EXPENSIVE!).
            noise_est: Whether to take the measured noise variances or make an \
                estimate for them.
                1) 'simple': simply try to estimate the noise variance using \
                    the rms in the visibilties.
                2) 'ecf': try a poor-man's ECF. WARNING: Changes the whole \
                    algorithm and makes RESOLVE considerably slower.
            map_algo: Which optimization algorithm to use for the signal estimate.\
                1) 'sd': steepest descent, recommended for robustness.
                2) 'lbfgs'
            pspec_algo: Which optimization algorithm to use for the pspec estimate.\
                1) 'cg': conjugate gradient, recommended for speed.
                2) 'sd': steepest descent, only for robustness.
            barea: if given a number, all units will be converted into
                units of Jy/beam for a choosen beam. Otherwise the units
                of images are always in Jy/px.
            map_conv: stop criterium for RESOLVE in map reconstruction.
            pspec_conv: stop criterium for RESOLVE in pspec reconstruction.
            save: If not None, save all iterations to disk using the given
                base name.
            callback: If given integer n, save every nth sub-iteration of \
                intermediate optimization routines.
            plot: Interactively plot diagnostical plots. Only advised for testing.
            simulate: Whether to simulate a signal or not.
            reffreq: reference-frequency, only needed for wideband mode.
            use_parset: whether to use the parameter-set file for parameter
                parsing.
            viscol: from which MS-column to read the visibility data.
        
        kwargs:
            Set numerical or simulational parameters. All are set to tested and \
            robust default values. For details see internal functions numparameters
            and simparameters.
            
        Returns:
            m: The MAP solution
            p: The power spectrum.
            u: Signal uncertainty (only if wanted).
    """


    # Turn off nifty warnings to avoid "infinite value" warnings
    about.warnings.off()
    
    # Make sure to not interactively plot. Produces a mess.
    pl.ioff()  
    
    # Define simulation parameter class if needed
    if params.simulating:
        simparams = sim.simparameters(params)
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
        '/u_reconstructions') and params.algorithm == 'ln-map_u':
            os.makedirs('resolve_output_' + str(params.save)+\
            '/u_reconstructions')  
    if not os.path.exists('resolve_output_' + str(params.save)+\
        '/mu_reconstructions') and params.algorithm == 'ln-map_u':
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
    if (params.init_type_s == 'fastResolve' or params.algorithm == 'fastResolve'):
       if not os.path.exists('resolve_output_' + str(params.save)+\
        '/fastresolve'):
            os.makedirs('resolve_output_' + str(params.save)+\
            '/fastresolve') 

    # Set up message logger            
    logfile = 'resolve_output_'+str(params.save)+'/general/' + params.save + \
        '.log'
    logger = M.Messenger(verbosity=params.verbosity, add_timestamp=False, \
        logfile=logfile)

    # Basic start messages, parameter log
    logger.header1('\nStarting RESOLVE package.')
    
    logger.message('\nThe choosen main parameters are:\n')
    for par in vars(params).items():
        logger.message(str(par[0])+' = '+str(par[1]),verb_level=2)
    logger.message('\nAll further default parameters are:\n')
    for par in vars(numparams).items():
        logger.message(str(par[0])+' = '+str(par[1]),verb_level=2)
    
    # Data setup
    if params.simulating:
        d, N, R, di, d_space, s_space, expI, n, sim_powspec = sim.simulate(params, numparams,simparams, \
            logger)
        np.save('resolve_output_' + str(params.save)+'/general/data',d.val)
        
    else:
        d, N, R, di, d_space, s_space = datasetup(params, logger) 
    
    if not params.init_type_s == 'fr_internal':
        # Standard Starting guesses setup
        if ((params.algorithm == ('ln-map') or params.algorithm == ('wf')) and (params.freq != 'wideband')):
            m_s, pspec, params, k_space = starting_guess_setup(params, logger, s_space, d_space)
            
        elif ((params.algorithm == 'ln-map_u') and (params.freq != 'wideband')):
            m_s, pspec, m_u, params, k_space = starting_guess_setup(params, logger, s_space, d_space)
    
        elif (params.algorithm == 'ln-map') and (params.freq == 'wideband'):
            m_s, pspec, m_a, pspec_a, params, k_space = starting_guess_setup(params, logger, s_space, d_space)
    
        elif (params.algorithm == 'ln-map_u') and (params.freq == 'wideband'):
            m_s, pspec, m_u, m_a, pspec_a, params, k_space = starting_guess_setup(params, logger, s_space, d_space)
            
    # Starting guess setup    
    # Check whether to do FastResolve for the starting guess, only for ln-map
    if params.algorithm == 'prefastResolve':
        if params.init_type_s == 'fr_internal':
            m_s = 'fr_internal'
        if params.init_type_p == 'fr_internal':
            pspec = 'fr_internal'
        m_s, pspec, k_space = ra.fastresolve(R, d, s_space, 'resolve_output_'+params.save+'/fastresolve/', noise_update=params.noise_update, noise_est=params.noise_est, msg=m_s, psg=pspec, point=500)
              
    # Begin: Start Filter *****************************************************
            
    if params.stokes != 'I':
        logger.failure('Pol-RESOLVE not yet implemented.')
        raise NotImplementedError('Pol-RESOLVE not yet implemented.')
        
    if params.algorithm == 'onlyfastResolve':
       
        logger.header2('\nStarting only fastRESOLVE reconstruction.')
        t1 = time()
        m_s, p_I, k_space = ra.fastresolve(R, d, s_space, 'resolve_output_'+params.save+'/fastresolve/', point=1000)
        t2 = time()
        logger.success("Completed algorithm.")
        logger.message("Time to complete: " + str((t2 - t1) / 3600.) + ' hours.')
        
    if params.algorithm == 'ln-map':

        logger.header2('\nStarting standard RESOLVE reconstruction.')
        
        if params.uncertainty == 'only':
            #single-band uncertainty map
            t1 = time()
            mapfilter_I(d, m_s, pspec, N, R, logger, k_space,\
                params, numparams)
            t2 = time()
            logger.success("Completed uncertainty map calculation.")
            logger.message("Time to complete: " + str((t2 - t1) / 3600.) + ' hours.')
            sys.exit(0)
            
        

        if params.freq != 'wideband':

            #single-band s-Filter
            t1 = time()           
            m_s, p_I = mapfilter_I(d, m_s, pspec, N, R, logger, k_space,\
                params, numparams)      
            t2 = time()
            
        else:
            logger.header2('Enabling RESOLVE wideband mode.')
            logger.warn('This mode is still experimental. Save files might '\
                + 'have misleading names.')
 
            if params.uncertainty_a == 'only':
                #single-band uncertainty map
                t1 = time()
                mapfilter_a(d, m_a, pspec_a, N, R, logger, \
                k_space, params, numparams, m_s, 0)
                t2 = time()
                logger.success("Completed alpha uncertainty map calculation.")
                logger.message("Time to complete: " + str((t2 - t1) / 3600.)\
                    + ' hours.')
                sys.exit(0)           
            
            #wide-band s/alpha-Filter
            t1 = time()
            wideband_git = 0
            while(wideband_git < numparams.wb_globiter):

                #s-Filter                                                                                                                                                                              
                m_s, p_I, = mapfilter_I(d, m_s, pspec, N, R, logger, \
                k_space, params, numparams, m_a, wideband_git)

                #a-Filter                                                                                                                                                                              
                m_a, p_a, = mapfilter_a(d, m_a, pspec_a, N, R, logger, \
                k_space, params, numparams, m_s, wideband_git)


                wideband_git += 1

            t2 = time()
            
    elif params.algorithm == 'ln-map_u':
        
        logger.header1('Starting Point-RESOLVE reconstruction.')
        
        if params.uncertainty == 'only':
            #single-band uncertainty map
            t1 = time()
            mapfilter_I_u(d, m_s, m_u, pspec, N, R, logger, k_space,\
                params, numparams)
            t2 = time()
            logger.success("Completed uncertainty map calculation.")
            logger.message("Time to complete: " + str((t2 - t1) / 3600.) + ' hours.')
            sys.exit(0)

        if params.freq != 'wideband':        
            t1 = time()                  
            
            m_s,m_u, p_I = mapfilter_I_u(d, m_s, m_u, pspec, N, R, logger, k_space, \
                params, numparams)                      
            t2 = time()
            
        else:
            logger.header2('Enabling RESOLVE wideband mode.')
            logger.warn('This mode is still experimental. Save files might '\
                + 'have misleading names.')
            
            if params.uncertainty_a == 'only':
                #single-band uncertainty map
                t1 = time()
                mapfilter_a(d, m_a, pspec_a, N, R, logger, rho0,\
                k_space, params, numparams, utils.log(exp(m_s)+exp(m_u)),0)
                t2 = time()
                logger.success("Completed alpha uncertainty map calculation.")
                logger.message("Time to complete: " + str((t2 - t1) / 3600.)\
                    + ' hours.')
                sys.exit(0)
                
            #wide-band su/alpha-Filter
            t1 = time()
            wideband_git = 0
            while(wideband_git < numparams.wb_globiter):

                #su-Filter                                                                                                                                                                              
                m_s,m_u, p_I = mapfilter_I_u(d, m_s,m_u, pspec, N, R, logger, k_space, \
                    params, numparams, m_a, wideband_git)      

                #a-Filter                                                                                                                                                                              
                m_a, p_a, = mapfilter_a(d, m_a, pspec_a, N, R, logger,\
                k_space, params, numparams, field(s_space,val=log(exp(m_s)+exp(m_u))), wideband_git)


                wideband_git += 1

            t2 = time()            
        
    elif params.algorithm == 'wf':
        
        logger.header2('Starting Wiener Filter reconstruction.')
        
        if params.uncertainty == 'only':
            #single-band uncertainty map
            t1 = time()
            wienerfilter(d, m_s, pspec, N, R, logger, k_space,\
                params, numparams)
            t2 = time()
            logger.success("Completed uncertainty map calculation.")
            logger.message("Time to complete: " + str((t2 - t1) / 3600.) + ' hours.')
            sys.exit(0)

        if params.freq is not 'wideband':
            #single-band s-Filter
            t1 = time()           
            m_s, p_I = wienerfilter(d, m_s, pspec, N, R, logger, k_space,\
                params, numparams)      
            t2 = time()
            
        else:
            logger.header2('Enabling RESOLVE wideband mode.')
            logger.warn('This mode is still experimental. Save files might '\
                + 'have misleading names.')

            if params.uncertainty_a == 'only':
                #single-band uncertainty map
                t1 = time()
                mapfilter_a(d, m_a, pspec_a, N, R, logger, \
                k_space, params, numparams, m_s, 0)
                t2 = time()
                logger.success("Completed alpha uncertainty map calculation.")
                logger.message("Time to complete: " + str((t2 - t1) / 3600.)\
                    + ' hours.')
                sys.exit(0)           
            
            #wide-band s/alpha-Filter
            t1 = time()
            wideband_git = 0
            while(wideband_git < numparams.wb_globiter):

                #s-Filter                                                                                                                                                                              
                m_s, p_I, = wienerfilter(d, m_s, pspec, N, R, logger, \
                k_space, params, numparams, m_a, wideband_git)

                #a-Filter                                                                                                                                                                              
                m_a, p_a, = mapfilter_a(d, m_a, pspec_a, N, R, logger, \
                k_space, params, numparams, log(m_s), wideband_git)


                wideband_git += 1

            t2 = time()
            
        
    elif params.algorithm =='gibbsenergy':
        
        logger.failure('Gibbs energy filter not yet implemented')
        raise NotImplementedError
        
    elif params.algorithm == 'sampling':
        
        logger.failure('Sampling algorithm not yet implemented')
        raise NotImplementedError
        

    logger.success("Completed algorithm.")
    logger.message("Time to complete: " + str((t2 - t1) / 3600.) + ' hours.')

    # Begin: Some plotting stuff **********************************************

    if params.save:
        
        utils.save_results(exp(m_s.val),"exp(Solution m)",\
            'resolve_output_' + str(params.save) + '/m_reconstructions/' + \
            params.save + "_expmfinal", rho0 = params.rho0)
        write_output_to_fits(np.transpose(exp(m_s.val)*params.rho0),params, notifier='final',mode='I')
        
        if params.freq == 'wideband':
            utils.save_results(m_a.val,"Solution a",\
                'resolve_output_' + str(params.save) + '/m_reconstructions/' +\
                params.save + "_mafinal", rho0 = params.rho0)
            write_output_to_fits(np.transpose(m_a.val),params, notifier ='final', mode='a')
        
        utils.save_results(k_space.get_power_indices()[0],"final power spectrum",\
                     'resolve_output_' + str(params.save) + \
                     '/p_reconstructions/' + params.save + "_powfinal", \
                     value2 = p_I, log='loglog')
                     
        if params.algorithm == 'ln-map_u':     
            utils.save_results(exp(m_u.val),"exp(Solution u)",\
                'resolve_output_' + str(params.save) + '/u_reconstructions/' + \
                params.save + "_expufinal", rho0 = params.rho0)
            write_output_to_fits(np.transpose(exp(m_u.val)*params.rho0),params, notifier='final',mode='I_u')
       
        if params.simulating:
            pl.figure()
            pl.loglog(k_space.get_power_indices()[0], p_I, label="final")
            pl.loglog(k_space.get_power_indices()[0],sim_powspec, label="simulated") 
            #pl.loglog(k_space.get_power_indices()[0],sim_powspec, label="simulated")
            pl.title("Compare final and simulated power spectrum")
            pl.legend()
            pl.savefig("resolve_output_" + str(params.save) +"/p_reconstructions/"\
                + params.save + "_compare.png")
            pl.close()  

    # ^^^ End: Some plotting stuff *****************************************^^^


#------------------------------------------------------------------------------


def datasetup(params, logger):
    """
        
    Data setup, feasible for the nifty framework.
        
    """
    
    logger.header2("\nRunning data setup.")
    
    #Somewhat inelegant solution, but WSclean needs its own I/O 
    if params.ftmode == 'wsclean':
        
        wscleanpars = w.ImagingParameters()
        wscleanpars.msPath = str(params.ms)
        wscleanpars.imageWidth = int(params.imsize)
        wscleanpars.imageHeight = int(params.imsize)
        wscleanpars.pixelScaleX = str(params.cellsize)+'rad'
        wscleanpars.pixelScaleY = str(params.cellsize)+'rad'
        wscleanpars.extraParameters = '-weight natural -nwlayers 1 -j 4 -channelrange 0 1'
      
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
            params.summary = np.load(params.ms+'_summary.npy')
        except IOError:
            logger.warn('No numpy MS header file found. FITS output will'\
                +' be deactivated.')
            params.summary = None

    #Non-WSclean (i.e. standard) loading routines
    else:
    
        vis, sigma, u, v, flags, freqs, nchan, nspw, nvis, params.summary = \
            utils.load_numpy_data(params.ms, logger)
    
    # definition of wideband data operators
    if params.freq == 'wideband':
        
        # wideband data and noise settings. Inelegant if not statement needed
        # because wsclean routines don't explicitly read out these things
        if not params.ftmode == 'wsclean': 
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
            
            variance = np.ones(np.shape(sigma))*np.mean(np.abs(vis*vis))/(1.+numparams.SNR_assumed)
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
        if params.ftmode == 'gfft':
            R = r.response_mfs(s_space, d_space, \
                               u,v,A,nspw,nchan,nvis,freqs,params.reffreq[0],\
                               params.reffreq[1],params.ftmode)
        else:
            logger.failure('For wideband mode only gfft support is available.')
        
        d = field(d_space, val=np.array(vis).flatten())

        N = N_operator(domain=d_space,imp=True,para=[variance])

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
        
    # definition of single band data operators
    else:
        
        # data and noise settings. Inelegant if not statement needed
        # because wsclean routines don't explicitly read out these things
        if not params.ftmode == 'wsclean':
            sspw,schan = params.freq[0], params.freq[1]
            vis = vis[sspw][schan]
            sigma = sigma[sspw]
            u = u[sspw][schan]
            v = v[sspw][schan] 
            flags = flags[sspw][schan]
            
            
            # cut away flagged data
            vis = np.delete(vis,np.where(flags)==0)
            u = np.delete(u,np.where(flags)==0)
            v = np.delete(v,np.where(flags)==0)
            sigma = np.delete(sigma,np.where(flags)==0)
            

        # Dumb simple estimate can be done now after reading in the data.
        # No time information needed.
        if params.noise_est == 'simple':
            
            variance = np.var(np.array(vis))*np.ones(np.shape(np.array(vis)))\
                .flatten()
        
        elif params.noise_est == 'SNR_assumed':
            
            variance = np.ones(np.shape(sigma))*np.mean(np.abs(vis*vis))/(1.+numparams.SNR_assumed)
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

        N = N_operator(domain=d_space,imp=True,para=[variance])
        
        # dirty image from CASA or Resolve for comparison
        if params.ftmode == 'wsclean':
            di = R.adjoint_times(d)
        else:
            R.adjointfactor = 1
            di = R.adjoint_times(d) * s_space.vol[0] * s_space.vol[1]
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
    

def starting_guess_setup(params, logger, s_space, d_space):
    
    # Starting guesses for m_s
    
    if params.init_type_s == 'const':
        m_s = field(s_space, val = numparams.m_start)

    elif params.init_type_s == 'dirty':
        m_s = field(s_space, target=s_space.get_codomain(), val=di)   
        
    else:
        if params.sglogim:             
            m_s = field(s_space, target=s_space.get_codomain(), \
                val=np.load(params.init_type_s))
        else:
            expm_s_val = np.abs(np.load(params.init_type_s))
            expm_s_val[expm_s_val==0] = 1e-12
            m_s = field(s_space, target=s_space.get_codomain(), \
                val=log(expm_s_val))
  
    # Optional starting guesses for m_u
            
    if params.algorithm == 'ln-map_u':
        
        
        if params.init_type_u == 'const':
            m_u = field(s_space, val = numparams.m_u_start)
        
        elif params.init_type_u == 'dirty':
            m_u = field(s_space, target=s_space.get_codomain(), val=di)   
            
        else:
            if params.sglogim_u:              
                m_u = field(s_space, target=s_space.get_codomain(),\
                    val=params.init_type_u)
            else:
                start_logu = np.abs(np.load(params.init_type_u))
                if np.any(start_logu) == 0:
                   start_logu[start_logu==0] = 1e-15  
                m_u = field(s_space, target=s_space.get_codomain(), \
                val=utils.log(start_logu))
                
    if params.rho0 == 'from_sg':
        
        if params.algorithm == 'ln-map_u':
            params.rho0 = np.mean(exp(m_s.val[np.where(exp(m_s).val>=np.max(exp(m_s).val)\
                / 10)] + m_u.val[np.where(exp(m_u).val>=np.max(exp(m_u).val)/ 10)]))
        else:
             params.rho0 = np.mean(exp(m_s.val[np.where(exp(m_s).val>=np.max(exp(m_s).val)\
                / 10)]))
        logger.message('rho0 was calculated as: ' + str(params.rho0)) 
        
    if not params.rho0 == 1.:
        
        m_s -= log(params.rho0)
        if params.algorithm == 'ln-map_u':
            m_u -= log(params.rho0)

    np.save('resolve_output_' + str(params.save)+'/general/rho0',params.rho0)
    if params.rho0 < 0:
        logger.warn('Monopole level of starting guess negative. Probably due \
            to too many imaging artifcts in userimage')
        
            
    # Starting guesses for pspec 

    # Basic k-space
    k_space = s_space.get_codomain()
    #Adapts the k-space properties if binning is activated.
    #if numparams.bins:
    #    k_space.set_power_indices(log=True, nbins=numparams.bins)
    # k-space prperties    
    kindex,rho_k,pindex,pundex = k_space.get_power_indices(log=True, nbins=numparams.bins)
    # Simple k^2 power spectrum with p0 from numpars and a fixed monopole from
    print len(kindex)
    # the m starting guess
    if params.init_type_p == 'k^2_mon':
        pspec = np.array((1+kindex)**-2 * numparams.p0)
        pspec_m_s = m_s.power(pindex=pindex, kindex=kindex, rho=rho_k)
        #see notes, use average power in dirty map to constrain monopole
        pspec[0] = (np.prod(k_space.vol)**(-2) * utils.log(\
            np.sqrt(pspec_m_s[0]) *  np.prod(k_space.vol))**2) / 2.
    # default simple k^2 spectrum with free monopole
    elif params.init_type_p == 'k^2':
        pspec = np.array((1+kindex)**-2 * numparams.p0)    
    # constant power spectrum guess 
    elif params.init_type_p == 'constant':
        pspec = np.ones(len(kindex)) * numparams.p0
    # power spectrum from last iteration 
    else:
        try:
            logger.message('Using pspec from file '+params.init_type_p)
            pspec = np.load(params.init_type_p)
        except IOError:
            logger.failure('No pspec file found. Set SG to k^2.')
            pspec = np.array((1+kindex)**-2 * numparams.p0)
        
    # check validity of starting pspec guesses
    if np.any(pspec) == 0:
        pspec[pspec==0] = 1e-25

    # Wideband mode starting guesses
    if params.freq == 'wideband':
        
        # Starting guesses for m_a

        if params.init_type_a == 'const':
            m_a = field(s_space, val = numparams.m_a_start)  
            
        else:
            m_a = field(s_space, target=s_space.get_codomain(), \
                val=np.load(params.init_type_a))        
        
        # Spectral index pspec starting guesses
        
        # default simple k^2 spectrum with free monopole
        if params.init_type_a_p == 'k^2':
            pspec_a = np.array((1+kindex)**-2 * numparams.p0_a)    
        # constant power spectrum guess 
        elif params.init_type_a_p == 'constant':
            pspec_a = numparams.p0_a
        # power spectrum from last iteration 
        else:
            logger.message('using last p-iteration from previous run.')
            pspec_a = np.load(params.init_type_a_p)

        if np.any(pspec_a) == 0:
            pspec_a[pspec_a==0] = 1e-25
 
    # diagnostic plot of m starting guess
    utils.save_results(exp(m_s.val),"TI exp(Starting guess)",\
        'resolve_output_' + str(params.save) + '/m_reconstructions/' +\
        params.save + "_expm0", rho0 = params.rho0)
    write_output_to_fits(np.transpose(exp(m_s.val)*params.rho0),params, \
        notifier='0', mode='I')
    if params.freq == 'wideband':
        utils.save_results(m_a.val,"Alpha Starting guess",\
            'resolve_output_' + str(params.save) + '/m_reconstructions/' +\
            params.save + "_ma0", rho0 = params.rho0) 
        write_output_to_fits(np.transpose(m_s.val),params, notifier='0', \
               mode='a') 
    if params.algorithm == 'ln-map_u':
        utils.save_results(exp(m_u.val),"TI exp(Starting guess)",\
            'resolve_output_' + str(params.save) + '/u_reconstructions/' +\
            params.save + "_expu0", rho0 = params.rho0)    
        write_output_to_fits(np.transpose(exp(m_u.val)*params.rho0),params, \
            notifier='0', mode='I_u')       
                
        utils.save_results(exp(m_s.val)+exp(m_u.val),"TI exp(Starting guess)",\
            'resolve_output_' + str(params.save) + '/mu_reconstructions/'+\
            params.save + "_expmu0", rho0 = params.rho0)    
        write_output_to_fits(np.transpose(exp(m_s.val)+exp(m_u.val)*params.rho0),\
            params, notifier='0', mode='I_mu')         
    
    if ((params.algorithm == 'ln-map' or params.algorithm == 'wf') and (params.freq != 'wideband')):
        return m_s, pspec, params, k_space
    
    elif ((params.algorithm == 'ln-map_u') and (params.freq != 'wideband')):
        return m_s, pspec, m_u, params, k_space

    elif ((params.algorithm == 'ln-map') and (params.freq == 'wideband')):
        return m_s, pspec, m_a, pspec_a, params, k_space

    elif ((params.algorithm == 'ln-map_u') and (params.freq == 'wideband')):
        return m_s, pspec, m_u, m_a, pspec_a, params, k_space


def mapfilter_I(d, m, pspec, N, R, logger, k_space, params, numparams,\
    *args):
    """
    Main standard MAP-filter iteration cycle routine.
    """

    if params.freq == 'wideband':
        logger.header1("\nBegin total intensity wideband RESOLVE iteration cycle.")  
    else:
        logger.header1("\nBegin total intensity standard RESOLVE iteration cycle.")
    
    s_space = m.domain
    kindex = k_space.get_power_indices()[0]
    
    if params.freq == 'wideband':
        aconst = args[0]
        wideband_git = args[1]
         
    # Sets the alpha prior parameter for all modes
    if numparams.alpha_prior:
        alpha = numparams.alpha_prior
    else:
        alpha = np.ones(np.shape(kindex))
        
    # Defines important operators
    S = power_operator(k_space, spec=pspec, bare=True)
    if params.freq == 'wideband':
        M = MI_operator(domain=s_space, sym=True, imp=True, para=[N, R, aconst])
        j = R.adjoint_times(N.inverse_times(d), a = aconst)
    else:    
        M = M_operator(domain=s_space, sym=True, imp=True, para=[N, R])
        j = R.adjoint_times(N.inverse_times(d))
    D = D_operator(domain=s_space, sym=True, imp=True, para=[S, M, m, j, \
        numparams.M0_start, params.rho0, params, numparams])

    
    # diagnostic plots

    if params.freq == 'wideband':
        utils.save_results(j,"j",'resolve_output_' + str(params.save) +\
            '/general/' + params.save + \
            str(wideband_git) + '_j',rho0 = params.rho0)
    else:
        utils.save_results(j,"j",'resolve_output_' + str(params.save) +\
            '/general/' + params.save + '_j',rho0 = params.rho0)

    # iteration parameters
    convergence = 0
    git = 1
    plist = [pspec]
    mlist = [m]
    call = utils.callbackclass(params.save,numparams.map_algo,params.callback) 

    while git <= numparams.global_iter:
        """
        Global filter loop.
        """
        logger.header2("Starting global iteration #" + str(git) + "\n")

        # update power operator after each iteration step
        S.set_power(newspec=pspec,bare=True)
          
        # uodate operator and optimizer arguments  
        args = (j, S, M, params.rho0)
        D.para = [S, M, m, j, numparams.M0_start, params.rho0, params,\
            numparams]

        if params.uncertainty=='only':
            logger.haeder2('Only calculating uncertainty map as requested.')
            D_hat = D.hat(domain=s_space,\
                ncpu=numparams.ncpu,nrun=numparams.nrun)
            utils.save_results(D_hat.val,"relative uncertainty", \
                'resolve_output_' + str(params.save) +\
                "/D_reconstructions/" + params.save + "_D")
            return None

        # run minimizer
        logger.header2("Computing the MAP estimate.\n")

        mold = m
       
        if numparams.map_algo == 'sd':
            en = energy(args)
            minimize = nt.steepest_descent(en.egg,spam=call.callbackfunc,note=True)
            m = minimize(x0=m, alpha=numparams.map_alpha, \
                tol=numparams.map_tol, clevel=numparams.map_clevel, \
                limii=numparams.map_iter)[0]
        elif numparams.map_algo == 'lbfgs':
            logger.warn('lbfgs algorithm implemented from scipy, but'\
                + ' experimental.')
            en = energy(args)
            m = utils.Energy_min(m,en,params,numparams,limii=numparams.map_iter)
           
        # save iteration results
        mlist.append(m)
        if params.freq == 'wideband':
            utils.save_results(exp(m.val), "map, iter #" + str(git), \
                'resolve_output_' + str(params.save) +\
                '/m_reconstructions/' + params.save + "_expm" +\
                str(wideband_git) + "_" + str(git), rho0 = params.rho0)
            write_output_to_fits(np.transpose(exp(m.val)*params.rho0),params, \
                notifier = str(wideband_git) + "_" + str(git),mode = 'I')
        else:
            utils.save_results(exp(m.val), "map, iter #" + str(git), \
                'resolve_output_' + str(params.save) +\
                '/m_reconstructions/' + params.save + "_expm" + str(git), \
                rho0 = params.rho0)
            write_output_to_fits(np.transpose(exp(m.val)*params.rho0),params,\
                notifier = str(git), mode='I')
                

        # check whether to do ecf-like noise update
        if (params.noise_update and (git==1 or git%3==0)):          
            # Do a "poor-man's" extended critical filter step using residual
            logger.header2("Trying simple noise estimate without any D.")
            #newvar = ((d - R(exp(m)))**2).mean()
            REG_VAR = 0.9
            est_var = (R(exp(m)) - d).val
            est_var = np.abs(est_var)**2
            est_var = REG_VAR*est_var + (1-REG_VAR)*est_var.mean()
            logger.message('old variance iteration '+str(git-1)+':' + str(N.diag()))
            logger.message('new variance iteration '+str(git)+':' + str(newvar))
            np.save('resolve_output_' + str(params.save) + '/general/oldvar_'+str(git),N.diag())
            np.save('resolve_output_' + str(params.save) +'/general/newvar_'+str(git),newvar)
            np.save('resolve_output_' + str(params.save) +'/general/absdmean_'\
                +str(git),abs(d.val).mean())
            np.save('resolve_output_' + str(params.save) +'/general/absRmmean_'\
                +str(git),abs(R(exp(m)).val*R.target.num()).mean())
            N.para = [newvar*np.ones(np.shape(N.diag()))]
            M = M_operator(domain=s_space, sym=True, imp=True, para=[N, R])
            j = R.adjoint_times(N.inverse_times(d))

        # Check whether to do the pspec iteration
        if params.pspec:
            logger.header2("Computing the power spectrum.\n")

            # extra loop to take care of possible nans in PS calculation
            psloop = True
            M0 = numparams.M0_start
            while psloop:
            
                D.para = [S, M, m, j, M0, params.rho0, params, numparams]
            
                Sk = projection_operator(domain=k_space)
                if params.ftmode == 'wsclean':
                    #bare=True?
                    #right now WSclean is not compatible with parallel probing
                    #due to its internal parallelization
                    logger.message('Calculating Dhat for pspec reconstruction.')
                    D_hathat = D.hathat(domain=s_space.get_codomain(),loop=True)
                    logger.message('Success.')
                else:
                   #bare=True?
                    logger.message('Calculating Dhat for pspec reconstruction.')
                    D_hathat = D.hathat(domain=s_space.get_codomain(),\
                        ncpu=numparams.ncpu,nrun=numparams.nrun)
                    logger.message('Success.') 

                pspec = infer_power(m,domain=Sk.domain,Sk=Sk,D=D_hathat,\
                    q=1E-42,alpha=alpha,perception=(1,0),smoothness=True,var=\
                    numparams.smoothing, bare=True)

                if np.any(pspec == False):
                    print 'D not positive definite, try increasing eta.'
                    if M0 == 0:
                        M0 += 0.1
                    M0 *= 1e6
                    D.para = [S, M, m, j, M0, params.rho0, params, numparams]
                else:
                    psloop = False
            
            logger.message("    Current M0:  " + str(D.para[4])+ '\n.')


        # check whether to do pspec saves
        if params.pspec:
            plist.append(pspec)

            if params.freq == 'wideband':
                utils.save_results(kindex,"ps, iter #" + str(git), \
                    'resolve_output_' + str(params.save) + \
                    "/p_reconstructions/" + params.save + "_p" + \
                    str(wideband_git) + "_" + str(git), value2=pspec,\
                    log='loglog')
            else:
                utils.save_results(kindex,"ps, iter #" + str(git), \
                    'resolve_output_' + str(params.save) +\
                    "/p_reconstructions/" + params.save + "_p" + str(git), \
                    value2=pspec,log='loglog')
            
            # powevol plot needs to be done in place
            pl.figure()
            for i in range(len(plist)):
                pl.loglog(kindex, plist[i], label="iter" + str(i))
            pl.title("Global iteration pspec progress")
            pl.legend()
            pl.savefig('resolve_output_' + str(params.save) + \
                "/p_reconstructions/" + params.save + "_powevol.png")
            pl.close()
        
        # convergence test in map reconstruction
        if np.max(np.abs(m - mold)) < params.map_conv:
            logger.message('Image converged.')
            convergence += 1
        
        # convergence test in power spectrum reconstruction 
        if np.max(np.abs(utils.log(pspec)/utils.log(S.get_power()))) < utils.log(1e-1):
            logger.message('Power spectrum converged.')
            convergence += 1
        
        #global convergence test
        if convergence >= numparams.final_convlevel:
            logger.message('Global convergence achieved at iteration' + \
                str(git) + '.')
            if params.uncertainty=='supp':
                logger.message('Calculating uncertainty map at end of'\
                    +'recpnstruction, as requested.')
                D_hat = D.hat(domain=s_space,\
                ncpu=numparams.ncpu,nrun=numparams.nrun)
                utils.save_results(D_hat.val,"relative uncertainty", \
                    'resolve_output_' + str(params.save) +\
                    "/D_reconstructions/" + params.save + "_D")
                
            return m, pspec

        git += 1

    return m, pspec
    
    
def mapfilter_I_u(d, m,u, pspec, N, R, logger, k_space, params, numparams,\
    *args):
    """
    """
    scipyminimizer = ('TNC','COBYLA','SLSQP','dogleg','trust-ncg',\
        'CG','BFGS', 'L-BFGS-B','Nelder-Mead','Powell','Newton-CG')
    
    if params.freq == 'wideband':
        logger.header1("Begin total intensity wideband Point-RESOLVE iteration cycle.")
    else:
        logger.header1("Begin total intensity Point-RESOLVE iteration cycle.")  
    
    s_space = m.domain
    kindex = k_space.get_power_indices()[0]
                 
    if params.freq == 'wideband':
        aconst = args[0]
        wideband_git = args[1]
        
    # Sets the alpha prior parameter for all modes
    if numparams.alpha_prior:
        alpha = numparams.alpha_prior
    else:
        alpha = np.ones(np.shape(kindex))

    # Defines important operators    
    S = power_operator(k_space, spec=pspec, bare=True)
    if params.freq == 'wideband':
        M = MI_operator(domain=s_space, sym=True, imp=True, para=[N, R, aconst])
        j = R.adjoint_times(N.inverse_times(d), a = aconst)
    else:    
        M = M_operator(domain=s_space, sym=True, imp=True, para=[N, R])
        j = R.adjoint_times(N.inverse_times(d))
    D = Dmu_operator(domain=s_space, sym=True, imp=True, para=[S, M, m, j, \
        numparams.M0_start, params.rho0, u, params, numparams])
   
    #diagnostic plots    
    if params.freq == 'wideband':
        utils.save_results(j,"j",'resolve_output_' + str(params.save) +\
        '/general/' + params.save + \
            str(wideband_git) + '_j',rho0 = params.rho0)
    else:
        utils.save_results(j,"j",'resolve_output_' + str(params.save) +\
            '/general/' + params.save + '_j',rho0 = params.rho0)
    
    # iteration parameters
    convergence = 0
    git = 1
    algo_run = 0
    plist = [pspec]
    mlist = [m]    
    call = utils.callbackclass(params.save,numparams.map_algo,params.callback)     

    while git <= numparams.global_iter:
        """
        Global filter loop.
        """

        logger.header2("Starting global up iteration #" + str(git) + "\n")

        # update power operator after each iteration step
        S.set_power(newspec=pspec,bare=True)
          
        # uodate operator and optimizer arguments  
        D.para = [S, M, m, j, numparams.M0_start, params.rho0,u, params, numparams]

        # Check whether to only calculte an uncertainty map
        if params.uncertainty=='only':
            logger.message('Only calculating uncertainty map as requested.')
            D_hat = D.hat(domain=s_space,\
                ncpu=numparams.ncpu,nrun=numparams.nrun)
            utils.save_results(D_hat.val,"relative uncertainty", \
                'resolve_output_' + str(params.save) +\
                "/D_reconstructions/" + params.save + "_D")
            return None


        #run nifty minimizer steepest descent class

        mold = m
        uold = u
        args = (j, S, M, params.rho0, numparams.beta, numparams.eta,m,u)
        en = energy_mu(args)
        if numparams.algo_liste[algo_run] == 'sd_u':

            logger.header2("Computing the u-MAP estimate.\n")
            minimize = nt.steepest_descent(en.egg_u,spam=call.callbackfunc_u,\
                note=True)
            u = minimize(x0=u, alpha=numparams.map_alpha_u, \
                tol=numparams.map_tol_u, clevel=numparams.map_clevel_u, \
                limii=numparams.map_iter_u)[0]
                
            utils.save_u(u,git,params)          
            utils.save_mu(m,u,git,params)        
            # convergence test in s/u-map reconstruction 
            if np.max(np.abs(u - uold)) < params.map_conv:
                logger.message('Compact image converged.')
                convergence += 1     
                
        elif numparams.algo_liste[algo_run] == 'sd_m':
               
            logger.header2("Computing the m-MAP estimate.\n")   
            minimize = nt.steepest_descent(en.egg_s,spam=call.callbackfunc_m,\
                note=True)
            m = minimize(x0=m, alpha=numparams.map_alpha, \
                tol=numparams.map_tol, clevel=numparams.map_clevel, \
                limii=numparams.map_iter)[0]   
                
            if params.freq == 'wideband':
                utils.save_m(m,git,params,wideband_git)                       
            else:
                utils.save_m(m,git,params)  
            mlist.append(m)
            utils.save_mu(m,u,git,params)  
            # convergence test in s/u-map reconstruction             
            if np.max(np.abs(m - mold)) < params.map_conv:
                logger.message('Extended image converged.')
                convergence += 1                    

       
        elif numparams.algo_liste[algo_run] in scipyminimizer:
            logger.warn(numparams.algo_liste[algo_run]+' algorithm implemented from scipy, but'\
                + ' experimental.')
            m,u = utils.Energy_min(m,en,params,numparams,numparams.algo_liste[algo_run],numparams.map_iter, x1 = u)

            if params.freq == 'wideband':
                utils.save_m(m,git,params,wideband_git)                       
            else:
                utils.save_m(m,git,params)  
            mlist.append(m)
            utils.save_u(u,git,params)
            utils.save_mu(m,u,git,params)  
        # convergence test in s/u-map reconstruction
            if np.max(np.abs(m - mold)) < params.map_conv and np.max(np.abs(u - uold)) < params.map_conv:
                logger.message('Image converged.')
                convergence += 1            
                
        #temporary
        elif numparams.algo_liste[algo_run] == 'test_u':
             u = utils.Energy_min(u,en,params,numparams,'L-BFGS-B',numparams.map_iter_u, x1 = None,pure='pure_u')
             utils.save_u(u,git,params)          
             utils.save_mu(m,u,git,params)   
             
        elif numparams.algo_liste[algo_run] == 'test_m':
             m = utils.Energy_min(m,en,params,numparams,'L-BFGS-B',numparams.map_iter, x1 = None,pure='pure_m')
             utils.save_m(m,git,params)          
             utils.save_mu(m,u,git,params)              
        #temporary
            
        # pure compact field reconstruction
        elif numparams.algo_liste[algo_run] == 'pure_points':
            args = (j, S, M, params.rho0, numparams.beta, numparams.eta)
            en = energy_u(args)  
            minimize = nt.steepest_descent(en.egg_u,spam=call.callbackfunc_u,note=True)
            u = minimize(x0=u, alpha=numparams.map_alpha_u, \
               tol=numparams.map_tol_u, clevel=numparams.map_clevel_u, \
               limii=numparams.map_iter_u)[0]
            utils.save_u(u,git,params)    
            # convergence test in s/u-map reconstruction 
            if np.max(np.abs(u - uold)) < params.map_conv:
                logger.message('Compact image converged.')
                convergence += 1     

        # check whether to do ecf-like noise update              
        elif numparams.algo_liste[algo_run] == 'update_noise':        
        #if params.noise_update:
            # Do a "poor-man's" extended critical filter step using residual
            logger.header2("Trying simple noise estimate without any D.")
            newvar = (np.abs((d.val - R(exp(m)+exp(u)))**2).mean())
            logger.message('old variance iteration '+str(git-1)+':' + str(N.diag()))
            logger.message('new variance iteration '+str(git)+':' + str(newvar))
            np.save('resolve_output_' + str(params.save) + 'oldvar_'+str(git),N.diag())
            np.save('resolve_output_' + str(params.save) +'newvar_'+str(git),newvar)
            np.save('resolve_output_' + str(params.save) +'absdmean_'\
                +str(git),abs(d.val).mean())
            np.save('resolve_output_' + str(params.save) +'absRmmean_'\
                +str(git),abs(R(m).val*R.target.num()).mean())
            N.para = [newvar*np.ones(np.shape(N.diag()))]
        
        # Check whether to do the pspec iteration
        elif numparams.algo_liste[algo_run]=='ps_rec':
            logger.header2("Computing the power spectrum.\n")

            #extra loop to take care of possible nans in PS calculation
            psloop = True
            M0 = numparams.M0_start
                  
            while psloop:
            
                D.para = [S, M, m, j, M0, params.rho0,u, params, numparams]
            
                Sk = projection_operator(domain=k_space)
                if params.ftmode == 'wsclean':
                    #bare=True?
                    #right now WSclean is not compatible with parallel probing
                    #due to its internal parallelization
                    logger.message('Calculating Dhat for pspec reconstruction.')
                    D_hathat = D.hathat(domain=s_space.get_codomain(),loop=True)
                    logger.message('Success.')
                else:
                   #bare=True?
                    logger.message('Calculating Dhat for pspec reconstruction.')
                    D_hathat = D.hathat(domain=s_space.get_codomain(),\
                        ncpu=numparams.ncpu,nrun=numparams.nrun)
                    logger.message('Success.')

                pspec = infer_power(m,domain=Sk.domain,Sk=Sk,D=D_hathat,\
                    q=1E-42,alpha=alpha,perception=(1,0),smoothness=True,var=\
                    numparams.smoothing, bare=True)

                if np.any(pspec == False):
                    print 'D not positive definite, try increasing eta.'
                    if M0 == 0:
                        M0 += 0.1
                    M0 *= 1e6
                    D.para = [S, M, m, j, M0, params.rho0, params, numparams]
                else:
                    psloop = False
            
            logger.message("    Current M0:  " + str(D.para[4])+ '\n.')

        # check whether to do pspec saves
            if params.pspec:
                plist.append(pspec)
                utils.save_results(kindex,"ps, iter #" + str(git), \
                    'resolve_output_' + str(params.save) +\
                    "/p_reconstructions/" + params.save + "_p" + str(git)+"_up", \
                    value2=pspec,log='loglog')
            
            # powevol plot needs to be done in place
                pl.figure()
                for i in range(len(plist)):
                    pl.loglog(kindex, plist[i], label="iter" + str(i))
                pl.title("Global iteration pspec progress")
                pl.legend()
                pl.savefig("resolve_output_" + str(params.save) +"/p_reconstructions/"\
                     + params.save + "_up_powevol.png")
                pl.close()
        # convergence test in power spectrum reconstruction 
            if np.max(np.abs(utils.log(pspec)/utils.log(S.get_power()))) < utils.log(1e-1):
                logger.message('Power spectrum converged.')
                convergence += 1
        
        #global convergence test
        if convergence >= numparams.final_convlevel:
            logger.message('Global convergence achieved at iteration' + \
                str(git) + '.')
            if params.uncertainty=='supp':
                logger.message('Calculating uncertainty map as requested.')
                D_hat = D.hat(domain=s_space,\
                ncpu=numparams.ncpu,nrun=numparams.nrun)
                utils.save_results(D_hat.val,"relative uncertainty", \
                    'resolve_output_' + str(params.save) +\
                    "/D_reconstructions/" + params.save + "_D")
                
            return m,u, pspec

        if algo_run == len(numparams.algo_liste)-1:
           algo_run = 0
           git += 1
        else:
           algo_run += 1

    return m,u, pspec
    


def mapfilter_a(d, m, pspec, N, R, logger, k_space, params, numparams,\
    m_s, wideband_git):
    """
    """

    logger.header1("Begin spectral index wideband RESOLVE iteration cycle.")  
      
    s_space = m.domain
    kindex = k_space.get_power_indices()[0]
    
    mconst = m_s
         
    # Sets the alpha prior parameter for all modes
    if numparams.alpha_prior_a:
        alpha = numparams.alpha_prior_a
    else:
        alpha = np.ones(np.shape(kindex))
        
    # Defines important operators. M and j are defined implicitly in energy_a
    S = power_operator(k_space, spec=pspec, bare=True)
    D = Da_operator(domain=s_space, sym=True, imp=True, para=[S, R, N, m,\
            numparams.M0_start_a, mconst, d, params.rho0, params, numparams])
    

    # iteration parameters
    convergence = 0
    git = 1
    plist = [pspec]
    mlist = [m]
    call = utils.callbackclass(params.save,numparams.map_algo,params.callback) 

    while git <= numparams.global_iter_a:
        """
        Global filter loop.
        """
        logger.header2("Starting spectral index "+
            "subiteration #" + str(git) + "\n")

        # update power operator after each iteration step
        S.set_power(newspec=pspec,bare=True)
          
        # uodate operator and optimizer arguments  
        args = (d,S, N, R,mconst)
        D.para = [S, R, N, m, numparams.M0_start_a, params.rho0, mconst, d,\
            params, numparams]
        
        if params.uncertainty_a=='only':
            logger.message('Only calculating alpha uncertainty as requested.')
            D_hat = D.hat(domain=s_space,\
                ncpu=numparams.ncpu,nrun=numparams.nrun)
            utils.save_results(D_hat.val,"relative uncertainty", \
                'resolve_output_' + str(params.save) +\
                "/D_reconstructions/" + params.save + "_D")
            return None

        #run nifty minimizer steepest descent class
        logger.header2("Computing the spectral index MAP estimate.\n")

        mold = m
        
        if numparams.map_algo == 'sd':
            en = energy_a(args)
            minimize = nt.steepest_descent(en.egg,spam=call.callbackfunc,note=True)
            m = minimize(x0=m, alpha=numparams.map_alpha_a, \
                tol=numparams.map_tol_a, clevel=numparams.map_clevel_a, \
                limii=numparams.map_iter_a)[0]
        elif numparams.map_algo == 'lbfgs':
            logger.warn('lbfgs algorithm implemented from scipy, but'\
            + 'experimental.')
            m = utils.BFGS(m,j,S,M,rho0,params,limii=numparams.map_iter_a)
        
        mlist.append(m)   
        utils.save_results(m.val, "map, iter #" + str(git), \
            'resolve_output_' + str(params.save) +\
            "/m_reconstructions/" + params.save + "_ma" + \
            str(wideband_git) + "_" + str(git), rho0 = params.rho0)
        write_output_to_fits(np.transpose(m.val),params, \
        notifier = str(wideband_git) + "_" + str(git), mode='a')
        

        if params.pspec_a:
            logger.header2("Computing the power spectrum.\n")
            
    
            #extra loop to take care of possible nans in PS calculation
            psloop = True
            M0 = numparams.M0_start_a
            while psloop:
                
                D.para = [S, R, N, m, numparams.M0_start_a, params.rho0,\
                mconst, d, params, numparams]
                
                Sk = projection_operator(domain=k_space)
                if params.ftmode == 'wsclean':
                    #bare=True?
                    #right now WSclean is not compatible with parallel probing
                    #due to its internal parallelization
                    logger.message('Calculating Dhat for pspec reconstruction.')
                    D_hathat = D.hathat(domain=s_space.get_codomain(),loop=True)
                    logger.message('Success.')
                else:
                   #bare=True?
                    logger.message('Calculating Dhat for pspec reconstruction.')
                    D_hathat = D.hathat(domain=s_space.get_codomain(),\
                        ncpu=numparams.ncpu,nrun=numparams.nrun)
                    logger.message('Success.')

    
                pspec = infer_power(m,domain=Sk.domain,Sk=Sk,D=D_hathat,\
                    q=1E-42,alpha=alpha,perception=(1,0),smoothness=True,var=\
                    numparams.smoothing_a, bare=True)
    
                if np.any(pspec == False):
                    logger.message('D not positive definite, try increasing'\
                        +'eta.')
                    if M0 == 0:
                        M0 += 0.1
                    M0 *= 1e6
                    D.para = [S, M, m, j, M0, params.rho0, params, numparams]
                else:
                    psloop = False
                
            logger.message("    Current M0:  " + str(D.para[4])+ '\n.')


        if params.pspec_a:
            plist.append(pspec)
            utils.save_results(kindex,"ps, iter #" + str(git), \
                'resolve_output_' + str(params.save) +\
                "/p_reconstructions/" + params.save + "_pa" + \
                str(wideband_git) + "_" + str(git), value2=pspec,log='loglog')
            
            # powevol plot needs to be done in place
            pl.figure()
            for i in range(len(plist)):
                pl.loglog(kindex, plist[i], label="iter" + str(i))
            pl.title("Global iteration pspec progress")
            pl.legend()
            pl.savefig('resolve_output_' + str(params.save) +\
                "/p_reconstructions/" + params.save + "_powevol.png")
            pl.close()
        
        # convergence test in map reconstruction
        if np.max(np.abs(m - mold)) < params.map_conv:
            logger.message('Image converged.')
            convergence += 1
        
        # convergence test in power spectrum reconstruction
        if params.pspec_a:
            if np.max(np.abs(utils.log(pspec)/utils.log(S.get_power())))\
                < utils.log(1e-1):
                    logger.message('Power spectrum converged.')
                    convergence += 1
        
        #global convergence test
        if convergence >= numparams.final_convlevel:
            logger.message('Global convergence achieved at iteration' + \
                str(git) + '.')
            if params.uncertainty_a=='supp':
                logger.message('Calculating alpha uncertainty as requested.')
                D_hat = D.hat(domain=s_space,\
                ncpu=numparams.ncpu_a,nrun=numparams.nrun_a)
                utils.save_results(D_hat.val,"relative uncertainty", \
                    'resolve_output_' + str(params.save) +\
                    "/D_reconstructions/" + params.save + "_Da")
                
            return m, pspec

        git += 1

    return m, pspec

def wienerfilter(d, m, pspec, N, R, logger, k_space, params, numparams,\
    *args):
    """
    Main Wienerfilter iteration cycle routine.
    """

    if params.freq == 'wideband':
        logger.header1("Begin total intensity wideband Wiener Filter" \
            + "iteration cycle.")  
    else:
        logger.header1("Begin total intensity standard Wiener Filter" \
                    + "iteration cycle.")
    
    s_space = m.domain
    kindex = k_space.get_power_indices()[0]
                           
    # Sets the alpha prior parameter for all modes
    if numparams.alpha_prior:
        alpha = numparams.alpha_prior
    else:
        alpha = np.ones(np.shape(kindex))
    
    if params.freq == 'wideband':
        aconst = args[0]
        wideband_git = args[1]
        
    # Defines important operators
    S = power_operator(k_space, spec=pspec, bare=True)
    if params.freq == 'wideband':
        M = MI_operator(domain=s_space, sym=True, imp=True, para=[N, R, aconst])
        j = R.adjoint_times(N.inverse_times(d), a = aconst)
    else:    
        M = M_operator(domain=s_space, sym=True, imp=True, para=[N, R])
        j = R.adjoint_times(N.inverse_times(d))
    D = D_operator(domain=s_space, sym=True, imp=True, para=[S, M, m, j, \
        numparams.M0_start, params.rho0, params, numparams])

    
    # diagnostic plots

    if params.freq == 'wideband':
        utils.save_results(j,"j",'resolve_output_' + str(params.save) +\
            '/general/' + params.save + \
            str(wideband_git) + '_j',rho0 = params.rho0)
    else:
        utils.save_results(j,"j",'resolve_output_' + str(params.save) +\
            '/general/' + params.save + '_j',rho0 = params.rho0)

    # iteration parameters
    convergence = 0
    git = 1
    plist = [pspec]
    mlist = [m]
    

    while git <= numparams.global_iter:
        """
        Global filter loop.
        """
        logger.header2("Starting global iteration #" + str(git) + "\n")

        # update power operator after each iteration step
        S.set_power(newspec=pspec,bare=True)
          
        # uodate operator and optimizer arguments  
        args = (j, S, M, params.rho0)
        D.para = [S, M, m, j, numparams.M0_start, params.rho0, params,\
            numparams]

        if params.uncertainty=='only':
            logger.message('Only calculating uncertainty map as requested.')
            D_hat = D.hat(domain=s_space,\
                ncpu=numparams.ncpu,nrun=numparams.nrun)
            utils.save_results(D_hat.val,"relative uncertainty", \
                'resolve_output_' + str(params.save) +\
                "/D_reconstructions/" + params.save + "_D")
            return None

        # run minimizer
        logger.header2("Computing the WF estimate.\n")

        mold = m

        m = D(j)       

        # save iteration results
        mlist.append(m)
        if params.freq == 'wideband':
            utils.save_results(m.val, "map, iter #" + str(git), \
                'resolve_output_' + str(params.save) +\
                '/m_reconstructions/' + params.save + "_m_WF" +\
                str(wideband_git) + "_" + str(git), rho0 = params.rho0)
            write_output_to_fits(np.transpose(m.val*params.rho0),params, \
                notifier = str(wideband_git) + "_" + str(git),mode = 'I')
        else:
            utils.save_results(m.val, "map, iter #" + str(git), \
                'resolve_output_' + str(params.save) +\
                '/m_reconstructions/' + params.save + "_m_WF" + str(git), \
                rho0 = params.rho0)
            write_output_to_fits(np.transpose(m.val*params.rho0),params,\
                notifier = str(git), mode='I')

        # check whether to do ecf-like noise update
        if params.noise_update:
            
            # Do a "poor-man's" extended critical filter step using residual
            logger.header2("Trying simple noise estimate without any D.")
            newvar = (np.abs((d.val - R(m))**2).mean())
            logger.message('old variance iteration '+str(git-1)+':' + str(N.diag()))
            logger.message('new variance iteration '+str(git)+':' + str(newvar))
            np.save('resolve_output_' + str(params.save) + 'oldvar_'+str(git),N.diag())
            np.save('resolve_output_' + str(params.save) +'newvar_'+str(git),newvar)
            np.save('resolve_output_' + str(params.save) +'absdmean_'\
                +str(git),abs(d.val).mean())
            np.save('resolve_output_' + str(params.save) +'absRmmean_'\
                +str(git),abs(R(m).val*R.target.num()).mean())
            N.para = [newvar*np.ones(np.shape(N.diag()))]

        # Check whether to do the pspec iteration
        if params.pspec:
            logger.header2("Computing the power spectrum.\n")

            # extra loop to take care of possible nans in PS calculation
            psloop = True
            M0 = numparams.M0_start
            while psloop:
            
                D.para = [S, M, m, j, M0, params.rho0, params, numparams]
            
                Sk = projection_operator(domain=k_space)
                #bare=True?
                logger.message('Calculating Dhat for pspec reconstruction.')
                D_hathat = D.hathat(domain=s_space.get_codomain(),\
                    ncpu=numparams.ncpu,nrun=numparams.nrun)
                logger.message('Success.')

                pspec = infer_power(m,domain=Sk.domain,Sk=Sk,D=D_hathat,\
                    q=1E-42,alpha=alpha,perception=(1,0),smoothness=True,var=\
                    numparams.smoothing, bare=True)

                if np.any(pspec == False):
                    print 'D not positive definite, try increasing eta.'
                    if M0 == 0:
                        M0 += 0.1
                    M0 *= 1e6
                    D.para = [S, M, m, j, M0, params.rho0, params, numparams]
                else:
                    psloop = False
            
            logger.message("    Current M0:  " + str(D.para[4])+ '\n.')


        # check whether to do pspec saves
        if params.pspec:
            plist.append(pspec)

            if params.freq == 'wideband':
                utils.save_results(kindex,"ps, iter #" + str(git), \
                    'resolve_output_' + str(params.save) + \
                    "/p_reconstructions/" + params.save + "_p" + \
                    str(wideband_git) + "_" + str(git), value2=pspec,\
                    log='loglog')
            else:
                utils.save_results(kindex,"ps, iter #" + str(git), \
                    'resolve_output_' + str(params.save) +\
                    "/p_reconstructions/" + params.save + "_p" + str(git), \
                    value2=pspec,log='loglog')
            
            # powevol plot needs to be done in place
            pl.figure()
            for i in range(len(plist)):
                pl.loglog(kindex, plist[i], label="iter" + str(i))
            pl.title("Global iteration pspec progress")
            pl.legend()
            pl.savefig('resolve_output_' + str(params.save) + \
                "/p_reconstructions/" + params.save + "_powevol.png")
            pl.close()
        
        # convergence test in map reconstruction
        if np.max(np.abs(m - mold)) < params.map_conv:
            logger.message('Image converged.')
            convergence += 1
        
        # convergence test in power spectrum reconstruction 
        if np.max(np.abs(utils.log(pspec)/utils.log(S.get_power()))) < utils.log(1e-1):
            logger.message('Power spectrum converged.')
            convergence += 1
        
        #global convergence test
        if convergence >= numparams.final_convlevel:
            logger.message('Global convergence achieved at iteration' + \
                str(git) + '.')
            if params.uncertainty=='supp':
                logger.message('Calculating uncertainty map at end of'\
                    +'recpnstruction, as requested.')
                D_hat = D.hat(domain=s_space,\
                ncpu=numparams.ncpu,nrun=numparams.nrun)
                utils.save_results(D_hat.val,"relative uncertainty", \
                    'resolve_output_' + str(params.save) +\
                    "/D_reconstructions/" + params.save + "_D")
                
                
            return m, pspec

        git += 1

    return m, pspec

def ML(d, m, pspec, N, R, logger, k_space, params, numparams,\
    *args):
    """
    Maximum Likelihood estimation.
    """

    if params.freq == 'wideband':
        logger.header1("Begin total intensity wideband Wiener Filter" \
            + "iteration cycle.")  
    else:
        logger.header1("Begin total intensity standard Wiener Filter" \
                    + "iteration cycle.")
    
    s_space = m.domain
    kindex = k_space.get_power_indices()[0]
    
    if params.freq == 'wideband':
        aconst = args[0]
        wideband_git = args[1]
        
    # Defines important operators
    if params.freq == 'wideband':
        M = MI_operator(domain=s_space, sym=True, imp=True, para=[N, R, aconst])
        j = R.adjoint_times(N.inverse_times(d), a = aconst)
    else:    
        M = M_operator(domain=s_space, sym=True, imp=True, para=[N, R])
        j = R.adjoint_times(N.inverse_times(d))
    D = D_operator(domain=s_space, sym=True, imp=True, para=[0, M, m, j, \
        numparams.M0_start, params.rho0, params, numparams])

    
    # diagnostic plots

    if params.freq == 'wideband':
        utils.save_results(j,"j",'resolve_output_' + str(params.save) +\
            '/general/' + params.save + \
            str(wideband_git) + '_j',rho0 = params.rho0)
    else:
        utils.save_results(j,"j",'resolve_output_' + str(params.save) +\
            '/general/' + params.save + '_j',rho0 = params.rho0)

    # iteration parameters
    git = 1
    mlist = [m]
    
          
    args = (j, 0, M, params.rho0)
    D.para = [0, M, m, j, numparams.M0_start, params.rho0, params,\
        numparams]

    if params.uncertainty=='only':
        logger.message('Only calculating uncertainty map as requested.')
        D_hat = D.hat(domain=s_space,\
            ncpu=numparams.ncpu,nrun=numparams.nrun)
        utils.save_results(D_hat.val,"relative uncertainty", \
            'resolve_output_' + str(params.save) +\
            "/D_reconstructions/" + params.save + "_D")
        return None

    # run minimizer
    logger.header2("Computing the ML estimate.\n")

    m = D(j)       

    # save iteration results
    mlist.append(m)
    if params.freq == 'wideband':
        utils.save_results(m.val, "map, iter #" + str(git), \
            'resolve_output_' + str(params.save) +\
            '/m_reconstructions/' + params.save + "_m_WF" +\
            str(wideband_git) + "_" + str(git), rho0 = params.rho0)
        write_output_to_fits(np.transpose(m.val*params.rho0),params, \
            notifier = str(wideband_git) + "_" + str(git),mode = 'I')
    else:
        utils.save_results(m.val, "map, iter #" + str(git), \
            'resolve_output_' + str(params.save) +\
            '/m_reconstructions/' + params.save + "_m_WF" + str(git), \
            rho0 = params.rho0)
        write_output_to_fits(np.transpose(m.val*params.rho0),params,\
            notifier = str(git), mode='I')


    return m 
        
#-------------------parameter classes & code I/O-------------------------------
        
class parameters(object):
    """
    Defines a parameter class for all parameters that are vital to controlling
    basic functionality of the code. Some are even mandatory and given by 
    the user. Performs checks on default arguments with hard-coded default
    values.
    """

    def __init__(self, parset, save, verbosity):

        #mandatory parameters
        self.ms = parset['ms']
        self.imsize = int(parset['imsize'])
        self.cellsize = float(parset['cellsize'])
        
        #non-parset parameters dealing with code-system interactions
        self.save = save
        self.verbosity = verbosity
        
        #main parameters that often need to be set by the user but have default
        #values
        self.check_default('algorithm', parset, 'ln-map')
        self.check_default('stokes', parset, 'I')
        self.check_default('ftmode', parset, 'gfft')
        if self.ftmode == 'wsclean' and wsclean_available == False:
            self.ftmode = 'gfft'
            print 'Wsclean is not available. Gridding routines are set to '\
                +'default despite different user input'
        self.check_default('init_type_s', parset, 'const', dtype=str)
        self.check_default('init_type_u', parset, 'const',dtype=str)
        self.check_default('rho0', parset, 1, dtype = str)
        if not self.rho0 == 'from_sg':
            self.rho0 = float(self.rho0) 
        self.check_default('freq', parset, [0,0], dtype = str)
        if self.freq != 'wideband':
            self.freq = np.array(self.freq.split(),dtype=int)
        self.check_default('pspec', parset, True, dtype = bool)
        self.check_default('pbeam', parset, False,dtype=str)
        self.check_default('uncertainty', parset, '', dtype = str)
        self.check_default('simulating', parset, '', dtype = bool)
        self.check_default('sglogim', parset, '',dtype = bool)
        if self.algorithm == 'ln-map_u':
            self.check_default('sglogim_u', parset, '',dtype = bool)
        if self.simulating:
            self.parset = parset
        self.check_default('noise_est', parset, False, dtype = str)
        self.check_default('noise_update', parset, False, dtype = bool)
        self.check_default('map_conv', parset, 1e-1, dtype = float)
        self.check_default('callback', parset, 3, dtype = int)
        if self.pspec:
            self.check_default('init_type_p', parset, 'k^2_mon', dtype = str)
            self.check_default('pspec_conv', parset, 1e-1, dtype = float)
        if self.freq == 'wideband':
            self.check_default('init_type_a', parset, 'const')
            self.check_default('reffreq', parset, [0,0])
            if self.reffreq != [0,0]:
                self.reffreq = np.array(self.reffreq.split(),dtype=int)
            self.check_default('pspec_a', parset, True, dtype = bool)
            self.check_default('uncertainty_a', parset, '', dtype = str)
            if self.pspec_a:
                self.check_default('init_type_a_p', parset, 'k^2_mon')
        
        
        # a few global parameters needed for callbackfunc
        global gcallback
        global gsave
        gcallback = self.callback
        gsave = self.save
        utils.update_globvars(gsave, gcallback)
        
    def check_default(self, parameter, parset, default, dtype=str):
        
        if parameter in parset:
            if dtype != bool:
                setattr(self, parameter, dtype(parset[str(parameter)]))
            else:
                if parset[str(parameter)] == 'True':
                    setattr(self, parameter,True)
                elif parset[str(parameter)] == 'False' or parset[str(parameter)] == '' :
                    setattr(self, parameter,False)                
        else:
            setattr(self, parameter, default)

class numparameters(object):
    """
    Defines a numerical parameter class for all parameters that mostly control
    numerical features and normally are kept on default values. This is being
    done also internally (and not only on the outside-user-level) for clarity
    so that a developer immediately knows which parameters are considered less
    important for basic code funtionality. Performs checks on default arguments 
    with hard-coded default values.
    """    
    
    def __init__(self, params, parset):
        
        self.check_default('m_start', parset, 0.1, dtype = float)
        self.check_default('global_iter', parset, 50, dtype = int)        
        self.check_default('alpha_prior', parset, False, dtype = str)
        if self.alpha_prior:
            self.alpha_prior = np.array(self.alpha_prior.split(),dtype=int)
        self.check_default('map_algo', parset, 'sd')
        self.check_default('map_alpha', parset, 1e-4, dtype = float)
        self.check_default('map_tol', parset, 1e-5, dtype = float)
        self.check_default('map_clevel', parset, 3, dtype = float)
        self.check_default('map_iter', parset, 100, dtype = int)
        self.check_default('final_convlevel', parset, 4, dtype = float)
        self.check_default('viscol', parset, 'data')
        if params.noise_est == 'SNR_assumed':
            self.check_default('SNR_assumed',parset,1,dtype = float)
        if params.init_type_s != ('const' or 'dirty'):
            self.check_default('zoomfactor', parset, 1, dtype = float)
        
        self.check_default('pspec', parset, True, dtype = bool)
        
        if params.pspec:
            
            self.check_default('pspec_algo', parset, 'cg')
            self.check_default('smoothing', parset, 10, dtype = float)
            self.check_default('M0_start', parset, 0, dtype = float)
            self.check_default('bins', parset, 70, dtype = float)
            self.check_default('p0', parset, 1, dtype = float)
            self.check_default('pspec_tol', parset, 1e-3, dtype = float)
            self.check_default('pspec_clevel', parset, 3, dtype = float)
            self.check_default('pspec_iter', parset, 150, dtype = float)
            self.check_default('ncpu', parset, 2, dtype = float)
            self.check_default('nrun', parset, 8, dtype = float)

        if params.freq == 'wideband':
            
            self.check_default('wb_globiter', parset, 10, dtype = float)
            self.check_default('m_a_start', parset, 0.1, dtype = float)
            self.check_default('global_iter_a', parset, 50, dtype = float)
            self.check_default('alpha_prior_a', parset, False, dtype=str)
            if self.alpha_prior_a:
               self.alpha_prior_a = np.array(self.alpha_prior_a.split(),\
                   dtype=int)
            self.check_default('map_alpha_a', parset, 1e-4, dtype = float)
            self.check_default('map_tol_a', parset, 1e-5, dtype = float)
            self.check_default('map_clevel_a', parset, 3, dtype = float)
            self.check_default('map_iter_a', parset, 100, dtype = float)
            if params.init_type_a != 'const':
                self.check_default('zoomfactor', parset, 1, dtype = float)
            
            if params.pspec_a:
                
                self.check_default('smoothing_a', parset, 10, dtype = float)
                self.check_default('M0_start_a', parset, 0, dtype = float)
                self.check_default('bins_a', parset, 70, dtype = float)
                self.check_default('p0_a', parset, 1, dtype = float)
                self.check_default('pspec_tol_a', parset, 1e-3, dtype = float)
                self.check_default('pspec_clevel_a', parset, 3, dtype = float)
                self.check_default('pspec_iter_a', parset, 150, dtype = float)
                self.check_default('ncpu_a', parset, 2, dtype = float)
                self.check_default('nrun_a', parset, 8, dtype = float)
                
        if params.algorithm == 'ln-map_u':
            
            self.check_default('m_u_start', parset,0.1, dtype = float)
            self.check_default('beta', parset,1.5, dtype = float)
            self.check_default('eta', parset,1e-7, dtype = float)
            self.check_default('map_alpha_u', parset,1e-4, dtype = float)
            self.check_default('map_tol_u', parset,1e-5, dtype = float)
            self.check_default('map_clevel_u', parset,3, dtype = float)
            self.check_default('map_iter_u', parset,100, dtype = float)
            self.algo_liste = self.set_algo_liste(self.map_algo)
            
    def set_algo_liste(self,st):
        
        scipyminimizer = ('TNC','COBYLA','SLSQP','dogleg','trust-ncg',\
            'CG','BFGS', 'L-BFGS-B','Nelder-Mead','Powell','Newton-CG')
    
        if st == 'sd':
            return ['sd_u','sd_m','ps_rec']
        elif st in scipyminimizer:
            return [st,'ps_rec']
        else:
            return np.array(st.split(),dtype=str)
        
    def check_default(self, parameter, parset, default, dtype=str):
        
        if parameter in parset:
            if dtype != bool:
                setattr(self, parameter, dtype(parset[str(parameter)]))
            else:
                if parset[str(parameter)] == 'True':
                    setattr(self, parameter,True)
                elif parset[str(parameter)] == 'False' or parset[str(parameter)] == '' :
                    setattr(self, parameter,False)                
        else:
            setattr(self, parameter, default)


def parse_input_file(parsetfn, save, verbosity):
    """ parse the parameter file."""

    reader = csv.reader(open(parsetfn, 'rb'), delimiter=" ",
                        skipinitialspace=True)
    parset = dict()

    # File must fulfil: No whitespaces beyond variables; no free lines;
    # one whitespace to set variables to False
    for row in reader:
        if len(row) != 0 and row[0] != '%' and len(row)>2:
            st = row[1]
            for i in range(2,len(row)):             
                st +=' '+row[i]
            parset[row[0]] = st #row[1]+' '+row[2]
        else:
            parset[row[0]] = row[1]


    params = parameters(parset, save, verbosity)       
    
    numparams = numparameters(params, parset)

    return params, numparams


def write_output_to_fits(m, params, notifier='',mode='I',u=None):
    """
    """

    hdu_main = pyfits.PrimaryHDU(utils.convert_RES_to_CASA(m,FITS=True))
    
    try:
        generate_fitsheader(hdu_main, params)
    except:
        print "Warning: There was a problem generating the FITS header, no " + \
            "header information stored!"
        print "Unexpected error:", sys.exc_info()[0]
    hdu_list = pyfits.HDUList([hdu_main])
    
        
    if mode == 'I':
        hdu_list.writeto('resolve_output_' + str(params.save) +\
                '/' + str(params.save) + '_' + 'expm' + str(notifier) + '.fits', clobber=True)
    elif mode == 'I_u':
        hdu_list.writeto('resolve_output_' + str(params.save) +\
                '/' + str(params.save) + '_' + 'expu' + str(notifier) + '.fits', clobber=True)  
    elif mode == 'I_mu':
        hdu_list.writeto('resolve_output_' + str(params.save) +\
                '/' + str(params.save) + '_' + 'expmu' + str(notifier) + '.fits', clobber=True)                 
    else:
        hdu_list.writeto('resolve_output_' + str(params.save) +\
                '/' + str(params.save) + '_' + 'a' + str(notifier) + '.fits', clobber=True)


def generate_fitsheader(hdu, params):
    """
    """

    
    today = datetime.datetime.today()
    
    #hdu.header = inhead.copy()
    
    hdu.header.update('ORIGIN', 'resolve.py', 'Origin of the data set')
    
    hdu.header.update('DATE', str(today), 'Date when the file was created')
    
    hdu.header.update('NAXIS', 2,
                      'Number of axes in the data array, must be 2')

    hdu.header.update('BUNIT','JY/PIXEL')
                      
    hdu.header.update('NAXIS1', params.imsize,
                      'Length of the RA axis')
    
    hdu.header.update('CTYPE1', 'RA--SIN', 'Axis type')
    
    hdu.header.update('CUNIT1', 'RAD', 'Axis units')
    
    # In FITS, the first pixel is 1, not 0!!!
    hdu.header.update('CRPIX1', params.imsize/2, 'Reference pixel')
    
    hdu.header.update('CRVAL1', params.summary['field_0']['direction']\
    ['m0']['value'], 'Reference value')
    
    hdu.header.update('CDELT1', -1 * params.cellsize, 'Size of pixel bin')
    
    hdu.header.update('NAXIS2', params.imsize,
                      'Length of the RA axis')
    
    hdu.header.update('CTYPE2', 'DEC-SIN', 'Axis type')
    
    hdu.header.update('CUNIT2', 'RAD', 'Axis units')
    
    # In FITS, the first pixel is 1, not 0!!!
    hdu.header.update('CRPIX2', params.imsize/2, 'Reference pixel')
    
    hdu.header.update('CRVAL2', params.summary['field_0']['direction']\
    ['m1']['value'], 'Reference value')
    
    hdu.header.update('CDELT2', params.cellsize, 'Size of pixel bin')
    
    hdu.header.add_history('RESOLVE: Reconstruction performed by ' +
                           'resolve.py.')

    
    hdu.header.__delitem__('NAXIS4')
    hdu.header.__delitem__('CTYPE4')
    hdu.header.__delitem__('CRVAL4')
    hdu.header.__delitem__('CRPIX4')
    hdu.header.__delitem__('CDELT4')
    hdu.header.__delitem__('CUNIT4')  
 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("parsetfn", type=str,
                        help="name of parset file to use")
    parser.add_argument("-s","--save", type=str, default='save',
                        help="output string for save directories.")
    parser.add_argument("-c","--casaload", type=bool, default=False,
                        help="attempt to read data using CASA. Not yet"\
                            +' fully implemented!')
    parser.add_argument("-v","--verbosity", type=int, default=2,
                        help="Reset verbosity level of code output. Default"\
                            +' is 2/5.')                    
    args = parser.parse_args()
    
    if args.casaload:
        raise NotImplementedError('Direct casaload-feature not yet'\
            +' implemented. Please provide ms-data in numpy-array form.')
    
    #this is done for greater customization and control of file parsing than
    #doing it directly with a file-type in argparse     
    params, numparams = parse_input_file(args.parsetfn, args.save, args.verbosity)
    
    resolve(params, numparams)
