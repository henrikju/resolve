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

from __future__ import division
from time import time
from subprocess import call

#import necessary standard modules
import pylab as pl
import numpy as np
from nifty import *
import messenger as M
import response as r
from nifty import nifty_tools as nt
import scipy.stats as sc
import pyfits
import sys
import datetime
import argparse

#import RESOLVE-package modules
import utility_functions as utils
import simulation.resolve_simulation as sim
import casatools.resolve_casa_functions as cas
from operators import *

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

    # Set up message logger            
    logfile = 'resolve_output_'+str(params.save)+'/general/' + params.save + \
        '.log'
    logger = M.Messenger(verbosity=params.verbosity, add_timestamp=False, \
        logfile=logfile)

    # Basic start messages, parameter log
    logger.header1('Starting RESOLVE.')
    
    logger.message('The choosen main parameters are:\n')
    for par in vars(params).items():
        logger.message(str(par[0])+' = '+str(par[1]),verb_level=2)
    logger.message('All further default parameters are:\n')
    for par in vars(numparams).items():
        logger.message(str(par[0])+' = '+str(par[1]),verb_level=2)
    
    
    # Data setup
    if simulating:
        d, N, R, di, d_space, s_space, expI, n = simulate(params, simparams, \
            logger)
        
    else:
        d, N, R, di, d_space, s_space = datasetup(params, logger)

    # Starting guesses setup
    if params.algorithm == 'ln-map':
        m_s, pspec, params = starting_guess_setup()

    if params.algorithm == 'ln-map_u':
        m_s, pspec, m_u, params = starting_guess_setup()

    if params.algorithm == 'ln-map' and params.freq == 'wideband':
        m_s, pspec, m_a, pspec_a, params = starting_guess_setup()

    if params.algorithm == 'ln-map_u' and params.freq == 'wideband':
        m_s, pspec, m_u, m_a, pspec_a, params = starting_guess_setup()        
              
    # Begin: Start Filter *****************************************************

    if params.algorithm == 'ln-map':
        
        logger.header2('Starting standard RESOLVE reconstruction.')
        
        if params.uncertainty == 'only':
            #single-band uncertainty map
            t1 = time()
            mapfilter_I(d, m_s, pspec, N, R, logger, k_space,\
                params, numparams)
            t2 = time()
            logger.success("Completed uncertainty map calculation.")
            logger.message("Time to complete: " + str((t2 - t1) / 3600.) + ' hours.')
            sys.exit(0)

        if params.freq is not 'wideband':
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

        if params.freq is not 'wideband':        
            t1 = time()                  
            m_s,m_u, p_I = mapfilter_I_u(d, m_s,m_u, pspec, N, R, logger, k_space, \
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
                k_space, params, numparams, np.log(exp(m_s)+exp(m_u)),0)
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
                    params, numparams)      

                #a-Filter                                                                                                                                                                              
                m_a, p_a, = mapfilter_a(d, m_a, pspec_a, N, R, logger, rho0,\
                k_space, params, numparams, np.log(exp(m_s)+exp(m_u)), wideband_git)


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
        
        save_results(exp(m_s.val),"exp(Solution m)",\
            'resolve_output_' + str(params.save) + '/m_reconstructions/' + \
            params.save + "_expmfinal", rho0 = params.rho0)
        write_output_to_fits(np.transpose(exp(m_s.val)*params.rho0),params, notifier='final',mode='I')
        
        if params.freq == 'wideband':
            save_results(m_a.val,"Solution a",\
                'resolve_output_' + str(params.save) + '/m_reconstructions/' +\
                params.save + "_mafinal", rho0 = params.rho0)
            write_output_to_fits(np.transpose(m_a.val),params, notifier ='final', mode='a')
            
        save_results(kindex,"final power spectrum",\
                     'resolve_output_' + str(params.save) + \
                     '/p_reconstructions/' + params.save + "_powfinal", \
                     value2 = p_I, log='loglog')
        

    # ^^^ End: Some plotting stuff *****************************************^^^


#------------------------------------------------------------------------------


def datasetup(params, logger):
    """
        
    Data setup, feasible for the nifty framework.
        
    """
    
    logger.header2("Running data setup.")
    
    #Somewhat inelegant solution, but WSclean needs its own I/O 
    if params.ftmode == 'wsclean':
        
        wscleanpars = w.ImagingParameters()
        wscleanpars.msPath = params.ms
        wscleanpars.imageWidth = params.imsize
        wscleanpars.imageHeight = params.imsize
        wscleanpars.pixelScaleX = str(params.cellsize)+'rad'
        wscleanpars.pixelScaleY = str(params.cellsize)+'rad'
        wscleanpars.extraParameters = '-weight natural -nwlayers 1 -j 4'
      
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
            logger.failure('No numpy MS header file found. FITS output will'\
                +'deactivated.')
            params.summary = None

    #Non-WSclean (i.e. standard) loading routines
    else:
    
        if params.casaload:
            
            try:
                
                if params.noise_est == 'full':
                    os.system('casapy --nologger -c' \
                    +'/casatools/resolve_casa_functions.py -ms ['\
                    +params.ms+','+params.viscol+','+'sigma,'+'tot,'+'True]')
                    
                else:
                    os.system('casapy --nologger -c' \
                    +'/casatools/resolve_casa_functions.py -ms ['\
                    +params.ms+','+params.viscol+','+'sigma,'+'tot,'+'False]')
                
                vis, sigma, u, v, freqs, nchan, nspw, nvis, params.summary = \
                utils.load_numpy_data(params.ms, logger)
    
            except:
                
                logger.failure("Could not use CASA to directly to read in "\
                    + "measurement set with name "+str(params.ms))
                logger.message("Trial read-in interpreting "+str(params.ms)+\
                    "as .npy-file suffix.")
                params.casaload = False
    
        if not params.casaload:
    
            vis, sigma, u, v, freqs, nchan, nspw, nvis, params.summary = \
                utils.load_numpy_data(params.ms, logger)
    
    # definition of wideband data operators
    if params.freq == 'wideband':
        
        # wideband data and noise settings. Inelegant if not statement needed
        # because wsclean routines don't explicitly read out these things
        if not params.ftmode == 'wsclean': 
            u = np.array(u)
            v = np.array(v)
            freqs = np.array(freqs)
            nspw = nspw[0]+1
            nchan = nchan[0]
            nvis = nvis[0]
        
        # Dumb simple estimate can be done now after reading in the data.
        # No time information needed.
        if params.noise_est == 'simple':
            
            variance = np.var(np.array(vis))*np.ones(np.shape(np.array(vis)))\
                .flatten()
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
        logger.message('Max. instrumental resolution over all spw\n' + 'rad ' \
            + str(dx_real_rad) + '\n' + 'asec ' + str(dx_real_rad/asec2rad))

        save_results(uflat,'UV_allspw', 'resolve_output_' + str(params.save) +\
            '/general/' + params.save + "_uvcov", \
            plotpar='o', value2 = vflat)
        
        d_space = point_space(nspw*nchan*nvis, datatype = np.complex128)
        s_space = rg_space(params.imsize, naxes=2, dist = params.cellsize,\
            zerocenter=True)
        
        # primary beam function
        if params.pbeam is None:
            A = np.array([[np.ones((params.imsize,params.imsize))\
                for k in range(nchan)] for j in range(nspw)])
        else:
            logger.message('No wideband primary beam implemented.'+
                'Automatically set to one.')
            A = np.array([[np.ones((params.imsize,params.imsize))\
                for k in range(nchan)] for j in range(nspw)])
            
        # response operator
        if params.ftmode == 'gfft':
            R = r.response_mfs(s_space, target=d_space, \
                               u,v,A,nspw,nchan,nvis,freqs,params.reffreq[0],\
                               params.reffreq[1],params.ftmode)
        else:
            logger.failure('For wideband mode only gfft support is available.')
        
        d = field(d_space, val=np.array(vis).flatten())

        N = N_operator(domain=d_space,imp=True,para=[variance])
        
        # dirty image from CASA or Resolve for comparison
        if params.init_type_s=='dirty':
            if params.casaload:
                try:
                    os.system('casapy --nologger -c' \
                        '/casatools/resolve_casa_functions.py -di [' \
                        +params.ms+str(params.cellsize)+str(params.imsize)\
                        +params.save+']')
                    di = np.load('/resolve_output_'+save+'/general/di.npy')
                except:
                    logger.warn('Some problem occured while attempting to'\
                    +'use CASA for dirty image production. Resort to interal'\
                    +'routines')
                    R.adjointfactor = 1
                    di = R.adjoint_times(d)
                    R()
                    
        else:
            R.adjointfactor = 1
            di = R.adjoint_times(d) * s_space.vol[0] * s_space.vol[1]
            R()
            
        # more diagnostics 
        # plot the dirty beam
        uvcov = field(d_space,val=np.ones(np.shape(np.array(vis).flatten()), \
             dtype = np.complex128))            
        db = R.adjoint_times(uvcov) * s_space.vol[0] * s_space.vol[1]       
        save_results(db,"dirty beam",'resolve_output_' + str(params.save)+\
            '/general/' + params.save + "_db")
            
        # plot the dirty image
        save_results(di,"dirty image",'resolve_output_' +str(params.save)+\
            '/general/' + params.save + "_di")


        return  d, N, R, di, d_space, s_space
        
    # definition of single band data operators
    else:
        
        # data and noise settings. Inelegant if not statement needed
        # because wsclean routines don't explicitly read out these things
        if not params.ftmode == 'wsclean':
            sspw,schan = params.freq[0], params.freq[1]
            vis = vis[sspw][schan]
            sigma = sigma[sspw][schan]
            u = u[sspw][schan]
            v = v[sspw][schan]        
        
        # Dumb simple estimate can be done now after reading in the data.
        # No time information needed.
        if params.noise_est == 'simple':
            
            variance = np.var(np.array(vis))*np.ones(np.shape(np.array(vis)))\
                .flatten()
        else:
            variance = (np.array(sigma)**2).flatten()

        # basic diagnostics                                                         
        # maximum k-mode and resolution of data                                      
        uvrange = np.array([np.sqrt(u[i]**2 + v[i]**2) for i in range(len(u))])
        dx_real_rad = (np.max(uvrange))**-1
        logger.message('Max. instrumental resolution\n' + 'rad ' \
            + str(dx_real_rad) + '\n' + 'asec ' + str(dx_real_rad/asec2rad))

        save_results(u,'UV', 'resolve_output_' + str(params.save) +\
            '/general/' + params.save + "_uvcov", plotpar='o', value2 = v)

        d_space = point_space(len(u), datatype = np.complex128)
        s_space = rg_space(params.imsize, naxes=2, dist = params.cellsize, \
            zerocenter=True)
        
        # primary beam function
        if params.pbeam:
            try:
                logger.message('Attempting to load primary beam from'\
                + 'file' + str(params.pbeam) +'.npy')
                A = np.load(params.pbeam+'.npy')
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
        if params.casaload:
            try:
                os.system('casapy --nologger -c' \
                    '/casatools/resolve_casa_functions.py -di [' \
                    +params.ms+str(params.cellsize)+str(params.imsize)\
                    +params.save+']')
                di = np.load('/resolve_output_'+save+'/general/di.npy')
            except:
                logger.warn('Some problem occured while attempting to'\
                +'use CASA for dirty image production. Resort to interal'\
                +'routines')
                R.adjointfactor = 1
                di = R.adjoint_times(d)
                R = r.response(s_space, d_space, u, v, A)
                    
        else:
            R.adjointfactor = 1
            di = R.adjoint_times(d) * s_space.vol[0] * s_space.vol[1]
            R = r.response(s_space, d_space, u, v, A)   
        
        # more diagnostics 
        # plot the dirty beam
        uvcov = field(d_space,val=np.ones(len(u), dtype = np.complex128))
        db = R.adjoint_times(uvcov) * s_space.vol[0] * s_space.vol[1]       
        save_results(db,"dirty beam", 'resolve_output_' +str(params.save)+\
            '/general/' + params.save + "_db")
        
        # plot the dirty image
        save_results(di,"dirty image",'resolve_output_' +str(params.save)+\
            '/general/' + params.save + "_di")

        return  d, N, R, di, d_space, s_space
    

def starting_guess_setup(params):
    
    # Starting guesses for m_s

    if params.init_type_s == 'const:
        m_s = field(s_space, val = numparams.m_start)

    if params.init_type_s == 'dirty':
        m_s = field(s_space, target=s_space.get_codomain(), val=di)   
        
    else:
        if params.casaload:
            try:
                # Read-in userimage, convert to Jy/px and transpose to Resolve
                userimage = read_image_from_CASA(params.init_type_s,\
                    numparams.zoomfactor)
            except:
                logger.warn("Could not find a CASA image at path"\
                    + params.init_type_s)
                logger.message("Trial read-in as .npy-file")
                try:
                    userimage = np.load(params.init_type_s)
                except:
                    logger.failure("No .npy-file existing. Default read-in of"\
                        +"dirty image as starting guess")
                    userimage = di
                
            m_s = field(s_space, target=s_space.get_codomain(), val=userimage)
        else:
            m_s = field(s_space, target=s_space.get_codomain(), \
            val=np.load(params.init_type_s))
    
    # Optional starting guesses for m_u
            
    if params.algorithm == 'ln-map_u':   
        
        if params.init_type_u == 'const':
            m_u = field(s_space, val = numparams.m_u_start)
        
        if params.init_type_u == 'dirty':
            m_u = field(s_space, target=s_space.get_codomain(), val=di)   
            
        else:
            if params.casaload:
                try:
                    # Read-in userimage, convert to Jy/px and transpose to 
                    # Resolve
                    userimage = read_image_from_CASA(params.init_type_u,\
                        numparams.zoomfactor)
                except:
                    logger.warn("Could not find a CASA image at path"\
                        + params.init_type_s)
                    logger.message("Trial read-in as .npy-file")
                    try:
                        userimage = np.load(params.init_type_s)
                    except:
                        logger.failure("No .npy-file existing. Default"\
                        +"read-in of dirty image as starting guess")
                        userimage = di
                    
                m_u = field(s_space, target=s_space.get_codomain(),\
                    val=userimage)
            else:
                m_u = field(s_space, target=s_space.get_codomain(), \
                val=np.load(params.init_type_u))
                
    if params.rho0 == 'from_sg':
        
        if params.algorithm == 'ln-map_u':
            params.rho0 = np.mean(m_s.val[np.where(m_s.val>=np.max(m_s.val)\
                / 10)] + m_u.val[np.where(m_u.val>=np.max(m_u.val)/ 10)])
        else:
             params.rho0 = np.mean(m_s.val[np.where(m_s.val>=np.max(m_s.val)\
                / 10)])
        logger.message('rho0 was calculated as: ' + str(params.rho0)) 
        
    if not params.rho0 == 1.:
        
        m_s /= params.rho0
        if params.algorithm == 'ln-map_u':
            m_u /= params.rho0

    np.save('resolve_output_' + str(params.save)+'/general/rho0',rho0)
    if rho0 < 0:
        logger.warn('Monopole level of starting guess negative. Probably due \
            to too many imaging artifcts in userimage')
        
            
    # Starting guesses for pspec 

    # Basic k-space
    k_space = R.domain.get_codomain()
        
    #Adapts the k-space properties if binning is activated.
    if not numparams.bin is None:
        k_space.set_power_indices(log=True, nbins=numparams.bin)
    
    # k-space prperties    
    kindex,rho_k,pindex,pundex = k_space.get_power_indices()

    # Simple k^2 power spectrum with p0 from numpars and a fixed monopole from
    # the m starting guess
    if params.init_type_p == 'k^2_mon':
        pspec = np.array((1+kindex)**-2 * numparams.p0)
        pspec_mtemp = mtemp.power(pindex=pindex, kindex=kindex, rho=rho_k)
        #see notes, use average power in dirty map to constrain monopole
        pspec[0] = (np.prod(k_space.vol)**(-2) * np.log(\
            np.sqrt(pspec_mtemp[0]) *  np.prod(k_space.vol))**2) / 2.
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
    if freq == 'wideband':
        
        # Starting guesses for m_a

        if params.init_type_a == 'const':
            m_a = field(s_space, val = numparams.m_a_start)  
            
        else:
            if params.casaload:
                try:
                    # Read-in userimage, convert to Jy/px and transpose to 
                    # Resolve
                    userimage = read_image_from_CASA(params.init_type_a,\
                        numparams.zoomfactor)
                except:
                    logger.warn("Could not find a CASA image at path"\
                        + params.init_type_s)
                    logger.message("Trial read-in as .npy-file")
                    try:
                        userimage = np.load(params.init_type_a)
                    except:
                        logger.failure("No .npy-file existing. Default set "\
                        +"of constant starting guess")
                        userimage = numparams.m_a_start
                    
                m_a = field(s_space, target=s_space.get_codomain(),\
                    val=userimage)
            else:
                m_a = field(s_space, target=s_space.get_codomain(), \
                    val=np.load(params.init_type_a))        
        
        # Spectral index pspec starting guesses
        
        # default simple k^2 spectrum with free monopole
        if params.init_type_p_a == 'k^2':
            pspec_a = np.array((1+kindex)**-2 * numparams.p0_a)    
        # constant power spectrum guess 
        elif params.init_type_p_a == 'constant':
            pspec_a = numparams.p0_a
        # power spectrum from last iteration 
        else:
            logger.message('using last p-iteration from previous run.')
            pspec_a = np.load(params.init_type_p_a)

        if np.any(pspec_a) == 0:
            pspec_a[pspec_a==0] = 1e-25
 
    # diagnostic plot of m starting guess
    save_results(exp(m_s.val),"TI exp(Starting guess)",\
        'resolve_output_' + str(params.save) + '/m_reconstructions/' +\
        params.save + "_expm0", rho0 = params.rho0)
    write_output_to_fits(np.transpose(exp(m_s.val)*rho0),params, \
        notifier='0', mode='I')
    if freq == 'wideband':
        save_results(m_a.val,"Alpha Starting guess",\
            'resolve_output_' + str(params.save) + '/m_reconstructions/' +\
            params.save + "_ma0", rho0 = params.rho0) 
        write_output_to_fits(np.transpose(m_s.val),params, notifier='0', \
               mode='a') 
    if params.algorithm == 'ln-map_u':
        save_results(exp(m_u.val),"TI exp(Starting guess)",\
            'resolve_output_' + str(params.save) + '/u_reconstructions/' +\
            params.save + "_expu0", rho0 = params.rho0)    
        write_output_to_fits(np.transpose(exp(m_u.val)*rho0),params, \
            notifier='0', mode='I_u')       
                
        save_results(exp(m_s.val)+exp(m_u.val),"TI exp(Starting guess)",\
            'resolve_output_' + str(params.save) + '/mu_reconstructions/'+\
            params.save + "_expmu0", rho0 = rho0)    
        write_output_to_fits(np.transpose(exp(m_s.val+expm_u.val)*rho0),\
            params, notifier='0', mode='I_mu')         
    
    if params.algorithm == 'ln-map':
        return m_s, pspec, params
    
    if params.algorithm == 'ln-map_u':
        return m_s, pspec, m_u, params

    if params.algorithm == 'ln-map' and params.freq == 'wideband':
        return m_s, pspec, m_a, pspec_a, params

    if params.algorithm == 'ln-map_u' and params.freq == 'wideband':
        return m_s, pspec, m_u, m_a, pspec_a, params


def mapfilter_I(d, m, pspec, N, R, logger, k_space, params, numparams,\
    *args):
    """
    Main standard MAP-filter iteration cycle routine.
    """

    if params.freq == 'wideband':
        logger.header1("Begin total intensity wideband RESOLVE iteration cycle.")  
    else:
        logger.header1("Begin total intensity standard RESOLVE iteration cycle.")
    
    s_space = m.domain
    kindex = k_space.get_power_indices()[0]
    
    if params.freq == 'wideband':
        aconst = args[0]
        wideband_git = args[1]
         
    # Sets the alpha prior parameter for all modes
    if not numparams.alpha_prior is None:
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
        numparams.M0_start, rho0, params, numparams])

    
    # diagnostic plots

    if params.freq == 'wideband':
        save_results(j,"j",'resolve_output_' + str(params.save) +\
            '/general/' + params.save + \
            str(wideband_git) + '_j',rho0 = params.rho0)
    else:
        save_results(j,"j",'resolve_output_' + str(params.save) +\
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
            save_results(D_hat.val,"relative uncertainty", \
                'resolve_output_' + str(params.save) +\
                "/D_reconstructions/" + params.save + "_D")
            return None

        # run minimizer
        logger.header2("Computing the MAP estimate.\n")

        mold = m
       
        if params.map_algo == 'sd':
            en = energy(args)
            minimize = nt.steepest_descent(en.egg,spam=callbackfunc,note=True)
            m = minimize(x0=m, alpha=numparams.map_alpha, \
                tol=numparams.map_tol, clevel=numparams.map_clevel, \
                limii=numparams.map_iter)[0]
        elif params.map_algo == 'lbfgs':
            logger.warn('lbfgs algorithm implemented from scipy, but'\
                + 'experimental.')
            m = utils.BFGS(m,j,S,M,rho0,params,limii=numparams.map_iter)
           
        # save iteration results
        mlist.append(m)
        if params.freq == 'wideband':
            save_results(exp(m.val), "map, iter #" + str(git), \
                'resolve_output_' + str(params.save) +\
                '/m_reconstructions/' + params.save + "_expm" +\
                str(wideband_git) + "_" + str(git), rho0 = params.rho0)
            write_output_to_fits(np.transpose(exp(m.val)*params.rho0),params, \
                notifier = str(wideband_git) + "_" + str(git),mode = 'I')
        else:
            save_results(exp(m.val), "map, iter #" + str(git), \
                'resolve_output_' + str(params.save) +\
                '/m_reconstructions/' + params.save + "_expm" + str(git), \
                rho0 = params.rho0)
            write_output_to_fits(np.transpose(exp(m.val)*params.rho0),params,\
                notifier = str(git), mode='I')

        # check whether to do ecf-like noise update
        if params.noise_update:
            
            # Do a "poor-man's" extended critical filter step using residual
            logger.header2("Trying simple noise estimate without any D.")
            newvar = (np.abs((d.val - R(exp(m)))**2).mean())
            logger.message('old variance iteration '+str(git-1)+':' + str(N.diag()))
            logger.message('new variance iteration '+str(git)+':' + str(newvar))
            np.save('resolve_output_' + str(params.save) + 'oldvar_'+str(git),N.diag())
            np.save('resolve_output_' + str(params.save) +'newvar_'+str(git),newvar)
            np.save('resolve_output_' + str(params.save) +'absdmean_'\
                +str(git),abs(d.val).mean())
            np.save('resolve_output_' + str(params.save) +'absRmmean_'\
                +str(git),abs(R(exp(m)).val*R.target.num()).mean())
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
                save_results(kindex,"ps, iter #" + str(git), \
                    'resolve_output_' + str(params.save) + \
                    "/p_reconstructions/" + params.save + "_p" + \
                    str(wideband_git) + "_" + str(git), value2=pspec,\
                    log='loglog')
            else:
                save_results(kindex,"ps, iter #" + str(git), \
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
        if np.max(np.abs(np.log(pspec)/np.log(S.get_power()))) < np.log(1e-1):
            logger.message('Power spectrum converged.')
            convergence += 1
        
        #global convergence test
        if convergence >= numparams.finalconvlevel:
            logger.message('Global convergence achieved at iteration' + \
                str(git) + '.')
            if params.uncertainty=='supp':
                logger.message('Calculating uncertainty map at end of'\
                    +'recpnstruction, as requested.')
                D_hat = D.hat(domain=s_space,\
                ncpu=numparams.ncpu,nrun=numparams.nrun)
                save_results(D_hat.val,"relative uncertainty", \
                    'resolve_output_' + str(params.save) +\
                    "/D_reconstructions/" + params.save + "_D")
                
                
            return m, pspec

        git += 1

    return m, pspec
    
    
def mapfilter_I_u(d, m,u, pspec, N, R, logger, rho0, k_space, params, numparams,\
    *args):
    """
    """

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
    if not numparams.alpha_prior is None:
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
   
    #diagnostic plots    
    if params.freq == 'wideband':
        save_results(j,"j",'resolve_output_' + str(params.save) +\
        '/general/' + params.save + \
            str(wideband_git) + '_j',rho0 = params.rho0)
    else:
        save_results(j,"j",'resolve_output_' + str(params.save) +\
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
        logger.header2("Starting global up iteration #" + str(git) + "\n")

        # update power operator after each iteration step
        S.set_power(newspec=pspec,bare=True)
          
        # uodate operator and optimizer arguments  
        D.para = [S, M, m, j, numparams.M0_start, params.rho0, params, numparams]

        # Check whether to only calculte an uncertainty map
        if params.uncertainty=='only':
            logger.message('Only calculating uncertainty map as requested.')
            D_hat = D.hat(domain=s_space,\
                ncpu=numparams.ncpu,nrun=numparams.nrun)
            save_results(D_hat.val,"relative uncertainty", \
                'resolve_output_' + str(params.save) +\
                "/D_reconstructions/" + params.save + "_D")
            return None


        #run nifty minimizer steepest descent class
        logger.header2("Computing the MAP estimate.\n")

        mold = m
        uold = u

        args = (j, S, M, params.rho0, numparams.beta, numparams.eta,m,u)
        
        if params.map_algo == 'sd':
            en = energy_mu(args) 
               
            minimize = nt.steepest_descent(en.egg_s,spam=callbackfunc_m,\
                note=True)
            m = minimize(x0=m, alpha=numparams.map_alpha, \
                tol=numparams.map_tol, clevel=numparams.map_clevel, \
                limii=numparams.map_iter)[0]    
            en.seff = m              
            
            minimize = nt.steepest_descent(en.egg_u,spam=callbackfunc_u,\
                note=True)
            u = minimize(x0=u, alpha=numparams.map_alpha_u, \
                tol=numparams.map_tol_u, clevel=numparams.map_clevel_u, \
                limii=numparams.map_iter_u)[0]
            en.ueff = u
       
        elif params.map_algo == 'lbfgs':
            logger.warn('lbfgs algorithm implemented from scipy, but'\
                + 'experimental.')
            raise NotImplementedError('WARNING: BFGS for mu still needed!')
            #m = utils.BFGS(m,j,S,M,rho0,numparams.beta, numparams.etalimii=numparams.map_iter)
            #u = utils.BFGS(u,j,S,M,rho0,params,limii=numparams.map_iter_u)                  
                
        # save iteration results
        mlist.append(m)
        if params.freq == 'wideband':
            save_results(exp(m.val), "map, iter #" + str(git), \
                'resolve_output_' + str(params.save) +\
                '/m_reconstructions/' + params.save + "_expm" +\
                str(wideband_git) + "_" + str(git), rho0 = params.rho0)
            write_output_to_fits(np.transpose(exp(m.val)*params.rho0),params, \
                notifier = str(wideband_git) + "_" + str(git),mode = 'I')
            
        else:
            save_results(exp(m.val), "map, iter #" + str(git), \
                'resolve_output_' + str(params.save) +\
                '/m_reconstructions/' + params.save + "_expm" + str(git), \
                rho0 = params.rho0)
            write_output_to_fits(np.transpose(exp(m.val)*params.rho0),params,\
                notifier = str(git), mode='I')
                
        save_results(exp(u.val), "map, iter #" + str(git), \
            'resolve_output_' + str(params.save) +\
            '/u_reconstructions/' + params.save + str(git), \
            rho0 = params.rho0)
        write_output_to_fits(np.transpose(exp(u.val)*params.rho0),params,\
            notifier = str(git), mode='I_u')       
                
        save_results(exp(u.val)+exp(m.val), "map, iter #" + str(git), \
            'resolve_output_' + str(params.save) +\
            '/mu_reconstructions/' + params.save + str(git), \
            rho0 = params.rho0)
        write_output_to_fits(np.transpose((exp(u.val)+exp(m.val))*params.rho0),params,\
            notifier = str(git), mode='I_mu')                      
        
        # check whether to do ecf-like noise update
        if params.noise_update:
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
        if params.pspec:
            logger.header2("Computing the power spectrum.\n")

            #extra loop to take care of possible nans in PS calculation
            psloop = True
            M0 = numparams.M0_start
                  
            while psloop:
            
                D.para = [S, M, m, j, M0, params.rho0, params, numparams]
            
                Sk = projection_operator(domain=k_space)
                #bare=True?
                logger.message('Calculating Dhat.')
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
            save_results(kindex,"ps, iter #" + str(git), \
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
        
        # convergence test in map reconstruction
        if np.max(np.abs(m - mold)) < params.map_conv and np.max(np.abs(u - uold)) < params.map_conv:
            logger.message('Image converged.')
            convergence += 1
        
        # convergence test in power spectrum reconstruction 
        if np.max(np.abs(np.log(pspec)/np.log(S.get_power()))) < np.log(1e-1):
            logger.message('Power spectrum converged.')
            convergence += 1
        
        #global convergence test
        if convergence >= numparams.finalconvlevel:
            logger.message('Global convergence achieved at iteration' + \
                str(git) + '.')
            if params.uncertainty=='supp':
                logger.message('Calculating uncertainty map as requested.')
                D_hat = D.hat(domain=s_space,\
                ncpu=numparams.ncpu,nrun=numparams.nrun)
                save_results(D_hat.val,"relative uncertainty", \
                    'resolve_output_' + str(params.save) +\
                    "/D_reconstructions/" + params.save + "_D")
                
            return m,u, pspec

        git += 1

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
    if not numparams.alpha_prior_a is None:
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
            save_results(D_hat.val,"relative uncertainty", \
                'resolve_output_' + str(params.save) +\
                "/D_reconstructions/" + params.save + "_D")
            return None

        #run nifty minimizer steepest descent class
        logger.header2("Computing the spectral index MAP estimate.\n")

        mold = m
        
        if params.map_algo == 'sd':
            en = energy_a(args)
            minimize = nt.steepest_descent(en.egg,spam=callbackfunc,note=True)
            m = minimize(x0=m, alpha=numparams.map_alpha_a, \
                tol=numparams.map_tol_a, clevel=numparams.map_clevel_a, \
                limii=numparams.map_iter_a)[0]
        elif params.map_algo == 'lbfgs':
            logger.warn('lbfgs algorithm implemented from scipy, but'\
            + 'experimental.')
            m = utils.BFGS(m,j,S,M,rho0,params,limii=numparams.map_iter_a)
        
        mlist.append(m)   
        save_results(m.val, "map, iter #" + str(git), \
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
                #bare=True?
                logger.message('Calculating Dhat.')
                D_hathat = D.hathat(domain=k_space,\
                    ncpu=numparams.ncpu_a,nrun=numparams.nrun_a)
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
            save_results(kindex,"ps, iter #" + str(git), \
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
            if np.max(np.abs(np.log(pspec)/np.log(S.get_power())))\
                < np.log(1e-1):
                    logger.message('Power spectrum converged.')
                    convergence += 1
        
        #global convergence test
        if convergence >= params.finalconvlevel:
            logger.message('Global convergence achieved at iteration' + \
                str(git) + '.')
            if params.uncertainty_a=='supp':
                logger.message('Calculating alpha uncertainty as requested.')
                D_hat = D.hat(domain=s_space,\
                ncpu=numparams.ncpu_a,nrun=numparams.nrun_a)
                save_results(D_hat.val,"relative uncertainty", \
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
    
    if params.freq == 'wideband':
        aconst = args[0]
        wideband_git = args[1]
         
    # Sets the alpha prior parameter for all modes
    if not numparams.alpha_prior is None:
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
        numparams.M0_start, rho0, params, numparams])

    
    # diagnostic plots

    if params.freq == 'wideband':
        save_results(j,"j",'resolve_output_' + str(params.save) +\
            '/general/' + params.save + \
            str(wideband_git) + '_j',rho0 = params.rho0)
    else:
        save_results(j,"j",'resolve_output_' + str(params.save) +\
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
            save_results(D_hat.val,"relative uncertainty", \
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
            save_results(m.val, "map, iter #" + str(git), \
                'resolve_output_' + str(params.save) +\
                '/m_reconstructions/' + params.save + "_m_WF" +\
                str(wideband_git) + "_" + str(git), rho0 = params.rho0)
            write_output_to_fits(np.transpose(m.val*params.rho0),params, \
                notifier = str(wideband_git) + "_" + str(git),mode = 'I')
        else:
            save_results(m.val, "map, iter #" + str(git), \
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
                save_results(kindex,"ps, iter #" + str(git), \
                    'resolve_output_' + str(params.save) + \
                    "/p_reconstructions/" + params.save + "_p" + \
                    str(wideband_git) + "_" + str(git), value2=pspec,\
                    log='loglog')
            else:
                save_results(kindex,"ps, iter #" + str(git), \
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
        if np.max(np.abs(np.log(pspec)/np.log(S.get_power()))) < np.log(1e-1):
            logger.message('Power spectrum converged.')
            convergence += 1
        
        #global convergence test
        if convergence >= numparams.finalconvlevel:
            logger.message('Global convergence achieved at iteration' + \
                str(git) + '.')
            if params.uncertainty=='supp':
                logger.message('Calculating uncertainty map at end of'\
                    +'recpnstruction, as requested.')
                D_hat = D.hat(domain=s_space,\
                ncpu=numparams.ncpu,nrun=numparams.nrun)
                save_results(D_hat.val,"relative uncertainty", \
                    'resolve_output_' + str(params.save) +\
                    "/D_reconstructions/" + params.save + "_D")
                
                
            return m, pspec

        git += 1

    return m, pspec    
        
#-------------------parameter classes & code I/O-------------------------------
        
class parameters(object):
    """
    Defines a parameter class for all parameters that are vital to controlling
    basic functionality of the code. Some are even mandatory and given by 
    the user. Performs checks on default arguments with hard-coded default
    values.
    """

    def __init__(parset, save, casaload, verbosity):

        #mandatory and outside-code parameters
        self.ms = parset['ms']
        self.imsize = parset['imsize']
        self.cellsize = parset['cellsize']
        self.save = save
        self.casaload = casaload
        self.verbosity = verbosity
        
        #main parameters that often need to be set by the user but have default
        #values
        self.check_default(algorithm,'ln-map')
        self.check_default(stokes,'I')
        self.check_default(ftmode,'gfft')
        if self.ftmode == 'wsclean' and wsclean_available == False:
            self.ftmode == 'gfft'
            print 'Wsclean is not available. Gridding routines are set to '\
                'default despite different user input'
        self.check_default(init_type_s,'const')
        self.check_default(init_type_u,'const')
        self.check_default(rho0,1)
        self.check_default(freq,[0,0])
        self.check_default(pspec,True)
        self.check_default(pbeam,False)
        self.check_default(uncertainty,False)
        self.check_default(simulating,False)
        self.check_default(noise_est,False)
        self.check_default(noise_update,False)
        self.check_default(map_conv,1e-1)
        self.check_default(callback,3)
        if self.pspec:
            self.check_default(init_type_p,'k^2_mon')
            self.check_default(pspec_conv,1e-1)
        if self.freq == 'wideband':
            self.check_default(init_type_a,'const')
            self.check_default(reffreq,'const')
            self.check_default(pspec_a,True)
            self.check_sefault(uncertainty_a,False)
            if self.pspec_a:
                self.check_default(init_type_a_p,'k^2_mon')
        
        
        # a few global parameters needed for callbackfunc
        global gcallback
        global gsave
        gcallback = callback
        gsave = save
        
    def check_default(self, parameter, parset, default):
        
        if str(parameter) in parset:
            self.parameter = parset[str(parameter)]
        else:
            self.parameter = default

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
        
        self.check_default(m_start,0.1)
        self.check_default(global_iter,50)        
        self.check_default(alpha_prior,None)
        self.check_default(map_alpha,1e-4)
        self.check_default(map_tol,1e-5)
        self.check_default(map_clevel,3)
        self.check_default(map_iter,100)
        self.check_default(final_convlevel,4)
        if params.init_type_s == '????':
            self.check_default(zoomfactor,1)
        
        if params.pspec:
            
            self.check_default(smoothing,10)
            self.check_default(M0_start,0)
            self.check_default(bins,70)
            self.check_default(p0,1)
            self.check_default(pspec_tol,1e-3)
            self.check_default(pspec_clevel,3)
            self.check_default(pspec_iter,150)
            self.check_default(ncpu,2)
            self.check_default(nrun,8)

        if params.freq == 'wideband':
            
            self.check_default(wb_globiter,10)
            self.check_default(m_a_start,0.1)
            self.check_default(global_iter_a,50)
            self.check_default(alpha_prior_a,None)
            self.check_default(map_alpha_a,1e-4)
            self.check_default(map_tol_a,1e-5)
            self.check_default(map_clevel_a,3)
            self.check_default(map_iter_a,100)
            if params.init_type_s_a == '????':
                self.check_default(zoomfactor,1)
            
            if params.pspec_a:
                
                self.check_default(smoothing_a,10)
                self.check_default(M0_start_a,0)
                self.check_default(bins_a,70)
                self.check_default(p0_a,1)
                self.check_default(pspec_tol_a,1e-3)
                self.check_default(pspec_clevel_a,3)
                self.check_default(pspec_iter_a,150)
                self.check_default(ncpu_a,2)
                self.check_default(nrun_a,8)
                
        if params.algorithm == 'ln-map_u':
            
            self.check_default(m_u_start,0.1)
            self.check_default(beta,1.5)
            self.check_default(eta,1e-7)
            self.check_default(map_alpha_u,1e-4)
            self.check_default(map_tol_u,1e-5)
            self.check_default(map_clevel_u,3)
            self.check_default(map_iter_u,100)
        

    def check_default(self, parameter, parset, default):
        
        if str(parameter) in parset:
            self.parameter = parset[str(parameter)]
        else:
            self.parameter = default
         
        


def parse_input_file(parsetfn, save, casaload, verbosity):
    """ parse the parameter file."""

    reader = csv.reader(open(parsetfn, 'rb'), delimiter=" ",
                        skipinitialspace=True)
    parset = dict()

    for row in reader:
        if len(row) != 0 and row[0] != '%':
            parset[row[0]] = row[1]


    params = parameters(parset, save, casaload, verbosity)       
    
    numparams = numparameters(params, parset)

    return params, numparams


def write_output_to_fits(m, params, notifier='',mode='I',u=None):
    """
    """

    hdu_main = pyfits.PrimaryHDU(convert_RES_to_CASA(m,FITS=True))
    
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
                        help="output string for save directories")
    parser.add_argument("-c","--casaload", type=bool, default=False,
                        help="attempt to read data using CASA")
    parser.add_argument("-v","--verbosity", type=int, default=2,
                        help="verbosity level of code output")                    
    args = parser.parse_args()
    
    #this is done for greater customization and control of file parsing than
    #doing it directly with a file-type in argparse 
    params, numparams = parse_input_file(args.parsetfn, args.save, \
        args.casaload, args.verbosity)
    
    resolve(params, numparams)
