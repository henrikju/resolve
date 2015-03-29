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

#import necessary modules
import pylab as pl
import numpy as np
from nifty import *
import Messenger as M
import general_response as r
from general_IO import read_data_from_ms
from nifty import nifty_tools as nt
from casa import ms as mst
from casa import image as ia 
import casa
import pyfits
import sys
import datetime


#a few global constants
q = 1e-15
C = 299792458
asec2rad = 4.84813681e-6



def resolve(ms, imsize, cellsize, algorithm = 'ln-map', init_type_s = 'dirty',\
    use_init_s = False, init_type_p = 'k-2_mon', init_type_p_a = 'k-2_mon',\
    lastit = None, freq = [0,0] , pbeam = None, uncertainty = False, \
    noise_est = None, map_algo = 'sd', pspec_algo = 'cg', barea = 1, \
    map_conv = 1e-1, pspec_conv = 1e-1, save = None, callback = 3, \
    plot = False, simulating = False, restfreq = [0,0], use_parset = False,
    **kwargs):

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
            init_type_s: What input should be used to fix the monopole of the map
                by effectively estimating rho_0.
                1) 'dirty'
                2) 'user-defined'
            use_init_s: Whether to use the init_type_s as a starting guess \
                (default is False and uses a constant map close to zero).
            init_type_p: Starting guess for the power spectrum.
                1) 'k^2': Simple k^2 power spectrum.
                2) 'k^2_mon': Simple k^2 power spectrum with fixed monopole.
                3) 'zero': Zero power spectrum.
            init_type_p_a: Starting guess for the power spectrum.
                1) 'k^2': Simple k^2 power spectrum.
                2) 'k^2_mon': Simple k^2 power spectrum with fixed monopole.
                3) 'zero': Zero power spectrum.
            lastit: Integer n or None. Whether to start with iteration n.
            freq: Whether to perform single band or wide band RESOLVE.
                1) [spw,cha]: single band
                2) 'wideband'
            pbeam: user-povided primary beam pattern.
            uncertainty: Whether to attempt calculating an uncertainty map \
                (EXPENSIVE!).
            noise_est: Whether to take the measure noise variances or make an \
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
    
    # Define parameter class
    
    if use_parset == True:
        params, numparams = parse_input_file('resolve_parset')

    else:
        params = parameters(ms, imsize, cellsize, algorithm, init_type_s, \
                        use_init_s, init_type_p, init_type_p_a, lastit, freq, \
                        pbeam, uncertainty, noise_est, map_algo, pspec_algo, \
                        barea, map_conv, pspec_conv, save, callback, \
                        plot, simulating, restfreq)
                        
        numparams = numparameters(params, kwargs)
    
    if simulating:
        
        simparams = simparameters(params, kwargs)
                                                    
    # Prepare a number of diagnostics if requested
    if params.plot:
        pl.ion()
    else:
        pl.ioff()
    if params.save:
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
            '/p_reconstructions'):
                os.makedirs('resolve_output_' + str(params.save)+\
                '/p_reconstructions')
        if not os.path.exists('resolve_output_' + str(params.save)+\
            '/D_reconstructions'):
                os.makedirs('resolve_output_' + str(params.save)+\
                '/D_reconstructions')
        logfile = 'resolve_output_'+str(params.save)+'/general/' + params.save + '.log'
    else:
        logfile = None
    logger = M.Messenger(verbosity=2, add_timestamp=False, logfile=logfile)

    logger.header1('Starting Bayesian total intensity reconstruction.')

    # data setup
    if simulating:
        d, N, R, di, d_space, s_space = simulate(params, simparams, logger)
    else:
        d, N, R, di, d_space, s_space = datasetup(params, logger)
    
    # Begin: Starting guesses for m *******************************************

    # estimate rho0, the constant part of the lognormal field rho = rho0 exp(s)
    # effectively sets a different prior mean so that the field doesn't go to 
    # zero in first iterations     
    if init_type_s == 'dirty':
        mtemp = field(s_space, target=s_space.get_codomain(), val=di)

    else:
        # Read-in userimage and convert to Jy/px
        mtemp = field(s_space, target=s_space.get_codomain(), \
                      val=np.load('userimage.npy')/params.barea)

    
    rho0 = np.mean(mtemp.val[np.where(mtemp.val>=np.max(mtemp.val) / 4)])
    logger.message('rho0: ' + str(rho0))
    if rho0 < 0:
        logger.warn('Monopole level of starting guess negative. Probably due \
            to too many imaging artifcts in userimage')
        
    # Starting guess for m, either constant close to zero, or lastit from
    # a file with save-basis-string 'save', or directly from the user
    if lastit == None:
        m_I = field(s_space, val = numparams.m_start)
        if freq == 'wideband':
            m_a = field(s_space, val = numparams.m_a_start)
    elif lastit != None:
        logger.message('using last m-iteration from previous run.')
        if freq == 'wideband':
            m_I = field(s_space, val = np.load(params.save + str(lastit) + \
            "_m.npy"))
            m_a = field(s_space, val = np.load(params.save + str(lastit) + \
            "_m_a.npy"))
        else:
            m_I = field(s_space, val = np.load(params.save + str(lastit) + \
            "_m.npy"))
    elif use_init_s:
        m_I = field(s_space, val = np.log(np.abs(mtemp)))

    # Begin: Starting guesses for pspec ***************************************

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
    # zero power spectrum guess 
    elif params.init_type_p == 'zero':
        pspec = 0
    # power spectrum from last iteration 
    elif lastit != None:
        logger.message('using last p-iteration from previous run.')
        pspec = np.load(params.save + str(lastit) + "_p.npy")
    # default simple k^2 spectrum with free monopole    
    else:
        pspec = np.array((1+kindex)**-2 * numparams.p0)
        
    if freq == 'wideband':
        # zero power spectrum guess 
        if init_type_p_a == 'zero':
            pspec_a = 0
        # power spectrum from last iteration 
        elif lastit != None:
            logger.message('using last p-iteration from previous run.')
            pspec_a = np.load(params.save + str(lastit) + "_p.npy")
        # default simple k^2 spectrum with free monopole    
        else:
            pspec_a = np.array((1+kindex)**-2 * numparams.p0_a) 
 
    # diagnostic plot of m starting guess
    if params.save:
        save_results(np.exp(m_I.val),"TI exp(Starting guess)",\
            'resolve_output_' + str(params.save) + '/m_reconstructions/' +\
            params.save + "_expm0", rho0 = rho0)
        write_output_to_fits(np.transpose(np.exp(m_I.val)/rho0),params, notifier='0', mode='I')
        if freq == 'wideband':
           save_results(m_a.val,"Alpha Starting guess",\
            'resolve_output_' + str(params.save) + '/m_reconstructions/' +\
            params.save + "_ma0", rho0 = rho0) 
           write_output_to_fits(np.transpose(m_I.val/rho0),params, notifier='0', mode='a') 
           
    # Begin: Start Filter *****************************************************

    if params.algorithm == 'ln-map':
        
        logger.header1('Starting RESOLVE.')
        
        if params.freq is not 'wideband':
            #single-band I-Filter
            t1 = time()
            m_I, p_I = mapfilter_I(d, m_I, pspec, N, R, logger, rho0, k_space, \
                params, numparams)
            t2 = time()
            
        else:
            #wide-band I/alpha-Filter
            t1 = time()
            wideband_git = 0
            while(wideband_git < params.wb_globiter):

                #I-Filter                                                                                                                                                                              
                m_I, p_I, = mapfilter_I(d, m_I, pspec, N, R, logger, rho0,\
                k_space, params, numparams, m_a, wideband_git)

                #a-Filter                                                                                                                                                                              
                m_a, p_a, = mapfilter_a(d, m_a, pspec_a, N, R, logger, rho0,\
                k_space, params, numparams, m_I, wideband_git)


                wideband_git += 1

            t2 = time()
            
    elif params.algorithm == 'wf':
        
        logger.failure('Wiener Filter not yet implemented')
        
    elif params.algorithm =='gibbsenergy':
        
        logger.failure('Gibbs energy filter not yet implemented')
        
    elif params.algorithm == 'sampling':
        
        logger.failure('Sampling algorithm not yet implemented')
        
        

    logger.success("Completed algorithm.")
    logger.message("Time to complete: " + str((t2 - t1) / 3600.) + ' hours.')

    # Begin: Some plotting stuff **********************************************

    if params.save:
        
        save_results(np.exp(m_I.val),"exp(Solution m)",\
            'resolve_output_' + str(params.save) + '/m_reconstructions/' + \
            params.save + "_expmfinal", rho0 = rho0)
        write_output_to_fits(np.transpose(np.exp(m_I.val)/rho0),params, notifier='final',mode='I')
        
        if params.freq == 'wideband':
            save_results(m_a.val,"Solution a",\
                'resolve_output_' + str(params.save) + '/m_reconstructions/' +\
                params.save + "_mafinal", rho0 = rho0)
            write_output_to_fits(np.transpose(m_a.val),params, notifier ='final', mode='a')
            
        save_results(kindex,"final power spectrum",\
                     'resolve_output_' + str(params.save) + \
                     '/p_reconstructions/' + params.save + "_powfinal", \
                     value2 = p_I, log='loglog')
        

    # ^^^ End: Some plotting stuff *****************************************^^^

    return m_I, p_I

#------------------------------------------------------------------------------


def datasetup(params, logger):
    """
        
    Data setup, feasible for the nifty framework.
        
    """
    
    logger.header2("Running data setup.")

    if params.noise_est == 'simple':
        vis, sigma, u, v, freqs, nchan, nspw, nvis, summary = \
            read_data_from_ms(params.ms, noise_est = True)
    else:
        vis, sigma, u, v, freqs, nchan, nspw, nvis, summary = \
            read_data_from_ms(params.ms)
    
    params.summary = summary
    
    # single-band imaging
    if not params.freq == 'wideband':
        sspw,schan = params.freq[0], params.freq[1]
        vis = vis[sspw][schan]
        sigma = sigma[sspw][schan]
        u = u[sspw][schan]
        v = v[sspw][schan]
        
    # noise estimation via ecf   
    if params.noise_est == 'ecf':
        logger.message('ECF noise estimate not yet implemented')
    
    # wideband imaging settings                                                     
    if params.freq == 'wideband':
        variance = (np.array(sigma)**2).flatten()
        u = np.array(u)
        v = np.array(v)
        freqs = np.array(freqs)
        nspw = nspw[0]+1
        nchan = nchan[0]
        nvis = nvis[0]

        
    else:
        variance = sigma**2
    # bad fix of possible problems in noise estimation
    variance[variance<1e-10] = np.mean(variance[variance>1e-10])
    
    # definition of wideband data operators
    if params.freq == 'wideband':

        # basic diagnostics                                                         
        if params.save:
            #maximum k-mode and resolution of data
            uflat = u.flatten()
            vflat = v.flatten()
            uvrange = np.array([np.sqrt(uflat[i]**2 + vflat[i]**2) for i in range(len(u))])
            dx_real_rad = (np.max(uvrange))**-1
            logger.message('resolution\n' + 'rad ' + str(dx_real_rad) + '\n' + \
                           'asec ' + str(dx_real_rad/asec2rad))

            save_results(uflat,'UV', 'resolve_output_' + str(params.save) + \
                '/general/' + params.save + "_uvcov", \
                plotpar='o', value2 = vflat)
        
        d_space = point_space(nspw*nchan*nvis, datatype = np.complex128)
        s_space = rg_space(params.imsize, naxes=2, dist = params.cellsize,\
            zerocenter=True)
        
        # primary beam function
        if not params.pbeam is None:
            logger.message('No wideband primary beam implemented.'+
                'Automatically set to one.')
            A = np.array([[np.ones((params.imsize,params.imsize))\
                for k in range(nchan)] for j in range(nspw)])
        else:
            A = np.array([[np.ones((params.imsize,params.imsize))\
                for k in range(nchan)] for j in range(nspw)])
            
        # response operator
        R = r.response_mfs(s_space, sym=False, imp=True, target=d_space, \
                           para=[u,v,A,nspw,nchan,False,freqs,\
                           params.restfreq[0],params.restfreq[1],nvis])
    
        d = field(d_space, val=np.array(vis).flatten())

        N = N_operator(domain=d_space,imp=True,para=[variance])
        
        # dirty image from CASA for comparison
        di = make_dirtyimage(params, logger)

        # more diagnostics if requested
        if params.save:
            # plot the dirty beam
            uvcov = field(d_space,val=np.ones(np.shape(np.array(vis).flatten()), dtype = np.complex128))
            
            db = R.adjoint_times(uvcov) * s_space.vol[0] * s_space.vol[1]       
            save_results(db,"dirty beam",'resolve_output_' + str(params.save)+\
                '/general/' + params.save + "_db")
            

            # plot the dirty image
            save_results(di,"dirty image",'resolve_output_' +str(params.save)+\
                '/general/' + params.save + "_di")


        return  d, N, R, di, d_space, s_space
        
    # definition of single band data operators
    else:

        # basic diagnostics                                                         
        if params.save:
            # maximum k-mode and resolution of data                                      
            uvrange = np.array([np.sqrt(u[i]**2 + v[i]**2) for i in range(len(u))])
            dx_real_rad = (np.max(uvrange))**-1
            logger.message('resolution\n' + 'rad ' + str(dx_real_rad) + '\n'+ \
                           'asec ' + str(dx_real_rad/asec2rad))

            save_results(u,'UV', 'resolve_output_' + str(params.save) +\
                '/general/' + params.save + "_uvcov", plotpar='o', value2 = v)

        d_space = point_space(len(u), datatype = np.complex128)
        s_space = rg_space(params.imsize, naxes=2, dist = params.cellsize, \
            zerocenter=True)
        
        # primary beam function
        if not params.pbeam is None:
            logger.message('Calculating VLA beam from Perley fit')
            A = make_primbeam_aips(s_space.dim(split = True)[0], \
                s_space.dim(split = True)[1] , s_space.dist()[0], s_space.dist()[1], \
                1.369)
        else:
            A = 1.
        # response operator
        R = r.response(s_space, sym=False, imp=True, target=d_space, \
                       para=[u,v,A,False])
    
        d = field(d_space, val=vis)

        N = N_operator(domain=d_space,imp=True,para=[variance])
        
        # dirty image from CASA for comparison
        di = make_dirtyimage(params, logger)

        # more diagnostics if requested
        if params.save:
            # plot the dirty beam
            uvcov = field(d_space,val=np.ones(len(u), dtype = np.complex128))
            db = R.adjoint_times(uvcov) * s_space.vol[0] * s_space.vol[1]       
            save_results(db,"dirty beam", 'resolve_output_' +str(params.save)+\
                '/general/' + params.save + "_db")
            
            # plot the dirty image
            save_results(di,"dirty image",'resolve_output_' +str(params.save)+\
                '/general/' + params.save + "_di")

        return  d, N, R, di, d_space, s_space
    


def mapfilter_I(d, m, pspec, N, R, logger, rho0, k_space, params, numparams,\
    *args):
    """
    """

    logger.header1("Begin total intensity filtering")  
      
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

    if params.save:
        if params.freq == 'wideband':
            save_results(j,"j",'resolve_output_' + str(params.save) +\
                '/general/' + params.save + \
                str(wideband_git) + '_j',rho0 = rho0)
        else:
            save_results(j,"j",'resolve_output_' + str(params.save) +\
                '/general/' + params.save + 'j',rho0 = rho0)

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
        args = (j, S, M, rho0)
        D.para = [S, M, m, j, numparams.M0_start, rho0, params, numparams]

        #run nifty minimizer steepest descent class
        logger.header2("Computing the MAP estimate.\n")

        mold = m
        
        if params.map_algo == 'sd':
            en = energy(args)
            minimize = nt.steepest_descent(en.egg,spam=callbackfunc,note=True)
            m = minimize(x0=m, alpha=numparams.map_alpha, \
                tol=numparams.map_tol, clevel=numparams.map_clevel, \
                limii=numparams.map_iter)[0]
        elif params.map_algo == 'lbfgs':
            logger.failure('lbfgs algorithm not yet implemented.')
           
        if params.save:
            if params.freq == 'wideband':
                save_results(exp(m.val), "map, iter #" + str(git), \
                    'resolve_output_' + str(params.save) +\
                    '/m_reconstructions/' + params.save + "_expm" +\
                    str(wideband_git) + "_" + str(git), rho0 = rho0)
                write_output_to_fits(np.transpose(exp(m.val)/rho0),params, \
                    notifier = str(wideband_git) + "_" + str(git),mode = 'I')
            else:
                save_results(exp(m.val), "map, iter #" + str(git), \
                    'resolve_output_' + str(params.save) +\
                    '/m_reconstructions/' + params.save + "_expm" + str(git), \
                    rho0 = rho0)
                write_output_to_fits(np.transpose(exp(m.val)/rho0),params,\
                notifier = str(git), mode='I')

        logger.header2("Computing the power spectrum.\n")

        #extra loop to take care of possible nans in PS calculation
        psloop = True
        M0 = numparams.M0_start
        while psloop:
            
            D.para = [S, M, m, j, M0, rho0, params, numparams]
            
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
                D.para = [S, M, m, j, M0, rho0, params, numparams]
            else:
                psloop = False
            
        logger.message("    Current M0:  " + str(D.para[4])+ '\n.')


        mlist.append(m)
        plist.append(pspec)

        
        if params.save:
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
        if convergence >= 4:
            logger.message('Global convergence achieved at iteration' + \
                str(git) + '.')
            if params.uncertainty:
                logger.message('Calculating uncertainty map as requested.')
                D_hat = D.hat(domain=s_space,\
                ncpu=numparams.ncpu,nrun=numparams.nrun)
                save_results(D_hat.val,"relative uncertainty", \
                    'resolve_output_' + str(params.save) +\
                    "/D_reconstructions/" + params.save + "_D")
                
                
            return m, pspec

        git += 1
    
    if params.uncertainty:
        logger.message('Calculating uncertainty map as requested.')
        D_hat = D.hat(domain=s_space,\
        ncpu=numparams.ncpu,nrun=numparams.nrun)
        save_results(D_hat.val,"relative uncertainty", \
            'resolve_output_' + str(params.save) +\
            "/D_reconstructions/" + params.save + "_D")

    return m, pspec


def mapfilter_a(d, m, pspec, N, R, logger, rho0, k_space, params, numparams,\
    m_I, wideband_git):
    """
    """

    logger.header1("Begin spectral index filtering")  
      
    s_space = m.domain
    kindex = k_space.get_power_indices()[0]
    
    mconst = m_I
         
    # Sets the alpha prior parameter for all modes
    if not numparams.alpha_prior_a is None:
        alpha = numparams.alpha_prior_a
    else:
        alpha = np.ones(np.shape(kindex))
        
    # Defines important operators. M and j are defined implicitly in energy_a
    S = power_operator(k_space, spec=pspec, bare=True)
    D = Da_operator(domain=s_space, sym=True, imp=True, para=[S, R, N, m,\
            numparams.M0_start_a, mconst, d, rho0, params, numparams])
    

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
        D.para = [S, R, N, m, numparams.M0_start_a, rho0, mconst, d,\
            params, numparams]

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
            logger.failure('lbfgs algorithm not yet implemented.')
           
        if params.save:
            save_results(m.val, "map, iter #" + str(git), \
                'resolve_output_' + str(params.save) +\
                "/m_reconstructions/" + params.save + "_ma" + \
                str(wideband_git) + "_" + str(git), rho0 = rho0)
            write_output_to_fits(np.transpose(m.val),params, \
            notifier = str(wideband_git) + "_" + str(git), mode='a')
        

        logger.header2("Computing the power spectrum.\n")
        

        #extra loop to take care of possible nans in PS calculation
        psloop = True
        M0 = numparams.M0_start_a
        while psloop:
            
            D.para = [S, R, N, m, numparams.M0_start_a, rho0, mconst, d,\
            params, numparams]
            
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
                logger.message('D not positive definite, try increasing eta.')
                if M0 == 0:
                    M0 += 0.1
                M0 *= 1e6
                D.para = [S, M, m, j, M0, rho0, params, numparams]
            else:
                psloop = False
            
        logger.message("    Current M0:  " + str(D.para[4])+ '\n.')


        mlist.append(m)
        plist.append(pspec)

        
        if params.save:
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
        if np.max(np.abs(np.log(pspec)/np.log(S.get_power()))) < np.log(1e-1):
            logger.message('Power spectrum converged.')
            convergence += 1
        
        #global convergence test
        if convergence >= 4:
            logger.message('Global convergence achieved at iteration' + \
                str(git) + '.')
            if params.uncertainty:
                logger.message('Calculating uncertainty map as requested.')
                D_hat = D.hat(domain=s_space,\
                ncpu=numparams.ncpu_a,nrun=numparams.nrun_a)
                save_results(D_hat.val,"relative uncertainty", \
                    'resolve_output_' + str(params.save) +\
                    "/D_reconstructions/" + params.save + "_Da")
                
            return m, pspec

        git += 1
    
    if params.uncertainty:
        logger.message('Calculating uncertainty map as requested.')
        D_hat = D.hat(domain=s_space,\
        ncpu=numparams.ncpu_a,nrun=numparams.nrun_a)
        save_results(D_hat.val,"relative uncertainty", \
            'resolve_output_' + str(params.save) + \
            "/D_reconstructions/" + params.save + "_Da")

    return m, pspec

    
#------------------------------------------------------------------------------


class N_operator(operator):
    """
    Wrapper around a standard Noise operator. Handles radio astronomical flags.
    """
    
    def _multiply(self, x):
        
        variance = self.para[0]
        
        mask = variance>0
        variance[variance==0] = 1.

        Ntemp = diagonal_operator(domain=self.domain, diag = variance)

        return mask * Ntemp(x)

    def _inverse_multiply(self, x):
        
        variance = self.para[0]
        
        mask = variance>0
        variance[variance==0] = 1.

        Ntemp = diagonal_operator(domain=self.domain, diag = variance)

        return mask * Ntemp.inverse_times(x)
        
#-------------------------single band operators--------------------------------
        
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
        numparams = self.para[7]
        params = self.para[6]

        if params.pspec_algo == 'cg':
            x_,convergence = nt.conjugate_gradient(self._matvec, x, \
                note=True)(tol=numparams.pspec_tol,clevel=numparams.pspec_clevel,\
                limii=numparams.pspec_iter)
                
        elif params.pspec_algo == 'sd':
            x_,convergence = nt.steepest_descent(self._matvec, x, \
                note=True)(tol=numparams.pspec_tol,clevel=numparams.pspec_clevel,\
                limii=numparams.pspec_iter)
                    
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



class energy(object):
    
    def __init__(self, args):
        self.j = args[0]
        self.S = args[1]
        self.M = args[2]
        self.A = args[3]
        
    def H(self,x):
        """
        """

        part1 = x.dot(self.S.inverse_times(x.weight(power = 0)))  / 2
        part2 = self.j.dot(self.A * exp(x))
        
        part3 = self.A * exp(x).dot(self.M(self.A * exp(x))) / 2
        
        
        return part1 - part2 + part3
    
    def gradH(self, x):
        """
        """
#        print 'x', x.domain
#        print 'j', self.j.domain
#        print 'M', self.M.domain
#        print 'S', self.S.domain.get_codomain()
#        print 'codo', self.S.domain.get_codomain()
#        
#        x = field(self.S.domain.get_codomain(), val=x)
    
        temp1 = self.S.inverse_times(x)
        #temp1 = temp1.weight(power=2)
        temp = -self.j * self.A * exp(x) + self.A* exp(x) * \
            self.M(self.A * exp(x)) + temp1
    
        return temp
    
    def egg(self, x):
        
        E = self.H(x)
        g = self.gradH(x)
        
        return E,g

#-----------------------wide band operators------------------------------------

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
        numparams = self.para[9]
        params = self.para[8]

        if params.pspec_algo == 'cg':
            x_,convergence = nt.conjugate_gradient(self._matvec, x, \
                note=True)(tol=numparams.pspec_tol_a,clevel=numparams.pspec_clevel_a,\
                limii=numparams.pspec_iter_a)
                
        elif params.pspec_algo == 'sd':
            x_,convergence = nt.steepest_descent(self._matvec, x, \
                note=True)(tol=numparams.pspec_tol_a,clevel=numparams.pspec_clevel_a,\
                limii=numparams.pspec_iter_a)
                    
        return x_
        
class energy_a(object):
    
    def __init__(self, args):
        self.d = args[0]
        self.S = args[1]
        self.N = args[2]
        self.R = args[3]
        self.m_I = args[4]
        
    def Ha(self,a):
        """
        """

        part1 = a.dot(self.S.inverse_times(a))/2

        part2 = self.R.adjoint_times(self.N.inverse_times(self.d), a = a).dot(\
            exp(self.m_I))

        Mexp = self.R.adjoint_times(self.N.inverse_times(\
            self.R(exp(self.m_I), a = a)), a = a)

        part3 = (exp(self.m_I)).dot(Mexp)/2.

        return part1 - part2 + part3
    
    def gradHa(self, a):
        """
        """

        temp = - self.R.adjoint_times(self.N.inverse_times(self.d), a = a, \
            mode = 'grad') * exp(self.m_I) + exp(self.m_I) * \
            self.R.adjoint_times(self.N.inverse_times(self.R.times( \
            exp(self.m_I), a = a)), a = a, mode = 'grad') + \
            self.S.inverse_times(a)

        return temp
    
    def egg(self, x):
        
        E = self.Ha(x)
        g = self.gradHa(x)
        
        return E,g

        
#-------------------------parameter classes------------------------------------        
        
class parameters(object):

    def __init__(self,ms, imsize, cellsize, algorithm, init_type_s, \
                 use_init_s, init_type_p, init_type_p_a, lastit, freq, pbeam, \
                 uncertainty, noise_est, map_algo, pspec_algo, barea,\
                 map_conv, pspec_conv, save, callback, plot, simulating,\
                 restfreq):

        self.ms = ms
        self.imsize = imsize
        self.cellsize = cellsize
        self.algorithm = algorithm
        self.init_type_s = init_type_s
        self.use_init_s = use_init_s
        self.init_type_p = init_type_p
        self.init_type_p_a = init_type_p_a
        self.lastit = lastit
        self.freq = freq
        self.pbeam = pbeam
        self.uncertainty = uncertainty
        self.noise_est = noise_est
        self.map_algo = map_algo
        self.pspec_algo = pspec_algo
        self.barea = barea
        self.save = save
        self.map_conv = map_conv
        self.pspec_conv = pspec_conv
        self.callback = callback
        self.plot = plot
        self.simulating = simulating
        self.restfreq = restfreq
        
        
        # a few global parameters needed for callbackfunc
        global gcallback
        global gsave
        gcallback = callback
        gsave = save

class numparameters(object):
    
    def __init__(self, params, kwargs):
        
        if 'm_start' in kwargs:
            self.m_start = kwargs['m_start']
        else: 
            self.m_start = 0.1
        if 'ma_start' in kwargs:
            self.m_a_start = kwargs['ma_start']
        else:
            if params.freq == 'wideband':
                self.m_a_start = 0.1
            
        if 'global_iter' in kwargs:
            self.global_iter = kwargs['global_iter']
        else: 
            self.global_iter = 50
        if 'global_iter_a' in kwargs:
            self.global_iter_a = kwargs['global_iter_a']
        else: 
            if params.freq == 'wideband':
                self.global_iter_a = 50
            
        if 'M0_start' in kwargs:
            self.M0_start = kwargs['M0_start']
        else: 
            self.M0_start = 0
        if 'M0_start_a' in kwargs:
            self.M0_start_a = kwargs['M0_start_a']
        else: 
            if params.freq == 'wideband':
                self.M0_start_a = 0
            
        if 'bins' in kwargs:
           self.bin = kwargs['bins']
        else:
           self.bin = 70
        if 'bins_a' in kwargs:
           self.bin_a = kwargs['bins_a']
        else:
           if params.freq == 'wideband':
                self.bin_a = 70
           
        if 'p0' in kwargs:
           self.p0 = kwargs['p0']
        else:
           self.p0 = 1
        if 'p0_a' in kwargs:
           self.p0_a = kwargs['p0_a']
        else:
           if params.freq == 'wideband':
                self.p0_a = 1.
           
        if 'alpha_prior' in kwargs:
           self.alpha_prior = kwargs['alpha_prior']
        else:
           self.alpha_prior = None
        if 'alpha_prior_a' in kwargs:
           self.alpha_prior_a = kwargs['alpha_prior_a']
        else:
           if params.freq == 'wideband':
                self.alpha_prior_a = None
           
        if 'smoothing' in kwargs:
           self.smoothing = kwargs['smoothing']
        else:
           self.smoothing = 10.
        if 'smoothing_a' in kwargs:
           self.smoothing_a = kwargs['smoothing_a']
        else:
           if params.freq == 'wideband':
                self.smoothing_a = 10.
               
        if 'pspec_tol' in kwargs:
           self.pspec_tol = kwargs['pspec_tol']
        else:
           self.pspec_tol = 1e-3
        if 'pspec_tol_a' in kwargs:
           self.pspec_tol_a = kwargs['pspec_tol_a']
        else:
           if params.freq == 'wideband':
                self.pspec_tol_a = 1e-3
           
        if 'pspec_clevel' in kwargs:
           self.pspec_clevel = kwargs['pspec_clevel']
        else:
           self.pspec_clevel = 3
        if 'pspec_clevel_a' in kwargs:
           self.pspec_clevel_a = kwargs['pspec_clevel_a']
        else:
           if params.freq == 'wideband':
                self.pspec_clevel_a = 3
        
        if 'pspec_iter' in kwargs:
           self.pspec_iter = kwargs['pspec_iter']
        else:
           self.pspec_iter = 150
        if 'pspec_iter_a' in kwargs:
           self.pspec_iter_a = kwargs['pspec_iter_a']
        else:
           if params.freq == 'wideband':
                self.pspec_iter = 150
           
        if 'map_alpha' in kwargs:
           self.map_alpha = kwargs['map_alpha']
        else:
           self.map_alpha = 1e-4
        if 'map_alpha_a' in kwargs:
           self.map_alpha_a = kwargs['map_alpha_a']
        else:
           if params.freq == 'wideband':
                self.map_alpha_a = 1e-4
           
        if 'map_tol' in kwargs:
           self.map_tol = kwargs['map_tol']
        else:
           self.map_tol = 1e-5
        if 'map_tol_a' in kwargs:
           self.map_tol_a = kwargs['map_tol_a']
        else:
           if params.freq == 'wideband':
                self.map_tol_a = 1e-5
           
        if 'map_clevel' in kwargs:
           self.map_clevel = kwargs['map_clevel']
        else:
           self.map_clevel = 3
        if 'map_clevel_a' in kwargs:
           self.map_clevel_a = kwargs['map_clevel_a']
        else:
           if params.freq == 'wideband':
                self.map_clevel_a = 3
           
        if 'map_iter' in kwargs:
           self.map_iter = kwargs['map_iter']
        else:
           self.map_iter = 100
        if 'map_iter_a' in kwargs:
           self.map_iter_a = kwargs['map_iter_a']
        else:
           if params.freq == 'wideband':
                self.map_iter_a = 100
           
        if 'ncpu' in kwargs:
           self.ncpu = kwargs['ncpu']
        else:
           self.ncpu = 2
        if 'ncpu_a' in kwargs:
           self.ncpu_a = kwargs['ncpu_a']
        else:
           if params.freq == 'wideband':
                self.ncpu_a = 2
        
        if 'nrun' in kwargs:
           self.nrun = kwargs['nrun']
        else:
           self.nrun = 8
        if 'nrun_a' in kwargs:
           self.nrun_a = kwargs['nrun_a']
        else:
           if params.freq == 'wideband':
                self.nrun_a = 8
                
        if params.freq == 'wideband':
            if 'wb_globiter' in kwargs:
                self.wb_globiter = kwargs['wb_globiter']
            else:
                self.wb_globiter = 10
         
        
class simparameters(object):
    
    def __init__(self, params, kwargs):
         
        if 'simpix' in kwargs:
            self.simpix = kwargs['simpix']
        else: 
            self.simpix = 100
            
        if 'nsources' in kwargs:
            self.nsources = kwargs['nsources']
        else: 
            self.nsources = 50
            
        if 'pfactor' in kwargs:
            self.pfactor = kwargs['pfactor']
        else: 
            self.pfactor = 5.
            
        if 'signal_seed' in kwargs:
           self.signal_seed = kwargs['signal_seed']
        else:
           self.signal_seed = 454810740
           
        if 'p0_sim' in kwargs:
           self.p0_sim = kwargs['p0_sim']
        else:
           self.p0_sim = 9.7e-18   
           
        if 'k0' in kwargs:
           self.k0 = kwargs['k0']
        else:
           self.k0 = 19099
           
        if 'sigalpha' in kwargs:
           self.sigalpha = kwargs['sigalpha']
        else:
           self.sigalpha = 2.
               
        if 'noise_seed' in kwargs:
           self.noise_seed = kwargs['noise_seed']
        else:
           self.noise_seed = 3127312
           
        if 'sigma' in kwargs:
           self.sigma = kwargs['sigma']
        else:
           self.sigma = 1e-2
           
        if 'offset' in kwargs:
           self.offset = kwargs['offset']
        else:
           self.offset = 0.
           
        if 'compact' in kwargs:
           self.compact = kwargs['compact']
        else:
           self.compact = False

#-----------------------auxiliary functions------------------------------------           
           
def make_dirtyimage(params, logger):

    casa.clean(vis = params.ms,imagename = 'di',cell = \
        str(params.cellsize) + 'rad', imsize = params.imsize, \
        niter = 0, mode='mfs')

    ia.open('di.image')
    imageinfo = ia.summary('di.image')
    
    norm = imageinfo['incr'][1]/np.pi*180*3600  #cellsize converted to arcsec
    
    beamdict = ia.restoringbeam()
    major = beamdict['major']['value'] / norm
    minor = beamdict['minor']['value'] / norm
    np.save('beamarea',1.13 * major * minor)
    
    di = ia.getchunk().reshape(imageinfo['shape'][0],imageinfo['shape'][1])\
        / (1.13 * major * minor)
    
    ia.close()
    
    call(["rm", "-rf", "di.image"])
    call(["rm", "-rf", "di.model"])
    call(["rm", "-rf", "di.psf"])
    call(["rm", "-rf", "di.flux"])
    call(["rm", "-rf", "di.residual"])

    
    return di
                

def callbackfunc(x, i):
    
    if i%gcallback == 0:
        print 'Callback at iteration' + str(i)
        
        if gsave:
           pl.figure()
           pl.imshow(np.exp(x))
           pl.colorbar()
           pl.title('Iteration' + str(i))
           pl.savefig('resolve_output_' + str(gsave)+ \
               "/last_iterations/" + 'iteration'+str(i))
           np.save('resolve_output_' + str(gsave)+ \
               "/last_iterations/" + 'iteration' + str(i),x)
               
def simulate(params, simparams, logger):
    """
    Setup for the simulated signal.
    """

    logger.header2("Simulating signal and data using provided UV-coverage.")

    u,v = read_data_from_ms(params.ms)[2:4]

    # wide-band imaging
    if params.freq == 'wideband':
        logger.message('Wideband imaging not yet implemented')
    # single-band imaging
    else:
        nspw,chan = params.freq[0], params.freq[1]
        u = u[nspw][chan]
        v = v[nspw][chan]
    
    
    d_space = point_space(len(u), datatype = np.complex128)
    s_space = rg_space(simparams.simpix, naxes=2, dist = params.cellsize, \
        zerocenter=True)
    k_space = s_space.get_codomain()
    kindex,rho_k,pindex,pundex = k_space.get_power_indices()
    
    #setting up signal power spectrum
    powspec_I = [simparams.p0_sim * (1. + (k / simparams.k0) ** 2) ** \
        (-simparams.sigalpha) for k in kindex]
    if params.save:      
        save_results(kindex,'simulated signal PS','resolve_output_' + \
            str(params.save) + "/general/" + params.save + '_ps_original',\
            log = 'loglog', value2 = powspec_I)

    S = power_operator(k_space, spec=powspec_I)

    # extended signal
    np.random.seed(simparams.signal_seed)
    I = field(s_space, random="syn", spec=S.get_power()) + simparams.offset
    np.random.seed()    
    
    # compact signal
    Ip = np.zeros((simparams.simpix,simparams.simpix))
    if simparams.compact:
        for i in range(simparams.nsources):
               Ip[np.random.randint(0,high=simparams.simpix),\
               np.random.randint(0,high=simparams.simpix)] = \
               np.random.random() * simparams.pfactor * np.max(exp(I))  
          
   
    if params.save:      
        save_results(exp(I),'simulated extended signal','resolve_output_' + \
            str(params.save) + "/general/" + params.save + '_expsimI')
        if simparams.compact:
            save_results(Ip,'simulated compact signal','resolve_output_' + \
                str(params.save) + "/general/" + params.save + '_expsimIp')
    
    if params.save:
        # maximum k-mode and resolution of data
        uvrange = np.array([np.sqrt(u[i]**2 + v[i]**2) for i in range(len(u))])
        dx_real_rad = (np.max(uvrange))**-1
        logger.message('resolution\n' + 'rad ' + str(dx_real_rad) + '\n' + \
            'asec ' + str(dx_real_rad/asec2rad))

        save_results(u,'UV','resolve_output_' + str(params.save) + \
            "/general/" +  params.save + "_uvcov", plotpar='o', value2 = v)

    # noise
    N = diagonal_operator(domain=d_space, diag=simparams.sigma**2)
    
    # response, no simulated primary beam
    A = 1.
    R = r.response(s_space, sym=False, imp=True, target=d_space, \
                       para=[u,v,A,False])
                 
    #Set up Noise
    np.random.seed(simparams.noise_seed)
    n = field(d_space, random="gau", var=N.diag(bare=True))
    # revert to unseeded randomness
    np.random.seed()  
    
    #plot Signal to noise
    sig = R(field(s_space, val = np.exp(I) + Ip))
    if params.save:
        save_results(abs(sig.val) / abs(n.val),'Signal to noise', \
           'resolve_output_' + str(params.save) + \
           "/general/" + params.save + '_StoN',log ='semilog')
        save_results(np.exp(I) + Ip,'Signal', \
           'resolve_output_' + str(params.save) + \
           "/general/" + params.save + '_signal')

    d = R(exp(I) + Ip) + n
    
    # reset imsize settings for requested parameters
    s_space = rg_space(params.imsize, naxes=2, dist = params.cellsize, \
        zerocenter=True)
    R = r.response(s_space, sym=False, imp=True, target=d_space, \
                       para=[u,v,A,False])
    
    # dirty image for comparison
    di = R.adjoint_times(d) * s_space.vol[0] * s_space.vol[1]

    # more diagnostics if requested
    if params.save:
        
        # plot the dirty image
        save_results(di,"dirty image",'resolve_output_' + str(params.save) +\
            "/general/" + params.save + "_di")
    
    return d, N, R, di, d_space, s_space


def parse_input_file(infile):
    """ parse the parameter file."""

    reader = csv.reader(open(infile, 'rb'), delimiter=" ",
                        skipinitialspace=True)
    parset = dict()

    for row in reader:
        if len(row) != 0 and row[0] != '%':
            parset[row[0]] = row[1]


    params = parameters(parset['ms'], parset['imsize'], parset['cellsize'],\
                        parset['algorithm'], parset['init_type_s'], \
                        parset['use_init_s'], parset['init_type_p'], \
                        parset['init_type_p_a'], parset['lastit'], \
                        parset['freq'], parset['pbeam'],parset['uncertainty'],\
                        parset['noise_est'], parset['map_algo'],\
                        parset['pspec_algo'], parset['barea'], \
                        parset['map_conv'], parset['pspec_conv'],\
                        pasret['save'], parset['callback'], parset['plot'],\
                        parset['simulating'], parset['restfreq'])       
    
    numparams = numparameters(params, parset)

    return params, numparams


def write_output_to_fits(m, params, notifier='',mode='I'):
    """
    """

    hdu_main = pyfits.PrimaryHDU(m)
    
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
    
    hdu.header.update('CDELT1', params.cellsize, 'Size of pixel bin')
    
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
    
def save_results(value,title,fname,log = None,value2 = None, \
    value3= None, plotpar = None, rho0 = 1., twoplot=False):
        
    rho0 = 1
    
    # produce plot and save it as png file   
    pl.figure()
    pl.title(title)
    if plotpar:
        if log == 'loglog' :
            pl.loglog(value,value2)
            if twoplot:
                pl.loglog(value,value3,plotpar)
        elif log == 'semilog':
            pl.semilogy(value)
            if twoplot:
                pl.semilogy(value3,plotpar)
        else :
            if len(np.shape(value)) > 1:
                pl.imshow(value/rho0)
                pl.colorbar()
            else:
                pl.plot(value,value2,plotpar)
                
            pl.savefig(fname + ".png")
    else:
        if log == 'loglog' :
            pl.loglog(value,value2)
            if twoplot:
                pl.loglog(value,value3)
        elif log == 'semilog':
            pl.semilogy(value)
            if twoplot:
                pl.semilogy(value3)
        else :
            if len(np.shape(value)) > 1:
                pl.imshow(value/rho0)
                pl.colorbar()
            else:
                pl.plot(value,value2)
        pl.savefig(fname + ".png")          
        
    pl.close
    
    # save data as npy-file
    if rho0 != 1.:
       np.save(fname,value/rho0)
    else:
       np.save(fname,value)


#*******************************************************************************
# Define truncatd exp and log functions for nifty fields to avoid NANs*********

def exp(x):

    if(isinstance(x,field)):
#        if(np.any(x.val>709)):
 #            print("** EXPSTROKE **")
        return field(x.domain,val=np.exp(np.minimum(709,x.val)),target=x.target)
        #return field(x.domain,val=np.exp(x.val),target=x.target)
    else:
#        if(np.any(x>709)):
#            print("** EXPSTROKE **")
        return np.exp(np.minimum(709,np.array(x)))
        #return np.exp(np.array(x))

def log(x,base=None):

    if(base is None):
        if(isinstance(x,field)):
#            if(np.any(x.val<1E-323)):
#                print("** LOGSTROKE **")
            return \
                field(x.domain,val=np.log(np.maximum(1E-323,x.val)),target=x.target)
            #return field(x.domain,val=np.log(x.val),target=x.target)
        else:
#            if(np.any(x<1E-323)):
#                print("** LOGSTROKE **")
            return np.log(np.array(np.maximum(1E-323,x)))
            #return np.log(np.array(x))

    base = np.array(base)
    if(np.all(base>0)):
        if(isinstance(x,field)):
#            if(np.any(x.val<1E-323)):
#                print("** LOGSTROKE **")
            return field(x.domain,val=np.log(np.maximum(1E-323,x.val))/np.log(base).astype(x.domain.datatype),target=x.target)
            #return field(x.domain,val=np.log(x.val)/np.log(base).astype(x.domain.datatype),target=x.target)
        else:
#            if(np.any(x<1E-323)):
#                print("** LOGSTROKE **")
            return np.log(np.array(np.maximum(1E-323,x)))/np.log(base)
    else:
        raise ValueError(about._errors.cstring("ERROR: invalid input basis."))
        

    
    
    
