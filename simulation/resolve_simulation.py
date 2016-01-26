#VERY unelegnat quick fix
import sys
sys.path.append('../')

from nifty import *
import numpy as np
import utility_functions as utils 


def simulate(params, simparams, logger):
    """
    Setup for the simulated signal.
    """

    logger.header2("Simulating signal and data using provided UV-coverage.")
    
    # Assumes uv-coverage to be accesible via numpy arrays
    u = np.load(msfn + '_u.npy')
    v = np.load(msfn + '_v.npy')

    # wide-band imaging
    if params.freq == 'wideband':
        logger.failure('Wideband simulation not yet implemented')
        raise NotImplementedError
        
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
#        for i in range(simparams.nsources):
#               Ip[np.random.randint(0,high=simparams.simpix),\
#               np.random.randint(0,high=simparams.simpix)] = \
#               np.random.random() * simparams.pfactor * np.max(exp(I))  
        np.random.seed(81232562353)
        Ip= sc.invgamma.rvs(0.5, size = simparams.simpix*simparams.simpix,scale = 1e-7) 
        Ip.shape = (simparams.simpix,simparams.simpix)
        np.random.seed()          
   

    save_results(exp(I),'simulated extended signal','resolve_output_' + \
        str(params.save) + "/general/" + params.save + '_expsimI')
    if simparams.compact:
        save_results(Ip,'simulated compact signal','resolve_output_' + \
            str(params.save) + "/general/" + params.save + '_expsimIp')
    

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
    R = r.response(s_space, d_space, u, v, A)
                 
    #Set up Noise
    np.random.seed(simparams.noise_seed)
    n = field(d_space, random="gau", var=N.diag(bare=True))
    # revert to unseeded randomness
    np.random.seed()  
    
    #plot Signal to noise
    sig = R(field(s_space, val = exp(I) + Ip))

    save_results(abs(sig.val) / abs(n.val),'Signal to noise', \
       'resolve_output_' + str(params.save) + \
       "/general/" + params.save + '_StoN',log ='semilog')
    save_results(exp(I) + Ip,'Signal', \
       'resolve_output_' + str(params.save) + \
       "/general/" + params.save + '_signal')

    d = R(exp(I) + Ip) + n
        
    # reset imsize settings for requested parameters
    s_space = rg_space(params.imsize, naxes=2, dist = params.cellsize, \
        zerocenter=True)
    R = r.response(s_space, d_space, u, v, A)
    
    # dirty image for comparison
    di = R.adjoint_times(d) * s_space.vol[0] * s_space.vol[1]

    # more diagnostics if requested
    if params.save:
        
        # plot the dirty image
        save_results(di,"dirty image",'resolve_output_' + str(params.save) +\
            "/general/" + params.save + "_di")
    
    return d, N, R, di, d_space, s_space, exp(I), n
    
class simparameters(object):
    """
    Defines a simulation parameter class, only needed when RESOLVE is in 
    simulating-mode.
    """      
    
    def __init__(self, params):
          
        parset = params.parset  
        self.check_default('simpix', parset, 100)
        self.check_default('signal_seed', parset, 454810740)
        self.check_default('noise_seed', parset, 3127312)
        self.check_default('p0_sim, parset', parset, 9.7e-18)
        self.check_default('k0', parset, 19099) 
        self.check_default('sigalpha', parset, 2)  
        self.check_default('sigma', parset, 1e-12)
        self.check_default('offset', parset, 0)   
        self.check_default('compact', parset, False)
        
        if self.compact:
            
            self.check_default('nsources', parset, 50)
            self.check_default('pfactor', parset, 5) 

    def check_default(self, parameter, parset, default):
        
        if parameter in parset:
            setattr(self, parameter, parset[str(parameter)])
        else:
            setattr(self, parameter, default)
