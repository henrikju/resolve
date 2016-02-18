#VERY unelegnat quick fix
import sys
sys.path.append('../')

from nifty import *
import numpy as np
import utility_functions as utils 
import response as r

asec2rad = 4.84813681e-6

def simulate(params, simparams, logger):
    """
    Setup for the simulated signal.
    """

    logger.header2("Simulating signal and data using provided UV-coverage.")

    # Assumes uv-coverage to be accesible via numpy arrays
    u = np.load(params.ms + '_u.npy')
    v = np.load(params.ms + '_v.npy')

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
    utils.save_results(kindex,'simulated signal PS','resolve_output_' + \
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
   

    utils.save_results(exp(I),'simulated extended signal','resolve_output_' + \
        str(params.save) + "/general/" + params.save + '_expsimI')
    if simparams.compact:
        utils.save_results(Ip,'simulated compact signal','resolve_output_' + \
            str(params.save) + "/general/" + params.save + '_expsimIp')
    

    # maximum k-mode and resolution of data
    uvrange = np.array([np.sqrt(u[i]**2 + v[i]**2) for i in range(len(u))])
    dx_real_rad = (np.max(uvrange))**-1
    logger.message('resolution\n' + 'rad ' + str(dx_real_rad) + '\n' + \
        'asec ' + str(dx_real_rad/asec2rad))

    utils.save_results(u,'UV','resolve_output_' + str(params.save) + \
        "/general/" +  params.save + "_uvcov", plotpar='o', value2 = v)

    # response, no simulated primary beam
    A = 1.
    R = r.response(s_space, d_space, u, v, A)

    # Set up Noise
    sig = R(field(s_space, val = exp(I) + Ip))
    SNR = simparams.SNR
    np.random.seed(simparams.noise_seed)
    var = abs(sig.dot(sig)) / SNR
    logger.message('Noise variance used in simulation: ' + str(var))
    N = diagonal_operator(domain=d_space, diag=var)
                 
    #Set up Noise
    np.random.seed(simparams.noise_seed)
    n = field(d_space, random="gau", var=N.diag(bare=True))
    # revert to unseeded randomness
    np.random.seed()  

    # Plot and save signal
    utils.save_results(exp(I) + Ip,'Signal', \
       'resolve_output_' + str(params.save) + \
       "/general/" + params.save + '_signal')

    d = R(exp(I) + Ip) + n
        
    # reset imsize settings for requested parameters; corrupt noise if wanted
    s_space = rg_space(params.imsize, naxes=2, dist = params.cellsize, \
        zerocenter=True)
    R = r.response(s_space, d_space, u, v, A)
    if simparams.noise_corruption:
        N = diagonal_operator(domain=d_space, diag=var*simparams.noise_corruption)
        logger.message('Corrupt noise variance by factor: ' + str(simparams.noise_corruption))
    
    # dirty image for comparison
    di = R.adjoint_times(d) * s_space.vol[0] * s_space.vol[1]
        
    utils.save_results(di,"dirty image",'resolve_output_' + str(params.save) +\
        "/general/" + params.save + "_di")
    
    return d, N, R, di, d_space, s_space, exp(I), n
    
class simparameters(object):
    """
    Defines a simulation parameter class, only needed when RESOLVE is in 
    simulating-mode.
    """      
    
    def __init__(self, params):
          
        parset = params.parset  
        self.check_default('simpix', parset, 100, dtype = int)
        self.check_default('signal_seed', parset, 454810740, dtype = int)
        self.check_default('noise_seed', parset, 3127312, dtype = int)
        self.check_default('p0_sim', parset, 9.7e-18, dtype = float)
        self.check_default('k0', parset, 19099, dtype = float) 
        self.check_default('sigalpha', parset, 2, dtype = float)  
        self.check_default('SNR', parset, 1., dtype = float)
        self.check_default('offset', parset, 0, dtype = float)   
        self.check_default('compact', parset, False, dtype = bool)
        self.check_default('noise_corruption', parset, 0, dtype = float)
        if self.compact:
            self.check_default('nsources', parset, 50, dtype = int)
            self.check_default('pfactor', parset, 5, dtype = float)
    
    def check_default(self, parameter, parset, default, dtype=str):

        if parameter in parset:
            setattr(self, parameter, dtype(parset[str(parameter)]))
        else:
            setattr(self, parameter, default)
