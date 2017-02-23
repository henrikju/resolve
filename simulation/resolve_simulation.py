"""
resolve.py
Written by Henrik Junklewitz

Simulation.py is a routine that provides simulated data set to
resolve given a defined set of uv coordiantes.

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


#VERY unelegant quick fix
import sys
sys.path.append('../')

from nifty import *
import numpy as np
import utility_functions as utils 
import response as r
from operators import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

asec2rad = 4.84813681e-6

def simulate(params, logger):
    """
    Setup for the simulated signal.
    """

    logger.header2("Simulating signal and data using provided UV-coverage.")

    # Assumes uv-coverage to be accesible via numpy arrays
    u = np.load(params.datafn + '_u.npy')
    v = np.load(params.datafn + '_v.npy')

    # wide-band imaging
    if params.multifrequency:
        logger.failure('Wideband simulation not yet implemented')
        raise NotImplementedError
        
    # single-band imaging
    else:
        nspw,chan = params.freq[0], params.freq[1]
        u = u[nspw][chan]
        v = v[nspw][chan]
    
    
    d_space = point_space(len(u), datatype = np.complex128)
    s_space = rg_space(params.simpix, naxes=2, dist = params.cellsize, \
        zerocenter=True)
    k_space = s_space.get_codomain()
    kindex,rho_k,pindex,pundex = k_space.get_power_indices()
    
    #setting up signal power spectrum
    if not params.sim_ext_from_file:
        powspec_s = [params.p0_sim * (1. + (k / params.k0) ** 2) ** \
            (-params.sigalpha) for k in kindex]
        utils.save_results(kindex,'simulated signal PS','resolve_output_' + \
           str(params.save) + "/general/" + params.save + '_ps_original',\
           log = 'loglog', value2 = powspec_I)

        S = power_operator(k_space, spec=powspec_s)

        # extended signal
        np.random.seed(params.signal_seed)
        s = field(s_space, random="syn", spec=S.get_power()) + params.offset
        np.random.seed()
    else:
        val = np.load(params.sim_ext_from_file)
        val[val<1e-6] = 1e-6
        s = field(s_space, val=log(val))
        powspec_s = s.power()
        S = power_operator(k_space, spec=powspec_s)   
    
    # get powerspectrum for comparison 
    k_space.set_power_indices(log=True, nbins=params.bins)    
    kindex_sim, rho_sim, pindex_sim,pundex_sim = k_space.get_power_indices()
    powspec_sim = s.power(pindex=pindex_sim,kindex=kindex_sim,rho=rho_sim)    
    
    # compact signal
    Ip = np.zeros((params.simpix,params.simpix))
    if params.compact:
        if not params.sim_compact_from_file:
            np.random.seed(params.compact_seed)
            Ip= sc.invgamma.rvs(0.5, size=params.simpix * params.simpix,
                                scale=params.sim_eta)
            Ip.shape = (params.simpix, params.simpix)
            np.random.seed()
        else:
            Ip = np.load(params.sim_compact_from_file) 

    utils.save_results(exp(I),'simulated extended signal','resolve_output_' + \
        str(params.save) + "/general/" + params.save + '_expsimI')
    if params.compact:
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
    sig = R(field(s_space, val = exp(s) + Ip))
    SNR = params.SNR
    np.random.seed(params.noise_seed)
    var = abs(sig.dot(sig)) / (params.simpix*params.simpix*SNR)
    logger.message('Noise variance used in simulation: ' + str(var))
    N = diagonal_operator(domain=d_space,diag=var*np.ones(d_space.num()))
                 
    #Set up Noise
    np.random.seed(params.noise_seed)
    n = field(d_space, random="gau", var=var)
    # revert to unseeded randomness
    np.random.seed()  
    
    # Plot and save signal
    utils.save_results(exp(s) + Ip,'Signal', \
       'resolve_output_' + str(params.save) + \
       "/general/" + params.save + '_signal')

    d = R(exp(s) + Ip) + n
        
    # reset imsize settings for requested parameters; corrupt noise if wanted
    s_space = rg_space(params.imsize, naxes=2, dist = params.cellsize, \
        zerocenter=True)

    R = r.response(s_space, d_space, u, v, A)
    if params.noise_corruption:
        N = diagonal_operator(domain=d_space, diag=var*params.noise_corruption
                              * np.ones(d_space.num()))
        logger.message('Corrupt noise variance by factor: '
                       + str(params.noise_corruption))

    # dirty image for comparison
    di = R.adjoint_times(d) * s_space.vol[0] * s_space.vol[1]
        
    utils.save_results(di,"dirty image",'resolve_output_' + str(params.save) +\
        "/general/" + params.save + "_di")
        
    # plot the dirty beam
    uvcov = field(d_space,val=np.ones(np.shape(d.val), \
        dtype = np.complex128))            
    db = R.adjoint_times(uvcov) * s_space.vol[0] * s_space.vol[1]       
    utils.save_results(db,"dirty beam",'resolve_output_' + str(params.save)+\
            '/general/' + params.save + "_db")
        
    # perform uv_cut on simulated data if required
    # use >10 for everting with uvrange over 10 % of the max uvrange
    if not params.sim_uv_cut_str == 'None':
        proz = float('0.' +
                     (params.sim_uv_cut_str[1:len(params.sim_uv_cut_str)]))
        uv_cut = proz*np.max(np.sqrt(u**2+v**2))
        if params.sim_uv_cut_str[0] == '>':
            u_top = u[np.sqrt(u**2 + v**2) > uv_cut]
            v_top = v[np.sqrt(u**2 + v**2) > uv_cut]
            d_u_space = point_space(len(u_top), datatype=np.complex128)
            Rdirty_u = r.response(s_space, d_u_space, u_top, v_top, A)
            N_u = diagonal_operator(domain=d_u_space, diag=var)
            n_u = field(d_u_space, random="gau", var=N_u.diag(bare=True))
            d_u = Rdirty_u(exp(s) + Ip) + n_u
            di_u = Rdirty_u.adjoint_times(d_u) \
                * s_space.vol[0] * s_space.vol[1]
            utils.save_results(di_u, "dirty image_u", 'resolve_output_'
                               + str(params.save) + "/general/" + params.save +
                               "_di_u")
            return d_u, N_u, Rdirty_u, di_u, d_u_space, s_space, exp(s),\
                n_u, powspec_sim
        elif params.sim_uv_cut_str[0] == '<':
            u_bot = u[np.sqrt(u**2 + v**2) < uv_cut]
            v_bot = v[np.sqrt(u**2 + v**2) < uv_cut]
            d_m_space = point_space(len(u_bot), datatype=np.complex128)
            Rdirty_m = r.response(s_space, d_m_space, u_bot, v_bot, A)
            N_m = diagonal_operator(domain=d_m_space, diag=var)
            n_m = field(d_m_space, random="gau", var=N_m.diag(bare=True))
            d_m = Rdirty_m(exp(s) + Ip) + n_m
            di_m = Rdirty_m.adjoint_times(d_m) \
                * s_space.vol[0] * s_space.vol[1]
            utils.save_results(di_m, "dirty image_m", 'resolve_output_'
                               + str(params.save) + "/general/" + params.save +
                               "_di_m")
            return d_m, N_m, Rdirty_m, di_m, d_m_space, s_space, \
                exp(s), n_m, powspec_sim

    return d, N, R, di, d_space, s_space, exp(s), n, powspec_sim