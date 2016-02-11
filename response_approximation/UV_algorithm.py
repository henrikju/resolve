from nifty import *
from UV_classes import UV_quantities, lognormal_Hamiltonian
import time as ttt

# ### u and v need to be in 1/rad
# NEEDED:
# u: np.array[float, ndim=1], the u values in 1/rad
# v: np.array[float, ndim=1], the v values in 1/rad
# d: np.array[complex, ndim=1], the visibilities in ???
# SNR_assumed: float, the assumed signal to noise ratio (e.g. 10)
# Max_point: int, the maximal amount of point sources allowed
# Npix: int (multiple of 2), amount of pixels in each dimension
# dist: float (positive), pixel edge length
# SAVE: bool, whether to save results to 'path' or not
# path: string, path where results are saved
    
def fastresolve(R, d, SNR_assumed, s_space, path, SAVE=True, do_point=False):

    print 'Starting fastResolve cycle'
    
    u = R.u
    v = R.v 
    A = R.A
    
    k_space = s_space.get_codomain()
    k_space.set_power_indices(log=True)
    
    d_original = d+0.
    varis = np.mean(np.abs(d*d))/(1.+SNR_assumed)
    
    print "Data set up for fastResolve"
    
    # #############################
    # # execute gridding routines #
    # #############################
    
    aaa = ttt.time()  # overall time stoppage for preparation
    
    UV_quants = UV_quantities(domain=s_space, codomain=k_space,
                              u=u, v=v, d=d, varis=varis, A=A)
    
    time_prep = ttt.time()-aaa
    
    print "the preparation took", time_prep, "seconds."
    
    bbbb = ttt.time()
    
    
    ###################################
    
    # create starting guess power spectrum
    klen_log = k_space.get_power_indices()[0]
    spec = (klen_log+k_space.vol[0])**(-3.)
    fac = 0.15/(4*spec[1]*k_space.vol.prod()**2)
    spec *= fac
    spec[0] *= 1.e7
    S = power_operator(k_space, spec=spec)
    
    m_matched = UV_quants.get_matched()
    RR = UV_quants.get_RR()
    NN = UV_quants.get_NN()
    dd = UV_quants.get_dd()
    
    mean = (m_matched-m_matched.min()).mean()
    
    m = field(s_space, target=k_space, val=log(mean))
    Ipoint = field(s_space, target=k_space, val=0.)
    
    Ilist = []
    
    Sk = S.get_projection_operator()
    
    min_wave_vari = 1.e-8
    alpha = 1+Sk.rho()*0.01
    q = (alpha-1)/k_space.vol.prod() * min_wave_vari
    
    spec_list = [S.get_power()]
    
    if do_point:
        for ii in xrange(100):
            aaa = ttt.time()
        
            H = lognormal_Hamiltonian(S=S, N=NN, R=RR, d=dd)
            if(ii == 0):
                m = H.min_BFGS(m, limii=200, note=True, tol=1.e-3)
            m = H.min_BFGS(m, limii=50, note=False, tol=1.e-3)
        
            print "minimization", ii, "took", ttt.time()-aaa, "seconds."
        
            I = exp(m)
        
            Ilist += [I+Ipoint]
        
            if(SAVE):
                np.save(path+"point_cycle_m%03d"%ii, m.val)
                np.save(path+"point_cycle_I%03d"%ii, Ilist[-1].val)
        
            Itemp = I+0.
            Itemp[np.where(Itemp.val < Itemp.max()/1.75)] = 0.
            Itemp *= (1-1./1.75)
        
            if(((Ipoint+Itemp).val > 0).sum() > Max_point):
                break
        
            m = log(I-Itemp)
        
            Ipoint += Itemp
        
            d_point = R(Ipoint)
        
            d = d_original - d_point
        
            aaa = ttt.time()
            UV_quants.set_data(d=d)
            print "data update", ii, "took", ttt.time()-aaa, "seconds."
        
            dd = UV_quants.get_dd()
    
        print "point source detection finished."
        if(SAVE):
            np.save(path+"point_cycle_Ipoint_final", Ipoint.val)
    
    for ii in xrange(40):
        aaa = ttt.time()
    
        H = lognormal_Hamiltonian(S=S, N=NN, R=RR, d=dd)
        m = H.min_BFGS(m, limii=50, note=False, tol=1.e-3)
    
        Ilist += [exp(m) + Ipoint]
    
        D = H.crude_Hessian(m)
    
        spec = infer_power(m, Sk=Sk, D=D, alpha=alpha, q=q,
                           smoothness=True, sigma=1)
    
        if(SAVE):
            np.save(path+"power_iter_m%03d"%ii, m.val)
            np.save(path+"power_iter_I%03d"%ii, Ilist[-1].val)
            np.save(path+"power_iter_spec%03d"%ii, spec)
    
        print "power spectrum estimation", ii, "took", ttt.time()-aaa, "seconds."
        print "maximal change in power by a log difference of",\
            np.max(np.abs(log(spec[1:]) - log(S.get_power()[1:])))
    
        S.set_power(spec)
        spec_list += [S.get_power()]
    
    
    H = lognormal_Hamiltonian(S=S, N=NN, R=RR, d=dd)
    m = H.min_BFGS(m, limii=200, note=True, tol=1.e-3)
    
    I = exp(m) + Ipoint
    Ilist += [exp(m) + Ipoint]
    
    if(SAVE):
        np.save(path+"final_I", Ilist[-1].val)
    
    return m, spec, k_space
