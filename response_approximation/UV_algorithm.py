from nifty import *
from UV_classes import UV_quantities, lognormal_Hamiltonian
import time as ttt


def fastresolve(R, d, s_space, path, noise_update=False, noise_est='fr_internal', msg='fr_internal', psg='fr_internal', point=0):

    print 'Starting fastResolve cycle.'

    k_space = s_space.get_codomain()
    k_space.set_power_indices(log=True)

    d_original = d+0.
	
    #Check whether to take the noise estimate from main routine or to do it internally
    if noise_est == 'fr_internal':
        SNR_assumed = 1	     
        varis = np.mean(np.abs(d*d))/(1.+SNR_assumed)
    else:
        varis = noise_est

    print "fastResolve internal data set up."

    # #############################
    # # execute gridding routines #
    # #############################
	
    u = R.u
    v = R.v 
    A = R.A

    aaa = ttt.time()  # overall time stoppage for preparation
     
    global UV_quants
    UV_quants = UV_quantities(domain=s_space, codomain=k_space,
							  u=u, v=v, d=d, varis=varis, A=A)
    time_prep = ttt.time()-aaa

    print "the preparation took", time_prep, "seconds."

    bbbb = ttt.time()

    # ###########################
    # # create starting guesses #
    # ###########################

    if psg=='fr_internal':
        # create internal starting guess power spectrum
        klen_log = k_space.get_power_indices()[0]
        spec = (klen_log+k_space.vol[0])**(-3.)
        fac = 0.15/(4*spec[1]*k_space.vol.prod()**2)
        spec *= fac
        spec[0] *= 1.e7
        global S
        S = power_operator(k_space, spec=spec)

    else:
        # take starting guess power spectrum from main routine
        spec = psg
	S = power_operator(k_space, spec=spec)
	
    if msg=='fr_internal':
        m_matched = UV_quants.get_matched()
        global RR
        global NN
        global dd
        RR = UV_quants.get_RR()
        NN = UV_quants.get_NN()
        dd = UV_quants.get_dd()

        mean = (m_matched-m_matched.min()).mean()
        global m
        m = field(s_space, target=k_space, val=log(mean))
		
    else:
        # take starting guess map from main routine
        global m
        m = msg
        global RR
        global NN
        global dd
        RR = UV_quants.get_RR()
        NN = UV_quants.get_NN()
        dd = UV_quants.get_dd()

    global Ipoint		
    Ipoint = field(s_space, target=k_space, val=0.)
	
    # #######################
    # # Further preparation #
    # #######################

    global Sk
    Sk = S.get_projection_operator()

    min_wave_vari = 1.e-8
    global alpha
    global q
    alpha = 1+Sk.rho()*0.05 + 1
    q = (alpha-1)/k_space.vol.prod() * min_wave_vari

    global t
    global Max_point
    t = 1.75
    Max_point = point
	
    # ###################
    # # Start Algorithm #
    # ###################

    pcycle = True

    Ilist1 = []
    spec_list1 = [S.get_power()]
    print "fastResolve loop 1"
    for ii in xrange(20):

        if(ii == 0):
            power_cycle(limii=2000)

        if(pcycle):
            for jj in xrange(10):
                power_cycle(limii=200)
            pcycle = False
            spec_list1 += [S.get_power()]
        else:
            while(not pcycle):
                pcycle = ext_point_cycle(limii=400)

        Ilist1 += [exp(m) + Ipoint]
		
        np.save(path+"cycle1_I_%02i.npy"%ii, (exp(m) + Ipoint).val)
        np.save(path+"cycle1_m_%02i.npy"%ii, m.val)
        np.save(path+"cycle1_Ipoint_%02i.npy"%ii, Ipoint.val)
        np.save(path+"cycle1_spec_%02i.npy"%ii, S.get_power())

    Ilist2 = []
    if(noise_update):
        print "fastResolve loop 2, activated noise update mode."
        for ii in xrange(40):
            H = lognormal_Hamiltonian(S=S, N=NN, R=RR, d=dd)
            m = H.min_BFGS(m, limii=400, note=False, tol=1.e-3)

            vari_estimation()

            Ilist2 += [exp(m) + Ipoint]
            np.save(path+"cycle2_I_%02i.npy"%ii, (exp(m) + Ipoint).val)
            np.save(path+"cycle2_m_%02i.npy"%ii, m.val)
            np.save(path+"cycle2_est_var_%02i.npy"%ii, est_var)

    return m, S.get_power(), k_space
	
				
def ext_point_cycle(limii=50):

	global m, Ipoint, UV_quants, dd

	aaa = ttt.time()

	H = lognormal_Hamiltonian(S=S, N=NN, R=RR, d=dd)
	m = H.min_BFGS(m, limii=limii, note=False, tol=1.e-3)

	print "minimization took", ttt.time()-aaa, "seconds."

	I = exp(m)

	Itemp = I+0.
	Itemp[np.where(Itemp.val < Itemp.max()/t)] = 0.
	Itemp *= (1-1./t)

	if(((Ipoint+Itemp).val > 0).sum() > Max_point):
		return True
	else:
		m = log(I-Itemp)

		Ipoint += Itemp

		d_point = R(Ipoint)

		d = d_original - d_point

		aaa = ttt.time()
		UV_quants.set_data(d=d)
		print "data update took", ttt.time()-aaa, "seconds."

		dd = UV_quants.get_dd()
		return False


def power_cycle(limii=20):
	global m, S

	aaa = ttt.time()

	H = lognormal_Hamiltonian(S=S, N=NN, R=RR, d=dd)
	m = H.min_BFGS(m, limii=limii, note=False, tol=1.e-3)

	D = H.crude_Hessian(m)

	spec = infer_power(m, Sk=Sk, D=D, alpha=alpha, q=q,
					   smoothness=True, sigma=1)

	print "power spectrum estimation took", ttt.time()-aaa, "seconds."
	print "maximal change in power by a log difference of",\
		np.max(np.abs(log(spec[1:]) - log(S.get_power()[1:])))

	S.set_power(spec)

	return None


def vari_estimation():
	global UV_quants, RR, NN, dd, est_var
	REG_VAR = 0.9

	aaa = ttt.time()
	est_var = (R(exp(m)) - d).val
	est_var = np.abs(est_var)**2
	est_var = REG_VAR*est_var + (1-REG_VAR)*est_var.mean()

	est_var = np.sqrt(UV_quants.varis * est_var)

	UV_quants = UV_quantities(domain=s_space, codomain=k_space,
							  u=u, v=v, d=d, varis=est_var, A=A)
	RR = UV_quants.get_RR()
	NN = UV_quants.get_NN()
	dd = UV_quants.get_dd()
	print "variance estimation took", ttt.time()-aaa, "seconds"
	print "and yielded a mean variance of", est_var.mean()
	print "which corresponds to a SNR of", \
		np.mean(np.abs(d_original)**2)/est_var.mean()-1

	return None
