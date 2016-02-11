from nifty import np, about, rg_space, field,\
    operator, identity, diagonal_operator, power_operator,\
    steepest_descent
from grid_function import grid_function
from scipy.optimize import fmin_l_bfgs_b


def exp(x):
    """
    exponential function with cut-off to prevent overflow
    """
    if(isinstance(x, field)):
        return field(x.domain,
                     val=np.exp(np.minimum(200, x.val)),
                     target=x.target)  # prevent overflow
    else:
        return np.exp(np.minimum(200, x))  # prevent overflow


class fast_Fourier_response(operator):

    def __init__(self, domain, fdiag=None, A=None, target=None):

        if(not isinstance(domain, rg_space)):
            raise ValueError("Error: invalid domain.")
        elif(domain.naxes() != 2):
            raise ValueError("Error: only 2D domains supported.")
        else:
            self.domain = domain

        if(target is None):
            self.target = domain.get_codomain()
        elif(self.domain.check_codomain(target)):
            self.target = target
        else:
            raise ValueError("Error: codomain and domain not compatible.")

        self.imp = True
        self.sym = False
        self.uni = False

        if(fdiag is None):
            fdiag = 1.
        self.Fdiag = diagonal_operator(self.target, diag=fdiag, bare=False)

        if(A is None):
            self.A = identity(self.domain)
        else:
            self.A = diagonal_operator(self.domain, diag=A, bare=False)

    def set_A(self, A):
        self.A = diagonal_operator(self.domain, diag=A, bare=False)

    def _multiply(self, x):

        res = self.A(x)

        res = res.transform(target=self.target)
        res = self.Fdiag(res)

        return res

    def _adjoint_multiply(self, x):

        res = self.Fdiag(x)
        res = res.transform(target=self.domain)
        res = self.A(res)

        return res


class UV_quantities(object):

    def __init__(self, domain=None, codomain=None,
                 u=None, v=None, d=None, varis=None, A=None):

        if(not isinstance(domain, rg_space)):
            raise ValueError("Error: invalid domain.")
        elif(domain.naxes() != 2):
            raise ValueError("Error: only 2D domains supported.")
        else:
            self.domain = domain

        if(codomain is None):
            self.codomain = domain.get_codomain()
        elif(self.domain.check_codomain(codomain)):
            self.codomain = codomain
        else:
            raise ValueError("Error: codomain and domain not compatible.")

        u = np.array(u, dtype=np.float64)
        v = np.array(v, dtype=np.float64)

        if(u.shape != v.shape):
            raise ValueError("Error: u and v arrays not compatible.")

        if(len(u.shape) != 1):
            raise ValueError("Error: u array not one-dimensional.")

        self.u = u
        self.v = v

        if(varis is None):
            about.warnings.cprint(
                "Warning: variance not given, assumed to be 1.")
            varis = np.ones(len(self.u))
        elif(np.isscalar(varis)):
            varis = np.ones(len(self.u))*varis
        else:
            varis = np.array(varis, dtype=np.float64)

        self.varis = varis

        if(d is None):
            about.warnings.cprint(
                "Warning: variance not given, assumed to be 0.")
            d = np.zeros(len(self.u), dtype=np.complex128)
        elif(np.isscalar(d)):
            d = np.ones(len(self.u), dtype=np.complex128) * d
        else:
            d = np.array(d, dtype=np.complex128)

        if(d.shape != self.u.shape):
            raise ValueError("Error: d and u not compatible.")

        self.d = d

        self.A = field(self.domain, target=self.codomain, val=A)

        self._jnorm = None

        self.j_noA_k = field(self.codomain, target=self.domain)

        self.Mdiag = None
        self.RR = None
        self.NN = None
        self.Mregu = None

        self._calc_UV_ops()
        self._calc_UV_source(d=self.d, calc_norm=True)

    def set_data(self, d=None):

        if(d is None):
            about.warnings.cprint(
                "Warning: variance not given, assumed to be 0.")
            d = np.zeros(len(self.u), dtype=np.complex128)
        elif(np.isscalar(d)):
            d = np.ones(len(self.u), dtype=np.complex128) * d
        else:
            d = np.array(d, dtype=np.complex128)

        if(d.shape != self.u.shape):
            raise ValueError("Error: d and u not compatible.")

        self.d = d
        self._calc_UV_source(d=self.d, calc_norm=False)

    def get_j(self, d=None, fourier=False, withA=True):

        if(withA):
            j = self.j_noA_k.transform(target=self.domain)
            j *= self.A
        else:
            j = self.j_noA_k

        if(fourier and (not j.domain.fourier)):
            j = j.transform(target=self.codomain)
        elif((not fourier) and j.domain.fourier):
            j = j.transform(target=self.domain)

        return j

    def get_dd(self):

        m_matched = self.Mregu.inverse_times(self.j_noA_k)
        m_matched = m_matched.transform(self.domain)
        dd = self.RR(m_matched)

        return dd

    def get_matched(self):

        m_matched = self.Mregu.inverse_times(self.j_noA_k)
        m_matched = m_matched.transform(self.domain)

        return m_matched

    def get_Mdiag(self):
        return self.Mdiag

    def get_RR(self):
        return self.RR

    def get_NN(self):
        return self.NN

    def get_Mregu(self):
        return self.Mregu

    def _calc_UV_source(self, d=None, calc_norm=True):

        s_space = self.domain
        k_space = self.codomain
        u = self.u
        v = self.v
        varis = self.varis

        # calculate inverse_noise weighted dirty image
        # and the PSF in Fourier space if needed
        j_noA_k = grid_function(codomain=k_space,
                                inpoints=d/varis,
                                u=u, v=v,
                                abs_squared=False,
                                precision=5)

        if(calc_norm):
            psf = grid_function(codomain=k_space,
                                inpoints=np.ones(u.shape),
                                u=u, v=v,
                                abs_squared=False,
                                precision=5)

            # make PSF array into field
            psf = field(k_space,
                        target=s_space,
                        val=psf)
            # calculate PSF in real space
            psf = psf.transform()
            # the PSF should be normalized to len(u)/s_space.vol.prod()
            # at the maximum
            self._jnorm = psf.max() / (len(u)/s_space.vol.prod())
            # no need for the psf field anymore
            del(psf)

        # j needs to be normalized
        j_noA_k = field(k_space,
                        target=s_space,
                        val=j_noA_k/self._jnorm)

        self.j_noA_k = j_noA_k

    def _calc_UV_ops(self):

        s_space = self.domain
        k_space = self.codomain
        u = self.u
        v = self.v
        varis = self.varis
        A = self.A

        # RR and Mdiag are without a primary beam
        # that way the PB gets interpreted as part of the signal
        RR = grid_function(codomain=k_space,
                           inpoints=np.ones(u.shape),
                           u=u, v=v,
                           abs_squared=True,
                           precision=30)

        if(np.all(varis == varis[0])):
            Mdiag = RR/varis[0]
        else:
            Mdiag = grid_function(codomain=k_space,
                                  inpoints=1./varis,
                                  u=u, v=v,
                                  abs_squared=True,
                                  precision=30)

        RR = fast_Fourier_response(s_space, fdiag=RR, A=1., target=k_space)
        Mdiag = diagonal_operator(k_space, diag=Mdiag, bare=False)

        # calculate flux normalization factor

        # delta function
        delta = field(s_space, val=0.)
        # Dirac delta is 1/(pixel volume)
        delta[tuple(s_space.dim(split=True)/2)] = 1.

        # expression needs to be normalized to 1
        fac = (RR.Fdiag(delta) / len(u) * s_space.vol.prod()).max()

        Mdiag.val /= fac

        # create noise covariance operator

        # RR NN^-1 RR = Mdiag
        # so NN = RR^2/Mdiag

        # get regularized Mdiag without zero-entries
        Mregu = Mdiag.diag(bare=False)+0.
        Mregu[np.where(Mregu < Mregu.max()/1.e8)] = Mregu.max()/1.e8
        Mregu = diagonal_operator(k_space,
                                  diag=Mregu,
                                  bare=False)

        # calculate NN
        NN = RR.Fdiag.diag(bare=False)**2/Mregu.diag(bare=False)
        # again avoid zero-entries
        NN[np.where(NN < NN.max()/1.e8)] = NN.max()/1.e8

        # final invertible noise operator
        NN = diagonal_operator(k_space,
                               diag=NN,
                               bare=False)

        # add primary beam to response
        RR.set_A(A)

        self.Mdiag = Mdiag
        self.RR = RR
        self.NN = NN
        self.Mregu = Mregu


class lognormal_Hamiltonian(object):

    def __init__(self, S=None, N=None, R=None, d=None):

        if(isinstance(S, power_operator)):
            self.S = S
        else:
            raise TypeError("Error: invalid input for S.")

        if(isinstance(N, operator)):
            N = (N,)

        if(isinstance(N, list) or isinstance(N, tuple)):
            N = tuple(N)
            for ii in xrange(len(N)):
                if(not isinstance(N[ii], operator)):
                    raise TypeError("Error: N has to be an operator or a\
                        list/tuple of operators.")
                elif(not N[ii].sym):
                    raise ValueError("Error: N[{}] is not a valid noise\
                        covariance.".format(ii))
            self.N = N
        else:
            raise TypeError("Error: N has to be an operator or a\
                list/tuple of operators.")

        if(isinstance(R, operator)):
            R = (R,)

        if(isinstance(R, list) or isinstance(R, tuple)):
            R = tuple(R)
            if(len(R) != len(self.N)):
                raise ValueError("Error: not the same length of R and N.")
            for ii in xrange(len(R)):
                if(not isinstance(R[ii], operator)):
                    raise TypeError("Error: R has to be an operator or a\
                        list/tuple of operators.")
                elif(R[ii].target != self.N[ii].domain):
                    raise ValueError("Error: R[{}] is not a compatible\
                        with N[{}].".format(ii, ii))
                elif(R[ii].domain != R[0].domain):
                    raise ValueError("Error: R domains are not equal.")
            self.R = R
        else:
            raise TypeError("Error: R has to be an operator or a\
                list/tuple of operators.")

        if(isinstance(d, field)):
            d = (d,)

        if(isinstance(d, list) or isinstance(d, tuple)):
            d = tuple(d)
            if(len(d) != len(self.N)):
                raise ValueError("Error: not the same length of d and N.")
            for ii in xrange(len(d)):
                if(not isinstance(d[ii], field)):
                    raise TypeError("Error: d has to be a field or a\
                        list/tuple of fields.")
                elif(d[ii].domain != self.R[ii].target):
                    raise ValueError("Error: d[{}] is not a compatible\
                        with N[{}].".format(ii, ii))
            self.d = d
        else:
            raise TypeError("Error: d has to be a field or a\
                list/tuple of fields.")

        self.domain = R[0].domain
        self.codomain = S.domain

        # container saving last called field and its transform
        self.last_field = {}
        self.last_field['real'] = field(self.domain, target=self.codomain)
        self.last_field['fourier'] = field(self.codomain, target=self.domain)
        self.last_field['exp'] = field(self.domain,
                                       target=self.codomain,
                                       val=1.)
        self.last_field['data'] = [self.R[ii](self.last_field['exp'])
                                   for ii in xrange(len(self.R))]

    def _prep_fields(self, xin):

        if(xin.domain == self.domain):
            if(np.any(xin.val != self.last_field['real'].val)):
                self._calc_fields(xin)
        elif(xin.domain == self.codomain):
            if(np.any(xin.val != self.last_field['fourier'].val)):
                self._calc_fields(xin)
        else:
            raise ValueError("Error: incompatible domains.")

    def _calc_fields(self, xin):
        if(xin.domain == self.domain):
            x = xin+0.
            xtr = x.transform(target=self.codomain)
        elif(xin.domain == self.codomain):
            xtr = xin+0.
            x = xtr.transform(target=self.domain)

        ex = exp(x)
        data = [self.R[ii](ex)
                for ii in xrange(len(self.R))]

        self.last_field['real'] = x
        self.last_field['fourier'] = xtr
        self.last_field['exp'] = ex
        self.last_field['data'] = data

    def value(self, xin):

        if(not isinstance(xin, field)):
            xin = field(self.domain, val=xin, target=self.codomain)

        self._prep_fields(xin)

        xtr = self.last_field['fourier']
        Rex = self.last_field['data']

        S = self.S
        N = self.N
        d = self.d

        prior = 0.5 * xtr.dot(S.inverse_times(xtr))

        like = 0.
        for ii in xrange(len(N)):
            temp = Rex[ii]-d[ii]
            like += 0.5 * temp.dot(N[ii].inverse_times(temp))

        return prior + like

    def grad(self, xin):

        if(not isinstance(xin, field)):
            xin = field(self.domain, val=xin, target=self.codomain)
            fourier = False
        else:
            fourier = xin.domain.fourier

        self._prep_fields(xin)

        xtr = self.last_field['fourier']
        ex = self.last_field['exp']
        Rex = self.last_field['data']

        S = self.S
        N = self.N
        R = self.R
        d = self.d

        prior = S.inverse_times(xtr)

        like = field(self.domain, target=self.codomain)
        for ii in xrange(len(N)):
            temp = Rex[ii]-d[ii]
            temp = N[ii].inverse_times(temp)
            temp = R[ii].adjoint_times(temp)
            temp = ex*temp
            like += temp

        if(fourier):
            like = like.transform(target=self.codomain)
        else:
            prior = prior.transform(target=self.domain)

        result = prior + like
        result = result.weight(power=1)

        return result

    def gradval(self, xin):
        return self.grad(xin).val.flatten()

    def eggs(self, xin):
        return self.value(xin), self.grad(xin)

    def min_BFGS(self, x0, limii=2000, tol=1.e-4, note=False):

        if(not isinstance(x0, field)):
            x0 = field(self.domain, target=self.codomain, val=x0)
            fourier = False
        elif(x0.domain == self.codomain):
            x0 = x0.transform(target=self.domain)
            fourier = True
        elif(x0.domain == self.domain):
            fourier = False
        else:
            raise ValueError("Error: incompatible domains.")

        res = fmin_l_bfgs_b(self.value,
                            x0.val.flatten(),
                            fprime=self.gradval,
                            pgtol=tol,
                            factr=1e7,
                            maxiter=limii)
        if(note):
            print 'allowed BFGS iterations:', limii
            print 'performed BFGS iterations:', res[2]['nit']
            print 'function evaluated', res[2]['funcalls'], 'times.'
            if(res[2]['warnflag'] == 0):
                print 'BFGS converged.'
            elif(res[2]['warnflag'] == 1):
                print 'BFGS not converged.'
            else:
                print 'BFGS stopped:'
                print res[2]['task']

        res = field(self.domain, target=self.codomain, val=res[0])
        if(fourier):
            res = res.transform(target=self.codomain)

        return res

    def min_SD(self, x0, limii=2000, tol=1.e-4, note=False):

        if(not isinstance(x0, field)):
            x0 = field(self.domain, target=self.codomain, val=x0)
            fourier = False
        elif(x0.domain == self.codomain):
            x0 = x0.transform(target=self.domain)
            fourier = True
        elif(x0.domain == self.domain):
            fourier = False
        else:
            raise ValueError("Error: incompatible domains.")

        SD = steepest_descent(self.eggs, note=note)
        res = SD(x0, limii=limii)

        if(fourier):
            res = res.transform(target=self.codomain)

        return res

    def crude_Hessian(self, xin):

        if(not isinstance(xin, field)):
            xin = field(self.domain, val=xin, target=self.codomain)

        self._prep_fields(xin)

        ex = self.last_field['exp']

        S = self.S
        N = self.N
        R = self.R

        inv_diag = S.inverse_diag(bare=False)
        for ii in xrange(len(N)):
            temp1 = ex*R[ii].A.hat()
            temp1 = temp1.mean()
            temp2 = R[ii].Fdiag.diag(bare=False)
            temp2 = (temp1*temp2)**2 / N[ii].diag(bare=False)
            inv_diag = inv_diag + temp2

        D = diagonal_operator(S.domain, diag=1./inv_diag, bare=False)

        return D


def find_good_domain(beamsize, max_uv):

    maxdist = 0.5/max_uv

    Nmin = beamsize/maxdist

    pow2 = np.log(Nmin)/np.log(2)

    if(pow2 % 1 > 0.1):
        pow2 = int(pow2)+1
    else:
        pow2 = int(pow2)

    fac = float(2**pow2) / Nmin

    dist = maxdist/np.sqrt(fac)

    domain = rg_space(2**pow2, 2, dist=dist)

    return domain
