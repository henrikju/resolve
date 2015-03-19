import numpy as np
import gfft as gf
from nifty import *
import pylab as pl

class response(operator):
    """
    """

    def _multiply(self, expI):
        """
        """
        dx = self.domain.dist()[0] 
        Nx = self.domain.dim(split = True)[0]
        dy = self.domain.dist()[1] 
        Ny = self.domain.dim(split = True)[1]
        u = self.para[0]
        v = self.para[1]
        normloop = self.para[3]
        A = self.para[2]
        
        #Volumefactor from R(Vs) if the signal is a density       
#        if normloop == False:            
#            expI = expI.weight()

        
        dgridvis = gf.gfft(A * expI.val, in_ax = [(dx, Nx), (dy, Ny)], \
            out_ax = [u,v], ftmachine = 'fft', in_zero_center = True, \
            out_zero_center = True, enforce_hermitian_symmetry = True, W = 6, \
            alpha=1.5 ,verbose=False)
            
        if normloop == False:
            dgridvis /=  normalize_R(self.domain, self.target, u, v, A)
            #pass
        
        return field(domain = self.target, val = dgridvis, \
            datatype = np.complex128)
            

    def _adjoint_multiply(self, x):
        """
        """
        u = self.para[0]
        v = self.para[1]
        dx = self.domain.dist()[0] 
        Nx = self.domain.dim(split = True)[0]
        dy = self.domain.dist()[1] 
        Ny = self.domain.dim(split = True)[1]
        normloop = self.para[3]
        A = self.para[2]
        
        gridvis = gf.gfft(x.val, in_ax =  [u,v], out_ax = [(dx, Nx), (dy, Ny)],\
            ftmachine = 'ifft', in_zero_center = True, \
            out_zero_center = True, enforce_hermitian_symmetry = True, W = 6, \
            alpha=1.5 ,verbose=False)

            
#        gridvis = self.domain.calc_weight(gridvis,power=1)
#        gridvis *= self.domain.dim(split=False)
#        gridvis /= dx * dy 
        
        if normloop == False:
            gridvis /=  normalize_Rd(self.domain, self.target, u, v, A)
            #pass
        
        expI = field(domain = self.domain, val = gridvis)
        
        #Volumefactor from R(Vs) for "signal is not a density"       
        if normloop == False:            
            expI = expI.weight(power=-1)
        
        return A * expI
        
        
class response_mfs(operator):
    """
    """

    def _multiply(self, exp_I, a = 0., mode = 'normal'):
        """
        """
            
        dx = self.domain.dist()[0] 
        Nx = self.domain.dim(split = True)[0]
        dy = self.domain.dist()[1] 
        Ny = self.domain.dim(split = True)[1]
        u = self.para[0]
        v = self.para[1]
        nspw = self.para[3]
        nchan = self.para[4]
        A = self.para[2]
        normloop = self.para[5]
        freq = self.para[6]
        restspw = self.para[7]
        restchan = self.para[8]
        
        
            
        #Volumefactor from R(Vs) if signal is a density        
#        if normloop == False:            
#            expI = expI.weight()
         
        visval = np.array([])
        for i in range(nspw):
            for j in range(nchan):
                temp = (A[i,j] * exp_I * exp(-log(freq[i,j]/freq[restspw,restchan])*a))\
                    
                
                temp = gf.gfft(temp, in_ax = [(dx, Nx), \
                    (dy, Ny)], out_ax = [u[i,j],v[i,j]], ftmachine = 'fft', \
                    in_zero_center = True, out_zero_center = True, \
                    enforce_hermitian_symmetry = True, W = 6, alpha=1.5 , \
                    verbose=False)
                
                if normloop == False:
                    temp /=  normalize_R_mfs(self.domain, u, v, i,j,A)
                visval = np.append(visval,temp)
            #pass
        
        return field(domain = self.target, val = visval.flatten(), \
            datatype = np.complex128)
            

    def _adjoint_multiply(self, x, a = 0., mode = 'normal'):
        """
        """

        u = self.para[0]
        v = self.para[1]
        A = self.para[2]
        nspw = self.para[3]
        nchan = self.para[4]
        dx = self.domain.dist()[0] 
        Nx = self.domain.dim(split = True)[0]
        dy = self.domain.dist()[1] 
        Ny = self.domain.dim(split = True)[1]
        normloop = self.para[5]
        freq = self.para[6]
        d_space = x.domain
        restspw = self.para[7]
        restchan = self.para[8]
        nvis = self.para[9]
        
        if mode == 'normal':        
        
            exp_I = np.zeros((Nx,Ny))
            
            vispoint = x.val.reshape(nspw,nchan,nvis)

            
            for i in range(nspw):
                for j in range(nchan):
            
                    
                    gridvis = gf.gfft(vispoint[i,j], in_ax =  [u[i,j],v[i,j]], \
                            out_ax = [(dx, Nx), (dy, Ny)], ftmachine = 'ifft', \
                            in_zero_center = True, out_zero_center = True, \
                            enforce_hermitian_symmetry = True, W = 6, alpha=1.5 ,\
                            verbose=False)
            
                    if normloop == False:
                        gridvis /=  normalize_Rd_mfs(self.domain, u, v, \
                                i, j, A, mode)

                    exp_I += A[i,j] * gridvis * exp(-log(freq[i,j]/freq[restspw,restchan])\
                            *a)
                            
            
            expI = field(domain = self.domain, val = exp_I)
            
            #Volumefactor from R(Vs)        
            if normloop == False:            
                expI = expI.weight(power=-1)
            
            return expI

        if mode == 'grad':        
        
            exp_I = np.zeros((Nx,Ny))
            
            vispoint = x.val.reshape(nspw,nchan,nvis)
            
            
            for i in range(nspw):
                for j in range(nchan):
            
                    gridvis = gf.gfft(vispoint[i,j], in_ax =  [u[i,j],v[i,j]], \
                            out_ax = [(dx, Nx), (dy, Ny)], ftmachine = 'ifft', \
                            in_zero_center = True, out_zero_center = True, \
                            enforce_hermitian_symmetry = True, W = 6, alpha=1.5 ,\
                            verbose=False)
            
                    if normloop == False:
                        gridvis /=  normalize_Rd_mfs(self.domain, u, v, \
                            i,j, A,mode)
                    exp_I += A[i,j] * gridvis * exp(-log(freq[i,j]/\
                        freq[restspw,restchan])*a) * \
                        (-log(freq[i,j]/freq[restspw,restchan]))
                
            
            expI = field(domain = self.domain, val = exp_I)
            
            #Volumefactor from R(Vs)        
            if normloop == False:            
                expI = expI.weight(power=-1)
            
            return expI
            
        
        if mode == 'D':        
        
            exp_I = np.zeros((Nx,Ny))
            
            vispoint = x.val.reshape(nspw,nchan,nvis)
            
            
            for i in range(nspw):
                for j in range(nchan):
            
                    gridvis = gf.gfft(vispoint[i,j], in_ax =  [u[i,j],v[i,j]], \
                            out_ax = [(dx, Nx), (dy, Ny)], ftmachine = 'ifft', \
                            in_zero_center = True, out_zero_center = True, \
                            enforce_hermitian_symmetry = True, W = 6, alpha=1.5 ,\
                            verbose=False)
            
                    if normloop == False:
                        gridvis /=  normalize_Rd_mfs(self.domain, u, v, \
                                i, j, A, mode)
                    exp_I += A[i,j] * gridvis * exp(-log(freq[i,j]/freq[restspw,restchan])\
                            *a) * (-log(freq[i,j]/freq[restspw,restchan]))**2
                
            
            expI = field(domain = self.domain, val = exp_I)
            
            #Volumefactor from R(Vs)        
            if normloop == False:            
                expI = expI.weight(power=-1)
            
            return expI


class response_mfs_normloop(operator):
    """
    """

    def _multiply(self, exp_I):
        """
        """
            
        dx = self.domain.dist()[0] 
        Nx = self.domain.dim(split = True)[0]
        dy = self.domain.dist()[1] 
        Ny = self.domain.dim(split = True)[1]
        u = self.para[0]
        v = self.para[1]
        i = self.para[3]
        j = self.para[4]
        A = self.para[5]
        
        
        #Volumefactor from R(Vs) if signal is a density        
#        if normloop == False:            
#            expI = expI.weight()
         
        temp = A[i,j] * (exp_I).val #* exp(-log(freq[i]/freq[0])*a)).val
        
        dgridvis = gf.gfft(temp, in_ax = [(dx, Nx), (dy, Ny)], \
                           out_ax = [u[i,j],v[i,j]], ftmachine = 'fft', in_zero_center = True, \
            out_zero_center = True, enforce_hermitian_symmetry = True, W = 6, \
            alpha=1.5 ,verbose=False)
        
        
        return field(domain = self.target, val = dgridvis, \
            datatype = np.complex128)
            

    def _adjoint_multiply(self, x, mode = 'normal'):
        """
        """

        u = self.para[0]
        v = self.para[1]
        dx = self.domain.dist()[0] 
        Nx = self.domain.dim(split = True)[0]
        dy = self.domain.dist()[1] 
        Ny = self.domain.dim(split = True)[1]
        i = self.para[3]
        j = self.para[4]
        A = self.para[5]

        
        if mode == 'normal':
            

            gridvis = gf.gfft(x.val, in_ax =  [u[i,j],v[i,j]], out_ax = [(dx, Nx), (dy, Ny)],\
                ftmachine = 'ifft', in_zero_center = True, \
                out_zero_center = True, enforce_hermitian_symmetry = True, W = 6, \
                alpha=1.5 ,verbose=False)

            #expI = field(domain = self.domain, val = gridvis)

            expI = A[i,j] * field(domain = self.domain, val = gridvis) #* exp(-log(freq[i]/freq[0])*a) #* \
                    #log(freq[i]/freq[0])**(bpower))

            #Volumefactor from R(Vs)        
    #        if normloop == False:            
    #            expI = expI.weight(power=-1)
    
        if mode == 'grad':
        
            gridvis = gf.gfft(x.val, in_ax =  [u[i,j],v[i,j]], out_ax = [(dx, Nx), (dy, Ny)],\
                ftmachine = 'ifft', in_zero_center = True, \
                out_zero_center = True, enforce_hermitian_symmetry = True, W = 6, \
                alpha=1.5 ,verbose=False)
            
            #expI = field(domain = self.domain, val = gridvis)
    
            expI = A[i,j] * field(domain = self.domain, val = gridvis) # exp(-log(freq[i]/freq[0])*a) * \
                    #-log(freq[i]/freq[0])
            
            #Volumefactor from R(Vs)        
    #        if normloop == False:            
    #            expI = expI.weight(power=-1)
    
        if mode == 'D':
            
            gridvis = gf.gfft(x.val, in_ax =  [u[i,j],v[i,j]], \
                    out_ax = [(dx, Nx), (dy, Ny)], ftmachine = 'ifft', \
                    in_zero_center = True, out_zero_center = True, \
                    enforce_hermitian_symmetry = True, W = 6, alpha=1.5 ,\
                    verbose=False)
            
                
            expI = A[i,j] * field(domain = self.domain, val = gridvis)
                #pass

        
        return expI                   
       
def normalize_R(s_space, d_space, u, v, A):
    """
    """
    
    R_norm = response(s_space, sym=False, imp=True, target=d_space, para= \
            [u, v, A, True])
    N = s_space.dim(split = True)
    v = np.zeros(N)
    v[N[0]/2, N[1]/2] = 1.
    norm = R_norm(v)
    return np.max(norm)
    
#    R_norm = response(s_space, sym=False, imp=True, target=d_space, para=[u, v, True])
#    v = np.ones(s_space.dim(split=True))
#    psf = R_norm(v)
#    return np.max(psf)


def normalize_Rd(s_space, d_space, u, v, A):
    """
    """
    
    R_norm = response(s_space, sym=False, imp=True, target=d_space, \
                      para=[u, v, A, True])
    vec = np.ones(d_space.dim(),dtype=np.complex128)
    psf = R_norm.adjoint_times(vec)
    return np.max(psf)
        
def normalize_R_mfs(s_space, u, v, i,j,A):
    """
    """
    d_space = point_space(len(u[i,j]),datatype=np.complex128)
    R_norm = response_mfs_normloop(s_space, sym=False, imp=True, target=d_space, \
           para=[u, v, True, i,j,A])
    N = s_space.dim(split = True)
    v = np.zeros(N)
    v[N[0]/2, N[1]/2] = 1.
    norm = R_norm(v)
    return np.max(norm)
    
#    R_norm = response(s_space, sym=False, imp=True, target=d_space, para=[u, v, True])
#    v = np.ones(s_space.dim(split=True))
#    psf = R_norm(v)
#    return np.max(psf)


def normalize_Rd_mfs(s_space, u, v, i, j, A, mode):
    """
    """

    d_space = point_space(len(u[i,j]),datatype=np.complex128)
    R_norm = response_mfs_normloop(s_space, sym=False, imp=True, target=d_space, \
           para=[u, v, True, i,j,A])
    vec = field(d_space, val = np.ones(len(u[i,j])),dtype=np.complex128)
    psf = R_norm.adjoint_times(vec, mode=mode)

    return np.max(psf)        
