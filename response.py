"""
general_response.py
Written by Henrik Junklewitz

General_response.py is part of the RESOLVE package and provides several 
routines and definitions for radio interferometric response functions.

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

import numpy as np
import gfft as gf
from nifty import *
import pylab as pl

#------------------------------------------------------------------------------
# single band response classes
#------------------------------------------------------------------------------

# GFFT gridded response

class response(operator):
    """
    """

    def __init__(self,domain,target,u,v,A,mode='gfft'):

        #necessary nifty inputs
        self.sym = False
        self.uni = False
        self.imp = True
        self.domain = domain
        self.target = target

        self.u = u
        self.v = v

        #primary beam
        self.A = A

        self.mode = mode

        #see below for explanation
        self.adjointfactor = 1.
        #"blind" normalization of R using a known delta peak of 1 at image center
        self.normR = 1.
        N = domain.dim(split = True)
        tempim = np.zeros(N)
        tempim[N[0]/2, N[1]/2] = 1. #/  domain.vol.prod()
        self.normR = np.max(self.times(tempim))
    
        #"blind" normalization of Rd using a known delta peak of 1 at image center
        self.normRd = 1.
        tempd = np.ones(target.dim(),dtype=np.complex128)
        self.normRd = np.max(self.adjoint_times(tempd)) #/ domain.vol.prod()
        
        #Now save settings to make the DFT/IDFT "artificially" adjoint for nifty. Note
        #that the natural adjointness of the FT is broken for a direct FT on irregular
        #grids. There ought to be a more elegant way...
        self.adjointfactor = abs(tempd.dot(self.times(tempim))) / self.adjoint_times(tempd).dot(tempim)
        
        del tempim
        del tempd

    def _multiply(self, expI):
        """
        """
        dx = self.domain.dist()[0] 
        Nx = self.domain.dim(split = True)[0]
        dy = self.domain.dist()[1] 
        Ny = self.domain.dim(split = True)[1]


        if self.mode == 'gfft':
            dgridvis = gf.gfft(self.A * expI.val, in_ax = [(dx, Nx), (dy, Ny)], \
                out_ax = [self.u,self.v], ftmachine = 'fft', in_zero_center = True, \
                out_zero_center = True, enforce_hermitian_symmetry = True, W = 6, \
                alpha=1.5 ,verbose=False)
        if self.mode == 'dft':
            x = np.arange(Nx) * dx - 0.5 * Nx * dx
            y = np.arange(Ny) * dy - 0.5 * Ny * dy
            dgridvis = directft2d(self.A * expI.val,x,y,self.u,self.v)
            return field(domain = self.target, val = dgridvis, \
            datatype = np.complex128)  

        #blind  normalization using the fact the a delta function of strength 1
        #should have controlable behavior under a FT
        dgridvis /=  self.normR
            
        #fft normalization due to gridding routines. Setting should be correct 
        #for W=6 and alpha=1.5. Anything left should be taken care of by
        #the blind normalization.
        dgridvis /= (2.12590966146)**2
        
        return field(domain = self.target, val = dgridvis, \
            datatype = np.complex128)
            

    def _adjoint_multiply(self, x):
        """
        """

        dx = self.domain.dist()[0] 
        Nx = self.domain.dim(split = True)[0]
        dy = self.domain.dist()[1] 
        Ny = self.domain.dim(split = True)[1]
        
        if self.mode == 'gfft':
            gridvis = gf.gfft(x.val, in_ax =  [self.u,self.v], out_ax = \
                [(dx, Nx), (dy, Ny)],ftmachine = 'ifft', in_zero_center = True, \
                out_zero_center = True, enforce_hermitian_symmetry = True, W = 6, \
                alpha=1.5 ,verbose=False)
        elif self.mode == 'dft':
            x = np.arange(Nx) * dx - 0.5 * Nx * dx
            y = np.arange(Ny) * dy - 0.5 * Ny * dy
            gridvis = directift2d(x.val,self.u,self.v,x,y)
            return field(domain = self.domain, val = self.A * np.real(gridvis))
            
        expI = field(domain = self.domain, val = np.real(gridvis))
        
        #flux normalization
        expI /=  self.normRd

        #fft normalization due to gridding routines. The factor 4 should be correct
        #for W=6 and alpha=1.5. Anything left should be taken care of by
        #the blind normalization.       
        expI = expI *Nx *Ny / self.target.num() / 4. * self.adjointfactor
        
        return self.A * expI



# direct FT function for comparison and diagonstics

def directft2d(fx,x1,x2,k1,k2):

        Nvis = len(k1)
        output = np.zeros(Nvis,dtype='complex128')
        for m in range(Nvis):
            for i in range(len(x1)):
                for j in range(len(x2)):
                    output[m] += fx[i,j] * np.exp(-2 * np.pi * np.complex(0,1) * (x1[i] * k1[m] + x2[j] * k2[m]))

        return output

def directift2d(fk,k1,k2,x1,x2):

        Nvis = len(fk)
        out = np.zeros((len(x1),len(x2)))
        for i in range(len(x1)):
            for j in range(len(x2)):
                for m in range(Nvis):
                    out[i,j] += np.real(fk[m] * np.exp(2 * np.pi * np.complex(0,1) * (x1[i] * k1[m] + x2[j] * k2[m])))
        return out / Nvis


#--------------------------------------------------------------------------------------------------
# Wideband response classes
#--------------------------------------------------------------------------------------------------
        
        
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
                    
                if normloop == False:
                    temp /= len(temp)
                    
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
