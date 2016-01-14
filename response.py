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

    def __init__(self,domain,target,u,v,A,mode='gfft',wscOP=None):

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

        if mode == 'wsclean':
            self.wscOP = wscOP

        else:
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
        elif self.mode == 'dft':
            x = np.arange(Nx) * dx - 0.5 * Nx * dx
            y = np.arange(Ny) * dy - 0.5 * Ny * dy
            dgridvis = directft2d(self.A * expI.val,x,y,self.u,self.v)
            return field(domain = self.target, val = dgridvis, \
            datatype = np.complex128)
        elif self.mode == 'wsclean':
            dgridvis = field(self.target,val=0.).val
            self.wscOP.forward(dgridvis,self.A * expI.val.reshape(Nx*Ny))
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
        
        elif self.mode == 'wsclean':
            gridvis = field(self.domain,val=0.).val
            self.wscOP.backward(gridvis.reshape(Nx*Ny),x.val)
            return field(domain = self.domain, val = self.A * gridvis)
    
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

    def __init__(self,domain,target,u,v,A,nspw,nchan,nvis,freq,refspw,refchan,mode='gfft', wscOP=None):

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

        #frequency settings
        self.freq = freq
        self.nspw = nspw
        self.nchan = nchan
        self.nvis = nvis
        self.refspw = refspw
        self.refchan = refchan

        self.mode = mode

        if mode == 'wsclean':
            #self.wscOP = wscOP
            print 'Wideband-wsclean mode not yet implementet'

        else:
            #see below for explanation
            self.adjointfactor = np.ones((nspw,nchan))
            self.normR = np.ones((nspw,nchan))
            self.normRd = np.ones((nspw,nchan))
            for i in range(nspw):
                for j in range(nchan):
                    Rfreq = response(domain,point_space(len(u[i,j]),datatype=np.complex128)\
                        ,u[i,j],v[i,j],A[i,j],mode='gfft',wscOP=None)
                    #"blind" normalization of R using a known delta peak of 1 at image center
                    N = domain.dim(split = True)
                    tempim = np.zeros(N)
                    tempim[N[0]/2, N[1]/2] = 1. #/  domain.vol.prod()
                    self.normR[i,j] = np.max(Rfreq.times(tempim))

                    #"blind" normalization of Rd using a known delta peak of 1 at image center
                    tempd = np.ones(point_space(len(u[i,j]),datatype=np.complex128).dim(),dtype=np.complex128)
                    self.normRd[i,j] = np.max(Rfreq.adjoint_times(tempd)) #/ domain.vol.prod()

                    #Now save settings to make the DFT/IDFT "artificially" adjoint for nifty. Note
                    #that the natural adjointness of the FT is broken for a direct FT on irregular
                    #grids. There ought to be a more elegant way...
                    self.adjointfactor[i,j] = abs(tempd.dot(Rtemp.times(tempim)))\
                        / Rtemp.adjoint_times(tempd).dot(tempim)

            del Rtemp
            del tempim
            del tempd

    def _multiply(self, exp_I, a = 0., minmode = 'normal'):
        """
        """
            
        dx = self.domain.dist()[0] 
        Nx = self.domain.dim(split = True)[0]
        dy = self.domain.dist()[1] 
        Ny = self.domain.dim(split = True)[1}
        
        if self.mode == 'gfft':
            visval = np.array([])
            for i in range(self.nspw):
                for j in range(self.nchan):
                    temp = (self.A[i,j] * exp_I * exp(-log(self.freq[i,j]/self.freq[self.restspw,self.restchan])*a))
                    
                
                    temp = gf.gfft(temp, in_ax = [(dx, Nx), \
                        (dy, Ny)], out_ax = [u[i,j],v[i,j]], ftmachine = 'fft', \
                        in_zero_center = True, out_zero_center = True, \
                        enforce_hermitian_symmetry = True, W = 6, alpha=1.5 , \
                        verbose=False)
                
                    #blind normalization using the fact the a delta function of strength 1
                    #should have controlable behavior under a FT
                    temp /=  self.normR[i,j]

                    #fft normalization due to gridding routines. Setting should be correct
                    #for W=6 and alpha=1.5. Anything left should be taken care of by
                    #the blind normalization.
                    temp /= (2.12590966146)**2
                    
                    visval = np.append(visval,temp)
        
            return field(domain = self.target, val = visval.flatten(), \
                datatype = np.complex128)

        else:
            print 'No other mode than gfft available for wideband resolve'
            

    def _adjoint_multiply(self, x, a = 0., mode = 'normal'):
        """
        """

        dx = self.domain.dist()[0] 
        Nx = self.domain.dim(split = True)[0]
        dy = self.domain.dist()[1] 
        Ny = self.domain.dim(split = True)[1]
        d_space = x.domain
        
        if mode == 'normal':        
        
            exp_I = np.zeros((Nx,Ny))
            
            vispoint = x.val.reshape(self.nspw,self.nchan,self.nvis)

            for i in range(self.nspw):
                for j in range(self.nchan):
            
                    gridvis = gf.gfft(vispoint[i,j], in_ax =  [self.u[i,j],self.v[i,j]], \
                            out_ax = [(dx, Nx), (dy, Ny)], ftmachine = 'ifft', \
                            in_zero_center = True, out_zero_center = True, \
                            enforce_hermitian_symmetry = True, W = 6, alpha=1.5 ,\
                            verbose=False)
            
                    #flux normalization
                    expI /=  self.normRd[i,j]

                    #fft normalization due to gridding routines. The factor 4 should be correct
                    #for W=6 and alpha=1.5. Anything left should be taken care of by
                    #the blind normalization.
                    expI = expI *Nx *Ny / self.target.num() / 4. * self.adjointfactor[i,j]

                    exp_I += self.A[i,j] * gridvis * exp(-log(self.freq[i,j]/self.freq[self.restspw,self.restchan])\
                            *a)
                            
            
            expI = field(domain = self.domain, val = exp_I)
            
            return expI

        if mode == 'grad':        
        
            exp_I = np.zeros((Nx,Ny))
            
            vispoint = x.val.reshape(nspw,nchan,nvis)
            
            for i in range(self.nspw):
                for j in range(self.nchan):

                    gridvis = gf.gfft(vispoint[i,j], in_ax =  [self.u[i,j],self.v[i,j]], \
                            out_ax = [(dx, Nx), (dy, Ny)], ftmachine = 'ifft', \
                            in_zero_center = True, out_zero_center = True, \
                            enforce_hermitian_symmetry = True, W = 6, alpha=1.5 ,\
                            verbose=False)

                    #flux normalization
                    expI /=  self.normRd[i,j]

                    #fft normalization due to gridding routines. The factor 4 should be correct
                    #for W=6 and alpha=1.5. Anything left should be taken care of by
                    #the blind normalization.
                    expI = expI *Nx *Ny / self.target.num() / 4. * self.adjointfactor[i,j]
                    
                    exp_I += self.A[i,j] * gridvis * exp(-log(self.freq[i,j]/\
                        self.freq[self.restspw,self.restchan])*a) * \
                        (-log(self.freq[i,j]/self.freq[self.restspw,self.restchan]))
                
            
            expI = field(domain = self.domain, val = exp_I)
            
            return expI
            
        
        if mode == 'D':        
        
            exp_I = np.zeros((Nx,Ny))
            
            vispoint = x.val.reshape(nspw,nchan,nvis)
            
            for i in range(self.nspw):
                for j in range(self.nchan):

                    gridvis = gf.gfft(vispoint[i,j], in_ax =  [self.u[i,j],self.v[i,j]], \
                            out_ax = [(dx, Nx), (dy, Ny)], ftmachine = 'ifft', \
                            in_zero_center = True, out_zero_center = True, \
                            enforce_hermitian_symmetry = True, W = 6, alpha=1.5 ,\
                            verbose=False)

                    #flux normalization
                    expI /=  self.normRd[i,j]

                    #fft normalization due to gridding routines. The factor 4 should be correct
                    #for W=6 and alpha=1.5. Anything left should be taken care of by
                    #the blind normalization.
                    expI = expI *Nx *Ny / self.target.num() / 4. * self.adjointfactor[i,j]

                    exp_I += self.A[i,j] * gridvis * exp(-log(self.freq[i,j]/self.freq[self.restspw,self.restchan])\
                            *a) * (-log(self.freq[i,j]/self.freq[self.restspw,self.restchan]))**2
                
            
            expI = field(domain = self.domain, val = exp_I)
            
            return expI

