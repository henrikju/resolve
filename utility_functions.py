"""
helper_functions.py
Written by Henrik Junklewitz

helper_functions.py is an auxiliary file for resolve.py and belongs to the 
RESOLVE package. It provides all needed auxiliary functions for the inference
code except the functions that need explicit CASA input.

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
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
from nifty import *
from scipy.optimize import minimize
from operators import *
import resolve as rs


def Energy_cal(x0,j, S, M, rho0,params,xdomain,numparams,mid=0,end=0):

    if params.algorithm == 'ln-map':
        args = (j, S, M, rho0,params)
        x = field(xdomain,val=x0)
        en = energy(args)
        return en.H(x), en.gradH(x).val.flatten() * xdomain.vol.prod()
        
    elif params.algorithm == 'ln-map_u':        
             
        sval =np.array(mid)  
        uval =np.array(mid)  
        sval = x0[0:mid]    
        uval = x0[mid:end]
        s = field(xdomain, val = sval)
        u = field(xdomain, val = uval)
  
        args = (j, S, M, rho0, numparams.beta, numparams.eta,s,u)
        en = energy_mu(args)
        gs = en.gradH_s(s,u)
        gu = en.gradH_u(s,u)
        E = en.H(s,u)
        
        g=np.ones(end)    
        gsval = gs.val.flatten()* xdomain.vol.prod()
        guval = gu.val.flatten()* xdomain.vol.prod()
        g[0:mid] =gsval
        g[mid:end] =guval

        return E,g

def Energy_min(x0,j,S,M,rho0,params,numparams,limii=10, x1 = None): #todo add numparams im aufruf
    # x1 = u
    numparams.map_algo = 'BFGS'
    min_method = numparams.map_algo
    # es sollten alle moeglichkeiten von minimise funtionieren verwende numparam.map_algo =
    if params.algorithm == 'ln-map':

        call = callbackclass(params.save)
        res = minimize(Energy_cal,(x0).val.flatten(),\
            args=(j,S,M,rho0,params,x0.domain,numparams),method = min_method,jac = True,\
            options={"maxiter":limii},callback=call.callbackscipy)[0]

        return field(x0.domain,target=x0.target,val=res)
        
    elif params.algorithm == 'ln-map_u':
        
        mid =params.imsize*params.imsize
        end =2*params.imsize*params.imsize        
        X = np.ones(end) 
        mval = x0.val.flatten()
        uval = x1.val.flatten()
        X[0 :mid] = mval
        X[mid:end] = uval
        
        call = callbackstuff(params.save)
        res = minimize(Energy_cal,X,args=(j,S,M,rho0,params,x0.domain,numparams,mid,end),\
            method = min_method,jac = True,\
            options={"maxiter":limii},callback=call.callbackscipy)[0]            
            
        mval =res[0:mid] 
        uval =res[mid:end]
        m = field(x0.domain,target=x0.target,val=mval)
        u = field(x0.domain,target=x0.target,val=uval)

        return m, u
                
    else:
        print 'WARNING, BFGS only available yet for standard resolve'

def convert_CASA_to_RES(imagearray_fromCASA):
    """
    Converts on image from CASA to be used internally in RESOLVE. e.g. as a
    starting guess.
    """
    #with resepect to CASA, the imagearray is already rotated by 90 degrees
    #clockwise because of 0-point-shift between CASAIM/FITS and python.
    return np.transpose(np.rot90(imagearray_fromCASA,1))
    #return imagearray_fromCASA
    
def convert_RES_to_CASA(imagearray_fromRES,FITS=False):
    """
    Converts on image from RESOLVE to be used externally, e.g. as an end result
    image.
    """
    #Internally the image only needs to be back-transposed because the FITS
    #output will automatically rotate the image
    if FITS:
        return np.transpose(imagearray_fromRES)
    #For direct comparison, all matplotlib images are correctly changed to
    #reflect the original CASA output
    else:
        return np.rot90(np.transpose(imagearray_fromRES),-1)
        #return imagearray_fromRES

def callbackfunc(x, i):
    
    if i%gcallback == 0:
        print ' Callback at iteration' + str(i)
        
        if gsave:
           pl.figure()
           pl.imshow(exp(x))
           pl.colorbar()
           pl.title('Iteration' + str(i))
           pl.savefig('resolve_output_' + str(gsave)+ \
               "/last_iterations/" + 'iteration'+str(i))
           np.save('resolve_output_' + str(gsave)+ \
                   "/last_iterations/" + 'iteration' + str(i),exp(x))
           pl.close()
               
def callbackfunc_u(x, i):
    
    if i%gcallback == 0:
        print 'Callback at point source iteration' + str(i) 
        
        if gsave:
           pl.figure()
           pl.imshow(exp(x))
           pl.colorbar()
           pl.title('Iteration' + str(i)+'_u')
           pl.savefig("resolve_output_"+ str(gsave)+"/last_iterations/iteration"+str(i)+"_expu")
           np.save("resolve_output_"+ str(gsave)+"/last_iterations/iteration"+str(i)+"_expu",exp(x))
               
def callbackfunc_m(x, i):
    
    if i%gcallback == 0:
        print 'Callback at extended source iteration' + str(i) 
        
        if gsave:
           pl.figure()
           pl.imshow(exp(x))
           pl.colorbar()
           pl.title('Iteration' + str(i)+'_m')
           pl.savefig("resolve_output_"+ str(gsave)+"/last_iterations/iteration"+str(i)+"_expm")
           np.save("resolve_output_"+ str(gsave)+"/last_iterations/iteration"+str(i)+"_expm",exp(x))

class callbackclass(object):
    
    def __init__(self, save):
        
        self.i = 0
        self.savename = save
        
    def callbackscipy(self,x):
        
        np.save("resolve_output_"+str(self.savename)+"/last_iterations/iteration"+str(self.i)+"_lbfgs",exp(x))
        self.i += 1

def callbackbfgs(x):

    np.save("resolve_output_"+ str(gsave)+"/last_iterations/iteration"+"_lbfgs",exp(x))

def save_results(value,title,fname,log = None,value2 = None, \
    value3= None, plotpar = None, rho0 = 1., twoplot=False):
    
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
                pl.imshow(convert_RES_to_CASA(value) * rho0)
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
                pl.imshow(value * rho0)
                pl.colorbar()
            else:
                pl.plot(value,value2)
        pl.savefig(fname + ".png")          
        
    pl.close
    
    # save data as npy-file
    if len(np.shape(value)) > 1:
        np.save(fname,value * rho0)
    else:
        np.save(fname,value2)

def load_numpy_data(msfn, logger):
    
    try:
        vis = np.load(msfn + '_vis.npy')
        sigma = np.load(msfn + '_sigma.npy')
        u = np.load(msfn + '_u.npy')
        v = np.load(msfn + '_v.npy')
        freqs = np.load(msfn + '_freq.npy')
        nchan = np.load(msfn + '_nchan.npy')    
        nspw = np.load(msfn + '_nspw.npy')
        summary = np.load(msfn + '_sum.npy')
        nvis = np.load(msfn + '_nvis.npy')

    except IOError:
        logger.failure('No numpy file exists in the working directory with '\
            + 'the suffix ' + msfn)

    return vis, sigma, u, v, freqs, nchan, nspw, nvis, summary
    
def update_globvars(gsavein, gcallbackin):

    global gsave
    global gcallback
    gsave = gsavein
    gcallback = gcallbackin


    
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
        

    
    
    
               
