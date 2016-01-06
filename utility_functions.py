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


def convert_CASA_to_RES(imagearray_fromCASA):
    """
    Converts on image from CASA to be used internally in RESOLVE. e.g. as a
    starting guess.
    """
    #with resepect to CASA, the imagearray is already rotated by 90 degrees
    #clockwise because of 0-point-shift between CASAIM/FITS and python.
    return np.rot90(np.transpose(imagearray_fromCASA),-1)
    
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
        return np.transpose(np.rot90(imagearray_fromRES,-1))

def callbackfunc(x, i):
    
    if i%gcallback == 0:
        print 'Callback at iteration' + str(i)
        
        if gsave:
           pl.figure()
           pl.imshow(convert_RES_to_CASA(exp(x)))
           pl.colorbar()
           pl.title('Iteration' + str(i))
           pl.savefig('resolve_output_' + str(gsave)+ \
               "/last_iterations/" + 'iteration'+str(i))
           np.save('resolve_output_' + str(gsave)+ \
               "/last_iterations/" + 'iteration' + str(i),x)
               
def callbackfunc_u(x, i):
    
    if i%gcallback == 0:
        print 'Callback at point source iteration' + str(i) 
        
        if gsave:
           pl.figure()
           pl.imshow(convert_CASA_to_RES(exp(x)))
           pl.colorbar()
           pl.title('Iteration' + str(i)+'_u')
           pl.savefig("resolve_output_"+ str(gsave)+"/last_iterations/iteration"+str(i)+"_expu")
           np.save("resolve_output_"+ str(gsave)+"/last_iterations/iteration"+str(i)+"_expu",x)
               
def callbackfunc_m(x, i):
    
    if i%gcallback == 0:
        print 'Callback at extended source iteration' + str(i) 
        
        if gsave:
           pl.figure()
           pl.imshow(convert_CASA_to_RES(exp(x)))
           pl.colorbar()
           pl.title('Iteration' + str(i)+'_m')
           pl.savefig("resolve_output_"+ str(gsave)+"/last_iterations/iteration"+str(i)+"_expm")
           np.save("resolve_output_"+ str(gsave)+"/last_iterations/iteration"+str(i)+"_expm",x)

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
                pl.imshow(convert_RES_to_CASA(value) * rho0)
                pl.colorbar()
            else:
                pl.plot(value,value2)
        pl.savefig(fname + ".png")          
        
    pl.close
    
    # save data as npy-file
    if len(np.shape(value)) > 1:
        np.save(fname,convert_RES_to_CASA(value) * rho0)
    else:
        np.save(fname,value)

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
        

    
    
    
               