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
import pylab as pl
import numpy as np
from nifty import *
from operators import *
import resolve as rs
import resolve_parser as pars


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

def uncertainty_functions(m,Dhat,smooth=0.):
    
    
    Dhat /= Dhat.domain.vol.prod()
    
    save_results(Dhat.val,"D", \
                'resolve_output_' + str(params.save) +\
                "/D_reconstructions/" + params.save + "_D")
    
    #Dhat smoothing            
    Dhat = Dhat.smooth(sigma=smooth)
    save_results(Dhat.val,"Dsmooth", \
                'resolve_output_' + str(params.save) +\
                "/D_reconstructions/" + params.save + "_Dsmooth")
    
    #Relative uncertainty under Saddlepoint approximation
    relun = np.sqrt(exp(Dhat)-1)
    save_results(relun,"relative uncertainty", \
                'resolve_output_' + str(params.save) +\
                "/D_reconstructions/" + params.save + "_relun")
    
    #Absolute Uncertainty under Saddlepoint approximation
    absun = exp(m) * relun
    save_results(absun.val,"absolute uncertainty", \
                'resolve_output_' + str(params.save) +\
                "/D_reconstructions/" + params.save + "_absun")
                
    #Max/Min value starting from Gaussian field variance
    maxval = exp(m+sqrt(Dhat))
    minval = exp(m-sqrt(Dhat))
    save_results(maxval,"maximum exponentiated Gaussian field", \
                'resolve_output_' + str(params.save) +\
                "/D_reconstructions/" + params.save + "_maxval")
    save_results(minval,"minimum exponentiated Gaussian field", \
                'resolve_output_' + str(params.save) +\
                "/D_reconstructions/" + params.save + "_minval")


class callbackclass(object):

    def __init__(self, save, map_algo, runlist_element, callbackfrequency,
                 imsize):

        self.iter = 0
        self.savename = save
        self.method = map_algo
        self.runlist_element = runlist_element
        self.callbackfrequency = callbackfrequency
        self.imsize = imsize

    def callbackscipy(self, x):

        if self.iter % self.callbackfrequency == 0:
            print 'Callback at iteration' + str(self.iter)
            if self.savename:
                x = x.reshape(self.imsize, self.imsize)

                pl.figure()
                pl.imshow(exp(x))
                pl.colorbar()
                pl.title('Iteration' + str(self.iter)+'_m')
                pl.savefig("resolve_output_" + str(self.savename) +
                           "/last_iterations/iteration" + str(self.iter) + "_"
                           + self.runlist_element + "_" + self.method)
                np.save("resolve_output_" + str(self.savename) +
                        "/last_iterations/iteration" + str(self.iter) + "_" +
                        self.runlist_element + "_" + self.method, exp(x))
                pl.close()

        self.iter += 1

    def callbacknifty(self, x, i):

        if i % self.callbackfrequency == 0:
            print 'Callback at iteration' + str(i)

            if self.savename:
                pl.figure()
                pl.imshow(exp(x))
                pl.colorbar()
                pl.title('Iteration' + str(i)+'_m')
                pl.savefig("resolve_output_" + str(self.savename) +
                           "/last_iterations/iteration" + str(i) + "_" +
                           self.runlist_element + "_" + self.method)
                np.save("resolve_output_" + str(self.savename) +
                        "/last_iterations/iteration" + str(i) + "_" +
                        self.runlist_element + "_" + self.method, exp(x))
                pl.close()


def save_u(x,git,params):
    save_results(exp(x.val), "map, iter #" + str(git), \
        'resolve_output_' + params.save +\
        '/u_reconstructions/' + params.save + "_expu"+ str(git), \
        rho0 = params.rho0)
    pars.write_output_to_fits(np.transpose(exp(x.val)*params.rho0),params,\
        notifier = str(git), mode='I_u')    
            
def save_m(x,git,params,w_git=0):
        
    if params.freq == 'wideband':
        save_results(exp(x.val), "map, iter #" + str(git), \
            'resolve_output_' + str(params.save) +\
            '/m_reconstructions/' + params.save + "_expm" +\
            str(wideband_git) + "_" + str(git), rho0 = params.rho0)
        pars.write_output_to_fits(np.transpose(exp(m.val)*params.rho0),params, \
            notifier = str(w_git) + "_" + str(git),mode = 'I')
               
    else:            
        save_results(exp(x.val), "map, iter #" + str(git), \
            'resolve_output_' + params.save +\
            '/m_reconstructions/' + params.save+ "_expm" + str(git), \
        rho0 = params.rho0)
        pars.write_output_to_fits(np.transpose(exp(x.val)*params.rho0),params,\
            notifier = str(git), mode='I')  

def save_mu(m,u,git,params):
    save_results(exp(u.val)+exp(m.val), "map, iter #" + str(git), \
        'resolve_output_' + str(params.save) +\
        '/mu_reconstructions/' + params.save + "_expmu"+ str(git), \
        rho0 = params.rho0)
    pars.write_output_to_fits(np.transpose((exp(u.val) + exp(m.val))
                              * params.rho0), params, notifier=str(git),
                              mode='I_mu')


def plot_figure_with_axes(image, title, fname, params, cmap='Greys',
                          clim=None, objects=True, wcscoords=True, rho0=1.):
    
    # astropy and wcs imports are really only needed here and are kept in this
    # functions so that Resolve as a whole does not depend on astropy.
    try:
        import matplotlib.pyplot as plot
        from astropy.io import fits
        from astropy.wcs import WCS
        from astropy import wcs

    except ImportError:
        raise ImportError('No astropy and wcs support available. No plotting'
                          + ' of python output with proper astronomical axes.')

    plot.ioff()
    plot.close('all')
    image *= rho0

    if not clim:
        clim = (image.min(), image.max())

    fig = plot.figure()
    fig.suptitle(r'${}$'.format('\mathrm{' + title + '}'))

    if wcscoords:

        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [params.imsize/2, params.imsize/2]
        w.wcs.cdelt = np.array([params.cellsize * 57.2958, params.cellsize
                               * 57.2958])
        w.wcs.crval = [params.summary['field_0']['direction']
                      ['m0']['value'] * 57.2958, params.summary['field_0']
                      ['direction']['m1']['value'] * 57.2958]
        w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
        if (params.summary['field_0']['direction']['refer'] == '1950' or 
            params.summary['field_0']['direction']['refer'] == '1950_VLA'or 
            params.summary['field_0']['direction']['refer'] == 'B1950' or 
            params.summary['field_0']['direction']['refer'] == 'B1950_VLA'):
        
            w.wcs.radesys = 'fk4'
            w.wcs.equinox = 1950
    
        elif (params.summary['field_0']['direction']['refer'] == '2000' or 
              params.summary['field_0']['direction']['refer'] == 'J2000'):
                  
            w.wcs.radesys = 'fk5'
            w.wcs.equinox = 2000              
    
        ax = fig.add_subplot(111, projection=w)
        ra = ax.coords['ra']
        dec = ax.coords['dec']
        ra.set_axislabel(r'${}$'.format('\mathrm{RA ' + 
                         params.summary['field_0']['direction']
                         ['refer'] + '}', size=15))
        dec.set_axislabel(r'${}$'.format('\mathrm{DEC ' + 
                         params.summary['field_0']['direction']
                         ['refer'] + '}', size=15, minpad=-0.5))
        ra.set_major_formatter('hh:mm:ss')
        
        plot.imshow(image,cmap=cmap, vmin=clim[0],vmax=clim[1])
        cbar = pl.colorbar(ticks=[clim[0],clim[1]])
        cbar.set_label(r'$\frac{\mathrm{Jy}}{\mathrm{px}}$',labelpad=-25, \
            size=22,rotation=360,verticalalignment='center')
        #labelpad=-25
            
    
    else:        

        plot.imshow(image,cmap=cmap, vmin=clim[0],vmax=clim[1])
        cbar = pl.colorbar(ticks=[clim[0],clim[1]])
        cbar.set_label(r'$\frac{\mathrm{Jy}}{\mathrm{px}}$', \
            size=22,rotation=360,verticalalignment='center')        
        
    
    plot.savefig(fname)
        

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
        summary = np.load(msfn + '_sum.npy').item()
        nvis = np.load(msfn + '_nvis.npy')
        flags =  np.load(msfn + '_flags.npy')

    except IOError:
        logger.failure('A needed numpy file does not exists in the working'\
            + ' directory with the prefix ' + msfn)
        raise IOError

    return vis, sigma, u, v, flags, freqs, nchan, nspw, nvis, summary
    
def read_data_from_ms_in_python(msfn, m, viscol="DATA", noisecol='SIGMA',
                                mode='tot'):
    """
    Reads polarization or total intensity data into a visibility and a 
    noise data array.
    
    Args:
        msfn: Name of the MeasurementSet file from which to read the data
        viscol: A string with the name of the MS column from which to read the 
            data [DATASET_STRING]
        noisecol: A string with the name of the MS column from which to read
            the noise or weights ['SIGMA']
        mode: Flag to set whether the function should read in
            polarization data ('pol') or total intensity data ('tot')
    
    Returns:
        Vis, Noise, u, v, as an array, nested in two lists of length
        nspw, nchan.
    """

    #conditionally import python-casacore  module; might not be available
    #due to complicated casacore dependencies
    try:
        from casacore import tables as pt
    except ImportError:
        raise ImportError('Cannot find python-casacore module for direct '
                          + 'read out of ms in python.')
    
    
    C = 299792458
    
    if mode == 'pol':
        m.header2("Reading polarization data from the MeasurementSet...")
        m.warn("Polarization read out has not been maintained well, likely to"
               + " encounter errors.")
    if mode == 'tot':
        m.header2("Reading total intensity data from the MeasurementSet...")    
    
    # number of rows to read in total. if zero, reads them all
    nrows = 0
    
    if pt.tableexists(msfn) == False:
        raise IOError('No Measurement Set with the given name')
    
    mt = pt.table(msfn) # main table, in read only mode
    swt = pt.table(msfn+'/SPECTRAL_WINDOW/') # spectral window table
    # table row interface, we only need channel frequencies and the widths
    # from this table.
    swr = swt.row(['CHAN_FREQ', 'CHAN_WIDTH'])
    
    # polarization table, only used to get the corr_type
    polt = pt.table(msfn+'/POLARIZATION/') 
    corr = polt.getcol('CORR_TYPE') # correlation type (XX-YY, RR-LL, etc.)
    
    if len(corr)>1:
        raise ValueError('Unexpected number of polarization configurations')
    
    # the Q,U OR the I part of the S Jones matrix (hence Spart)
    # from the Stokes enumeration defined in the casa core libraries
    # http://www.astron.nl/casacore/trunk/casacore/doc/html \
        #/classcasa_1_1Stokes.html#e3cb0ef26262eb3fdfbef8273c455e0c
    # this defines which polarization type the data columns correspond to

    corr_announce = "Correlation type detected to be "        
    
    if mode == 'pol':
        if corr[0,0] == 5: # RR, RL, LR, LL
            Spart = np.array([[0, 0.5, 0.5, 0], [0, -0.5j, 0.5j, 0]])
            corr_announce += "RR, RL, LR, LL"
        elif corr[0,0] == 1: # I, Q, U, V
            Spart = np.array([[0, 1., 0, 0], [0, 0, 1., 0]])
            corr_announce += "I, Q, U, V"
        elif corr[0,0] == 9: # XX, XY, YX, YY    
            Spart = np.array([[0.5, 0, 0, -0.5], [0, 0.5, 0.5, 0]])
            corr_announce += "XX, XY, YX, YY"
    if mode == 'tot':
        if corr[0,0] == 5: # RR, RL, LR, LL
            Spart = np.array([0.5, 0, 0, 0.5])
            corr_announce += "RR, RL, LR, LL"
        elif corr[0,0] == 1: # I, Q, U, V
            Spart = np.array([1., 0, 0, 0])
            corr_announce += "I, Q, U, V"
        elif corr[0,0] == 9: # XX, XY, YX, YY    
            Spart = np.array([0.5, 0, 0, 0.5])
            corr_announce += "XX, XY, YX, YY"           
    
    m.message(corr_announce, 2)        
    
    # total number of rows to read. Each row has nchan records, so there are
    # a total of nrows*nchan records
    if nrows == 0:
        nrows = mt.nrows()
    
    nspw = swt.nrows() # Number of spectral windows (aka subbands)
    
    
    # retrieve the list of l2 or frequency values that are sampled
    freqs = list()
    if mode == 'pol':
    
        for i in range(nspw):
            chan_freqs = swr.get(i).get('CHAN_FREQ')
            chan_widths = swr.get(i).get('CHAN_WIDTH')

            nchan = len(chan_freqs)
            l2_vec = np.zeros(nchan)
            
            for j in range(nchan):
                templ2 = 0.5*C2*((chan_freqs[j]-chan_widths[j]*0.5)**-2 \
                    + (chan_freqs[j]+chan_widths[j]*0.5)**-2)
                l2_vec[j] = templ2/PI
            freqs.append(l2_vec)
        
#           vis.coords.freqs = freqs
    if mode == 'tot':
        
        lambs = list()
        for i in range(nspw):
            chan_freqs = swr.get(i).get('CHAN_FREQ')
            chan_widths = swr.get(i).get('CHAN_WIDTH')

            nchan = len(chan_freqs)
            freq_vec = np.zeros(nchan)
            
            for j in range(nchan):
                tempfreq = (chan_freqs[j]-chan_widths[j]*0.5) \
                    + (chan_freqs[j]+chan_widths[j]*0.5)
                freq_vec[j] = tempfreq
            freqs.append(freq_vec)
            lambs.append(C/freq_vec)


    nfreqs = 0. # Total number of frequencies
    for i in range(len(freqs)):
        nfreqs += len(freqs[i])
        
    nstokes = 4 # Number of Stokes parameters. Must be 4
    
    m.message("MeasurementSet information", 1)
    m.message("Number of rows: " + str(nrows),1)
    m.message("Number of spectral windows: " + str(nspw),1)
    m.message("Number of frequencies: " + str(nfreqs),1)
    

    if mode == 'pol':
        Qvis = [[]]
        Uvis = [[]]
        
    if mode == 'tot':
        vis = [[]]
        
        
    noise = []        
    u = [[]]
    v = [[]]
    Flags = [[]]
    
    for i in range(nspw):
        m.message("Reading spectral window number " + str(i) + "...", 2)
        
        stab = pt.taql("SELECT FROM $mt WHERE DATA_DESC_ID == $i")
        
        nrecs_stab = stab.nrows()
        
        uvw = stab.getcol('UVW') # u,v,w coords (in meters)
        nchan = len(freqs[i])
        
        # one noise per SPW (also per cross-corr), applies to all channels
        noise_recs = stab.getcol(noisecol.upper())
#            noise_recs = noise_recs.reshape(nrecs_stab, nstokes)

        if mode == 'pol':
            Qvis.append([])
            Uvis.append([])
        if mode == 'tot':
            vis.append([])
        u.append([])
        v.append([])
        Flags.append([])
        
        for j in range(nchan):
            m.message(".   Reading channel " + str(freqs[i][j]) \
                + " which has " + str(nrecs_stab) + " records ...", 3)
            
            if mode == 'pol':
                Qvis_array = np.zeros(nrecs_stab, dtype=np.complex)
                Uvis_array = np.zeros(nrecs_stab, dtype=np.complex)
            if mode == 'tot':
                vis_array = np.zeros(nrecs_stab, dtype=np.complex)
            noise_array = np.zeros(nrecs_stab, dtype=np.float)
            u_array = np.zeros(nrecs_stab)
            v_array = np.zeros(nrecs_stab)
            
            data_recs = stab.getcolslice(viscol.upper(), [j,0], [j,3])
#                noise_recs = stab.getcolslice(noisecol.upper(), [j], [j])
            flag_recs = stab.getcolslice('FLAG', [j,0], [j,3])
            
            data_recs = data_recs.reshape(nrecs_stab,nstokes)
            flag_recs = flag_recs.reshape(nrecs_stab,nstokes)
            
            flags = 1. - flag_recs

            if mode == 'tot':
                if len(flags) > 2:
                    if not(np.sum(flags[0])==np.sum(flags[3])):
                       m.warn('Warning: Different flags for ' \
                       +'different correlations/channels. '\
                       +'Hard flag is applied: If any necessary '\
                       +'correlation is flagged, this gets '\
                       +'extended to all.')
                       flag = np.ones(np.shape(flags[0]))
                       flag[flags[0]==0.] == 0.
                       flag[flags[3]==0.] == 0.
                         
                    else:
                        flag = flags[0]
                else:
                    flag = flags[0]    
                    
            elif mode == 'pol':
                if not(np.sum(flags[1])==np .sum(flags[2])):
                   m.warn('Warning: Different flags for ' \
                   +'different correlations/channels. '\
                   +'Hard flag is applied: If any necessary '\
                   +'correlation is flagged, this gets '\
                   +'extended to all.')
                   flag = np.ones(np.shape(flags[1]))
                   flag[flags[1]==0.] == 0.
                   flag[flags[2]==0.] == 0.
                else:
                    flag = flags[1]

            #read in data
            if mode == 'pol':
                for k in range(nrecs_stab):
                    Qvis_array[k] = np.dot(Spart[0], data_recs[k])
                    Uvis_array[k] = np.dot(Spart[1], data_recs[k])
                    qnoise = np.dot(Spart[0], noise_recs[k])
                    unoise = np.dot(Spart[1], noise_recs[k])
                    noise_array[k] = np.sqrt(qnoise**2 + unoise**2)
                
                    u_array[k] = uvw[k,0]/np.sqrt(PI*freqs[i][j])
                    v_array[k] = uvw[k,1]/np.sqrt(PI*freqs[i][j])
                    
                Qvis[i].append(Qvis_array)
                Uvis[i].append(Uvis_array)
                Flags[i].append(flag)
                                        
                    
            if mode == 'tot':
                for k in range(nrecs_stab):
                    vis_array[k] = np.dot(Spart, data_recs[k])
                    noise_array[k] = np.dot(Spart, noise_recs[k])
                
                    u_array[k] = uvw[k,0]/lambs[i][j]
                    v_array[k] = uvw[k,1]/lambs[i][j]
            
                vis[i].append(vis_array)
                Flags[i].append(flag)
                
            u[i].append(u_array)
            v[i].append(v_array)
            noise.append(noise_array)
            
        stab.close()
        m.message("Done!", 2)
    
    mt.close()
    swt.close()
    polt.close()
    
    m.success("Finished reading data from the MeasurementSet!",0)

    if mode == 'pol':
        
        return Qvis, Uvis, noise, u, v, freqs
    
    if mode == 'tot':
        
        return vis, noise, u, v, Flags, freqs, nchan, nspw, len(vis[0][0]),\
            False
    
def update_globvars(gsavein, gcallbackin):

    global gsave
    global gcallback
    gsave = gsavein
    gcallback = gcallbackin

def str2bool(v):
  return v == 'True'
    
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
            return field(x.domain,val=utils.log(np.maximum(1E-323,x.val))/np.log(base).astype(x.domain.datatype),target=x.target)
            #return field(x.domain,val=np.log(x.val)/np.log(base).astype(x.domain.datatype),target=x.target)
        else:
#            if(np.any(x<1E-323)):
#                print("** LOGSTROKE **")
            return np.log(np.array(np.maximum(1E-323,x)))/np.log(base)
    else:
        raise ValueError(about._errors.cstring("ERROR: invalid input basis."))
        

    
    
    
               
