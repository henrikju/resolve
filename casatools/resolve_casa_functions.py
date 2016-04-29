"""
resolve_casa_functions.py
Written by Henrik Junklewitz

resolve_casa_functions.py is an auxiliary file for resolve.py and belongs to 
the RESOLVE package. It provides all the functions that need explicit CASA 
input and currently cannot be used from a python interpreter.

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

#VERY unelegnat quick fix
import sys
sys.path.append('../')


import numpy as np
import casa_IO as ca
import scipy.ndimage.interpolation as sci
from casa import image as ia
import casa

import utility_functions as utils



#------------------------CASA-MS-I/O wrapper-----------------------------------

def read_data_withCASA(ms, viscol="DATA", noisecol='SIGMA', \
    mode='tot', noise_est = False, save=None):
        
    if mode == 'tot':
        
        vis, sigma, u, v, allflags, freq, nchan, nspw, nvis, summary = \
            ca.read_data_from_ms(ms, viscol=viscol, noisecol=noisecol, \
            mode=mode, noise_est = noise_est)
        
        if save:
            np.save('../'+save+'_vis',vis)
            np.save('../'+save+'_sigma',sigma)
            np.save('../'+save+'_u',u)
            np.save('../'+save+'_v',v)
            np.save('../'+save+'_freq',freq)
            np.save('../'+save+'_nchan',nchan)
            np.save('../'+save+'_nspw',nspw)
            np.save('../'+save+'_sum',summary)
            np.save('../'+save+'_nvis',nvis)
            np.save('../'+save+'_flags',allflags)
        
        return vis, sigma, u, v, freq, nchan, nspw, nvis, summary
            
    elif mode == 'pol':
        
        Qvis, Qsigma, Uvis, Usigma, allflags, freq, lamb, u, v, nchan, nspw, summary = \
        ca.read_data_from_ms(ms, viscol=viscol, noisecol=noisecol, \
            mode=mode, noise_est = noise_est)
        
        if save:
            np.save(save+'_Qvis',Qvis)
            np.save(save+'_Uvis',Uvis)
            np.save(save+'_Qsigma',Qsigma)
            np.save(save+'_Usigma',Usigma)
            np.save(save+'_u',u)
            np.save(save+'_v',v)
            np.save(save+'_freq',freq)
            np.save(save+'_nchan',nchan)
            np.save(save+'_nspw',nspw)
            np.save(save+'_sum',summary)
            np.save(save+'_flags',allflags)
        
        return Qvis, Uvis, Qsigma, Usigma, u, v, freq, nchan, nspw, nvis, \
            summary

#------------------------single utility functions------------------------------


def make_dirtyimage(ms, cellsize, imsize, save):

    casa.clean(vis = ms,imagename = 'di',cell = \
        str(cellsize) + 'rad', imsize = imsize, \
        niter = 0, mode='mfs')

    ia.open('di.image')
    imageinfo = ia.summary('di.image')
    
    norm = imageinfo['incr'][1]/np.pi*180*3600  #cellsize converted to arcsec
    
    beamdict = ia.restoringbeam()
    major = beamdict['major']['value'] / norm
    minor = beamdict['minor']['value'] / norm
    np.save('beamarea',1.13 * major * minor)
    
    di = ia.getchunk().reshape(imageinfo['shape'][0],imageinfo['shape'][1])\
        / (1.13 * major * minor)
    
    ia.close()
    
    call(["rm", "-rf", "di.image"])
    call(["rm", "-rf", "di.model"])
    call(["rm", "-rf", "di.psf"])
    call(["rm", "-rf", "di.flux"])
    call(["rm", "-rf", "di.residual"])
    
    np.save('../resolve_output_'+str(save)+'/general/di.npy',\
        utils.convert_CASA_to_RES(di))


def read_pbeam_from_CASA(pbeamimagename, outn):
    
    ia.open(pbeamimagename)
    imageinfo = ia.summary(pbeamimagename)
    pbeam = ia.getchunk().reshape(imageinfo['shape'][0],imageinfo['shape'][1])
    
    ia.close()
    
    np.save(outn,pbeam)

def read_image_from_CASA(casaimagename,zoomfactor, outn):

    ia.open(casaimagename)
    imageinfo = ia.summary(casaimagename)

    norm = imageinfo['incr'][1]/np.pi*180*3600  #cellsize converted to arcsec

    beamdict = ia.restoringbeam()
    major = beamdict['major']['value'] / norm
    minor = beamdict['minor']['value'] / norm

    image = ia.getchunk().reshape(imageinfo['shape'][0],imageinfo['shape'][1])\
        / (1.13 * major * minor)

    ia.close()
    
    image = sci.zoom(image,zoom=zoomfactor)
    np.save(outn,utils.convert_CASA_to_RES(image))

def read_model_from_CASA(casaimagename,zoomfactor, outn):

    ia.open(casaimagename)
    imageinfo = ia.summary(casaimagename)

    image = ia.getchunk().reshape(imageinfo['shape'][0],imageinfo['shape'][1])\
    

    ia.close()
    
    image = sci.zoom(image,zoom=zoomfactor)
    np.save(outn,utils.convert_CASA_to_RES(image))    
        