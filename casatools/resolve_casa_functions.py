"""
resolve_casa_functions.py
Written by Henrik Junklewitz

resolve_casa_functions.py is an auxiliary file for resolve.py and belongs to 
the RESOLVE package. It provides all the functions that need explicit CASA 
input.

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
import casa_IO as ca


#------------------------CASA-MS-I/O wrapper-----------------------------------

def read_data_withCASA(ms,save=None):
    
    vis, sigma, u, v, freq, nchan, nspw, nvis, summary = \
        ca.read_data_from_ms(msfn, viscol="DATA", noisecol='SIGMA', mode='tot',\
        noise_est = False)

    if save:
        np.save(save+'_vis',vis)
        np.save(save+'_sigma',sigma)
        np.save(save+'_u',u)
        np.save(save+'_v',v)
        np.save(save+'_freq',freq)
        np.save(save+'_nchan',nchan)
        np.save(save+'_nspw',nspw)
        np.save(save+'_sum',summary)
        
    return vis, sigma, u, v, freq, nchan, nspw, nvis, summary

#------------------------single utility functions------------------------------


def make_dirtyimage(params, logger):

    casa.clean(vis = params.ms,imagename = 'di',cell = \
        str(params.cellsize) + 'rad', imsize = params.imsize, \
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

    
    return convert_CASA_to_RES(di)

def read_image_from_CASA(casaimagename,zoomfactor):

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
    
    return convert_CASA_to_RES(image)