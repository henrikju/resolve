# -*- coding: utf-8 -*-

"""
parser.py
Written by Quirin Kronseder and Henrik Junklewitz.

Resolve.py defines the main function that runs RESOLVE on a measurement 
set with radio interferometric data.

Copyright 2016 Henrik Junklewitz and Quirin Kronseder

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
import sys
import csv
import numpy as np
import datetime
import pyfits

import utility_functions as utils


#---------------- Parsing runfile and parameters ------------------------------

def get_parameter_and_runscript_from_runfile(runfile):
    
    actions = []     
    actions_buffer = []     
    parameter_file = []                   
    parameter_file_buffer = []
    parameter_changes = []    
    parameters = dict()

    """ parse parameter file."""
    path = './configuration'


# get Runfile 
    try:
        reader = csv.reader(open(path+'/runfiles/'+runfile, 'rb'),\
            delimiter=" ", skipinitialspace=True)
      
        for row in reader:
            if len(row) >= 3:
                for i in range(2,len(row)):
                    row.remove(row[i])
            if len(row) == 1:
                actions_buffer.append(row[0])
                parameter_file_buffer.append('')
                
            if len(row) == 2:
                actions_buffer.append(row[0])
                parameter_file_buffer.append(row[1])
        
        
        
        i = 0
        stop = 0
        go = 0
        start = 0
        times = 0

        for i in range(len(actions_buffer)):
            if str.isdigit(actions_buffer[i]):
                start = min(i+1,len(actions_buffer)-1)#i + 1
                for go in range(start,len(actions_buffer)):
                    if str.isdigit(actions_buffer[go]): 
                        stop = go
                        break
                    elif go == len(actions_buffer)-1:
                        stop = go+1
                        break
                
                for times in range(0,int(actions_buffer[start-1])):
                    if start == stop:
                        actions.append(actions_buffer[start])
                        parameter_file.append(parameter_file_buffer[start])
                        
                    else:
                        for go in range(start,stop):
                            actions.append(actions_buffer[go])
                            parameter_file.append(parameter_file_buffer[go])
                start = 0
                stop = 0
    
        
        # read parameter changes
        for t in range(len(parameter_file)):
            if parameter_file[t] != '':
                reader = csv.reader(open(path+'/parameters/'\
                    + parameter_file[t], 'rb'), delimiter=" ",\
                    skipinitialspace=True)
                for row in reader:
                    if len(row) >= 4:
                        if row[1] == 'freq':
                                parameters[row[1]] = \
                                    [int(row[2]),int(row[3])]
                        else:
                            for ii in range(3,len(row)):
                                row.remove(row[ii])
                    elif len(row) == 3:
                        if row[0].lower() == 'boolean':
                            parameters[row[1]] = utils.str2bool(row[2])
                        elif row[0].lower() == 'float':
                            parameters[row[1]] = float(row[2])
                        elif row[0].lower() == 'integer':
                            parameters[row[1]] = int(row[2])
                        elif row[0].lower() == 'string':
                            if row[2] == 'False':
                               parameters[row[1]] = ''
                            else:
                               parameters[row[1]] = str(row[2])
                
                
                parameter_changes.append(parameters.copy())
                
            else:
                parameter_changes.append('')
        print '#################################'
        
        return actions, parameter_changes
    
    except IOError, e:
        print e.errno
        print e





def load_default_parameters ():

    param = dict()
    try:
        reader = csv.reader(open('./configuration/parameters/'\
            + 'default_parameters.cfg', 'rb'), delimiter=" ",\
                skipinitialspace=True)
      
        for row in reader:
            if len(row) >= 4:
                if row[1] == 'freq':
                    param[row[1]] = [int(row[2]),int(row[3])]
                else:
                    for i in range(3,len(row)):
                        row.remove(row[i])
            elif len(row) <= 2 and len(row) > 0:
               print 'Default parameter file corupted!'
               print './configuration/parameters/default_parameters.cfg'
#               print row
               break
            elif len(row) == 3:
                 if row[0].lower() == 'boolean':
                    param[row[1]] = utils.str2bool(row[2])
                 elif row[0].lower() == 'float':
                    param[row[1]] = float(row[2])
                 elif row[0].lower() == 'integer':
                    param[row[1]] = int(row[2])
                 elif row[0].lower() == 'string':
                    if row[2] == 'False':
                        param[row[1]] = ''
                    else:
                        param[row[1]] = str(row[2])

            

        return param   
    except IOError, e:
        print e.errno
        print e
        
#------------------------ Internal parameter class ----------------------------

def update_parameterclass(params, changep):
    
    for key in changep:
        if key in dir(params):
            setattr(params,key,changep[key])
        else:
            raise IOError('Wrong parameter definition from config files')
    return params
            

class parameters(object):
    """
    Defines a parameter class for all parameters that are vital to controlling
    basic functionality of the code. Some are even mandatory and given by 
    the user. Performs checks on default arguments with hard-coded default
    values.
    """

    def __init__(self, params_dict, datafn, imsize, cellsize, save, 
                 python_casacore, verbosity):

        # mandatory parameters
        self.datafn = datafn
        self.imsize = int(imsize)
        self.cellsize = float(cellsize)
        self.save = save
        self.python_casacore = python_casacore
        self.verbosity = verbosity

        # read dict into parameter class
        for key in params_dict:
            setattr(self,key,params_dict[key])


#----------------------- Code I/O ---------------------------------------------
        

def write_output_to_fits(m, params, notifier='',mode='I',u=None):
    """
    """

    hdu_main = pyfits.PrimaryHDU(utils.convert_RES_to_CASA(m,FITS=True))
    
    try:
        generate_fitsheader(hdu_main, params)
    except:
        print "Warning: There was a problem generating the FITS header, no " + \
            "header information stored!"
        print "Unexpected error:", sys.exc_info()[0]
    hdu_list = pyfits.HDUList([hdu_main])
    
        
    if mode == 'I':
        hdu_list.writeto('resolve_output_' + str(params.save) +\
                '/' + str(params.save) + '_' + 'expm' + str(notifier) + '.fits', clobber=True)
    elif mode == 'I_u':
        hdu_list.writeto('resolve_output_' + str(params.save) +\
                '/' + str(params.save) + '_' + 'expu' + str(notifier) + '.fits', clobber=True)  
    elif mode == 'I_mu':
        hdu_list.writeto('resolve_output_' + str(params.save) +\
                '/' + str(params.save) + '_' + 'expmu' + str(notifier) + '.fits', clobber=True)                 
    else:
        hdu_list.writeto('resolve_output_' + str(params.save) +\
                '/' + str(params.save) + '_' + 'a' + str(notifier) + '.fits', clobber=True)


def generate_fitsheader(hdu, params):
    """
    """
    
    today = datetime.datetime.today()
    
    #hdu.header = inhead.copy()
    
    hdu.header.set('ORIGIN', 'resolve.py', 'Origin of the data set')
    
    hdu.header.set('DATE', str(today), 'Date when the file was created')
    
  #  hdu.header.set('OBJECT', params.summary['name'], 'Name of object')
    
    hdu.header.set('NAXIS', 3,
                      'Number of axes in the data array, must be 2')

    hdu.header.set('BUNIT','JY/PIXEL')
    
    hdu.header.set('BTYPE','Intensity')
                      
    hdu.header.set('NAXIS1', params.imsize,
                      'Length of the RA axis')
    
    hdu.header.set('CTYPE1', 'RA---SIN', 'Axis type')
    
    hdu.header.set('CUNIT1', 'deg', 'Axis units')
    
    # In FITS, the first pixel is 1, not 0!!!
    hdu.header.set('CRPIX1', params.imsize/2, 'Reference pixel')
    
    hdu.header.set('CRVAL1', params.summary['field_0']['direction']\
    ['m0']['value'] * 57.2958, 'Reference value')
    
    hdu.header.set('CDELT1', -1 * params.cellsize * 57.2958, 'Size of pixel bin')
    
    hdu.header.set('NAXIS2', params.imsize,
                      'Length of the RA axis')
    
    hdu.header.set('CTYPE2', 'DEC--SIN', 'Axis type')
    
    hdu.header.set('CUNIT2', 'deg', 'Axis units')
    
    # In FITS, the first pixel is 1, not 0!!!
    hdu.header.set('CRPIX2', params.imsize/2, 'Reference pixel')
    
    hdu.header.set('CRVAL2', params.summary['field_0']['direction']\
    ['m1']['value'] * 57.2958, 'Reference value')
    
    hdu.header.set('CDELT2', params.cellsize * 57.2958, 'Size of pixel bin')
    
    if (params.summary['field_0']['direction']['refer'] == '1950' or \
        params.summary['field_0']['direction']['refer'] == '1950_VLA'or \
        params.summary['field_0']['direction']['refer'] == 'B1950' or \
        params.summary['field_0']['direction']['refer'] == 'B1950_VLA'):
            
        hdu.header.set('EQUINOX', 1950)
                      
        hdhdu.header.set('RADESYS', 'FK4')
    
    elif (params.summary['field_0']['direction']['refer'] == '2000' or \
        params.summary['field_0']['direction']['refer'] == 'J2000'):

        hdu.header.set('EQUINOX', 2000)
                      
        hdhdu.header.set('RADESYS', 'FK5')
    
    if params.freq == 'wideband':
        raise NotImplementedError('No widefield FITS-cube output yet.')
    else:
        hdu.header.set('NAXIS3', 1, 'Length of frequency axis')
        
        hdu.header.set('CTYPE3', 'FREQ', 'Axis type')
        
        hdu.header.set('CRVAL3', params.freqs[params.freq[0],params.freq[1]]\
            , 'Frequency Reference value')
            
        
    hdu.header.add_history('RESOLVE: Reconstruction performed by ' +
                               'resolve.py.')

    
    hdu.header.__delitem__('NAXIS4')
    hdu.header.__delitem__('CTYPE4')
    hdu.header.__delitem__('CRVAL4')
    hdu.header.__delitem__('CRPIX4')
    hdu.header.__delitem__('CDELT4')
    hdu.header.__delitem__('CUNIT4')
