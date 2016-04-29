"""
general_IO.py
Written by Henrik Junklewitz

General_IO.py is an optional part of the RESOLVE package and provides CASA 
specific reading routines for measuremnet sets. All functions can only be run
in a CASA environment.


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

#VERY unelegnat quick fix
import sys
sys.path.append('../')

from casa import ms
import numpy as np
import Messenger as M

C = 299792458
PI = 3.14159265358979323846

#General purpose routine for MS-IO using CASA
def read_data_from_ms(msfn, viscol="DATA", noisecol='SIGMA',
                      mode='tot',noise_est = False,cross_pol=True):
    """
    Reads polarization or total intensity data in visibility and noise arrays.

    Args:
        msfn: Name of the MeasurementSet file from which to read the data
        viscol: A string with the name of the MS column from which to read the
            data [DATASET_STRING]
        noisecol: A string with the name of the MS column from which to read
            the noise or weights ['SIGMA']
        mode: Flag to set whether the function should read in
            polarization data ('pol') or total intensity data ('tot')

    Returns:
        vis
	noise
    """

    m = M.Messenger(2)

    if mode == 'pol':
        m.header2("Reading polarization data from the MeasurementSet...")
    if mode == 'tot':
        m.header2("Reading total intensity data from the MeasurementSet...")
    
    viscol = viscol.lower()
    noisecol = noisecol.lower()

    if noise_est == 'old':
        m.message("Performing simple noise estimate")
        computenoise_martin(msfn,viscol,m,minsamp=10)

    ms.open(msfn)
    meta = ms.metadata()
    nspw = range(meta.nspw())
    nchan = []
    nvis = []
    u = []
    v = []
    freq = []
    allflags = []
    if mode == 'pol':
  
      Qvis = []
      Uvis = []
      Qsigma = []
      Usigma = []
      lambs2 = []
     
    if mode == 'tot':
  
      vis = []
      sigma = []
  
    # the Q,U OR the I part of the S Jones matrix (hence Spart)
    # from the Stokes enumeration defined in the casa core libraries
    # http://www.astron.nl/casacore/trunk/casacore/doc/html \
        #/classcasa_1_1Stokes.html#e3cb0ef26262eb3fdfbef8273c455e0c
    # this defines which polarization type the data columns correspond to

    corr = ms.getdata(['axis_info'])['axis_info']['corr_axis']

    corr_announce = "Correlation type detected to be "

    ii = complex(0, 1)

    if mode == 'pol':
        if corr[0] == 'RR':  # RR, RL, LR, LL
            QSpart = np.array([0, 0.5, 0.5, 0])
            USpart = np.array([0, -0.5 * ii, 0.5 * ii, 0])
            corr_announce += "RR, RL, LR, LL"
        elif corr[0] == 'I':  # I, Q, U, V
            QSpart = np.array([0, 1., 0, 0]) 
            USpart = np.array([0, 0, 1., 0])
            corr_announce += "I, Q, U, V"
        elif corr[0] == 'XX':  # XX, XY, YX, YY
            QSpart = np.array([0.5, 0, 0, -0.5])
            USpart = np.array([0, 0.5, 0.5, 0])
            corr_announce += "XX, XY, YX, YY"
    if mode == 'tot':
        if corr[0] == 'RR':  # RR, RL, LR, LL
            Spart = np.array([0.5, 0, 0, 0.5])
            corr_announce += "RR, RL, LR, LL"
        elif corr[0] == 'I':  # I, Q, U, V
            Spart = np.array([1., 0, 0, 0])
            corr_announce += "I, Q, U, V"
        elif corr[0] == 'XX':  # XX, XY, YX, YY
            Spart = np.array([0.5, 0, 0, 0.5])
            corr_announce += "XX, XY, YX, YY"

    print corr_announce

    if noise_est != 'full':
        for spw in nspw:
            nchan.append(meta.nchan(spw))
      
            ms.selectinit(datadescid=spw)
            temp = ms.getdata([viscol,"axis_info"],ifraxis=False)
            data_temp = temp[viscol]
            info_temp = temp["axis_info"]
            if noisecol == 'weight_spectrum':
                raise NotImplementedError('Weight_spectrum loadnot implemented yet.')
            else:
	        s_temp = ms.getdata(['sigma'],\
                    ifraxis=False)['sigma']
            flags = 1. - ms.getdata(["flag"])['flag']

            if mode == 'tot':
                if len(flags) > 2:
                    if not(np.sum(flags[0])==np.sum(flags[3])):
                       m.warn('Warning: Different flags for ' \
                       +'different correlations/channels. '\
                       +'Hard flag is applied: If any necessary'\
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
                   +'Hard flag is applied: If any necessary'\
                   +'correlation is flagged, this gets '\
                   +'extended to all.')
                   flag = np.ones(np.shape(flags[1]))
                   flag[flags[1]==0.] == 0.
                   flag[flags[2]==0.] == 0.
                else:
                    flag = flags[1]

            #Start reading data.

            if mode == 'tot': 
                if cross_pol:
                    vis_temp = (Spart[0] * data_temp[0]\
                        + Spart[1] * data_temp[1] + Spart[2] *\
                        data_temp[2] + Spart[3] * data_temp[3])
                    sigma_temp = (Spart[0] * s_temp[0]\
                        + Spart[1] * s_temp[1] + Spart[2] *\
                        + s_temp[2] + Spart[3] * s_temp[3])
                else:
                    vis_temp = (Spart[0] * data_temp[0]\
                        + Spart[1] * data_temp[1])
                    sigma_temp = (Spart[0] * s_temp[0]\
                        + Spart[1] * s_temp[1])
                        
                vis.append(vis_temp)
                sigma.append(sigma_temp)

                nvis.append(len(vis_temp[0]))


            if mode == 'pol':
                Qvis_temp = (QSpart[0] *\
                    data_temp[0] + QSpart[1] *\
		         data_temp[1]+QSpart[2] *data_temp[2]\
		         + QSpart[3] * data_temp[3])
                Qsigma_temp = (QSpart[0] *\
                    s_temp[0] + QSpart[1] * s_temp[1]+\
                    QSpart[2] * s_temp[2] +\
                    QSpart[3] * s_temp[3])

                Qvis.append(Qvis_temp)
                Qsigma.append(Qsigma_temp)
     
                Uvis_temp = (USpart[0] * data_temp[0] + USpart[1]\
                    * data_temp[1] + USpart[2] *\
                    data_temp[2] + USpart[3] * data_temp[3])
                Usigma_temp = (USpart[0] * s_temp[0] + USpart[1] * \
                    s_temp[1] + USpart[2] * s_temp[2] +\
                    USpart[3] * s_temp[3])

                Uvis.append(Uvis_temp)
                Usigma.append(Usigma_temp)

                nvis.append(len(Uvis_temp[0]))

   
            #uvflat give all uv coordinates for the chosen 
            #spw in m
            uflat = ms.getdata(['u'])['u']
            vflat = ms.getdata(['v'])['v']

            freqtemp = info_temp["freq_axis"]["chan_freq"]
            freq.append(freqtemp)
            lamb = C / freqtemp
            if mode == 'pol':
                lambs2.append(lamb**2 / PI)

            #calculates uv coordinates per channel in 
            #lambda 
            utemp = np.array([uflat/k for k in lamb])
            vtemp = np.array([vflat/k for k in lamb])
 
            #Reads the uv coordates into lists. Delete 
            #functions take care of flags.
            u.append(utemp)
            v.append(vtemp)
            
            #Reads all flags into general array
            allflags.append(flag)

        try:
            summary = ms.summary()
        except:
            print "Warning: Could not create a summary"
            summary = None        
    
        ms.close()

        if mode =='tot':

            return vis, sigma, u, v, allflags, freq, nchan, nspw,\
                nvis, summary


        if mode == 'pol':
	  
            return Qvis, Qsigma, Uvis, Usigma, allflags, freq, lamb,\
                u, v, nchan, nspw, summary
        
    else:
        
        for spw in nspw:
            nchan.append(meta.nchan(spw))
            if mode == 'tot':
                visspw = []
                sigmaspw = []
            if mode == 'pol':
	        Uvisspw = []
	        Qvisspw = []
	        Usigmaspw = []
	        Qsigmaspw = []
            ants = meta.antennaids()
            for ant1 in ants:
                for ant2 in ants:
                    if ant1 != ant2:
		        ms.selectinit(datadescid=spw)
		        ms.select({'antenna1':[ant1],\
                            'antenna2':[ant2]})
                        temp = ms.getdata([viscol,\
                            "axis_info"], ifraxis=False)
			if not temp:
			    continue
                        data_temp = temp[viscol]
                        info_temp = temp["axis_info"]
                        if noisecol == 'weight_spectrum':
                            raise NotImplementedError(\
                            'Weight_spectrum'\
                            +' load not implemented yet.')
                        else:
	                    s_val = np.std(np.abs(data_temp[:,int(\
                               np.shape(data_temp)[1]/2.)]-\
                               data_temp[:,int(np.shape(data_temp)[1]/2.)+1]))
			    s_temp=np.ones(np.shape(data_temp))*s_val
                        flags = 1. - ms.getdata(["flag"])\
                            ['flag']

                        if not(np.sum(flags[0])==\
                            np.sum(flags[1])==np.sum\
                            (flags[2]) == np.sum\
                            (flags[3])):
                            m.warn('Warning: Different'\
                                +' flags for different'\
	                        +' correlations/channels.'\
                                +' Hard flag is applied:'\
                                +' If any correlation'\
	                        +' is flagged, this gets'\
                                +'extended to all.')
                            maximum =\
                                np.ones(np.shape(flags[0]))
                            for i in range(4):
                                if flags[i].sum() <\
                                    maximum.sum():
                                    maximum = flags[i]
                                flag = maximum
                        else:
                            flag = flags[0]      

                        #Start reading data.

                        if mode == 'tot': 
                            vis_temp = flag * (Spart[0] *\
                                data_temp[0] + Spart[1] *\
		                data_temp[1] + Spart[2] *\
                                data_temp[2] + Spart[3] *\
                                data_temp[3])
                            sigma_temp = flag * (Spart[0]*\
                                s_temp[0] + Spart[1] *\
		                s_temp[1] + Spart[2] *\
                                s_temp[2] + Spart[3] *\
                                s_temp[3])

                            visspw.append(vis_temp)
                            sigmaspw.append(sigma_temp)

            


                        if mode == 'pol':
                            Qvis_temp = flag * (QSpart[0] *\
                                data_temp[0] + QSpart[1] *\
		                data_temp[1]+QSpart[2] *\
		                data_temp[2]+ QSpart[3] *\
		                data_temp[3])
                            Qsigma_temp = flag *\
                                (QSpart[0]*s_temp[0] +\
		                QSpart[1] * s_temp[1] +\
                                QSpart[2] * s_temp[2] +\
                                QSpart[3] * s_temp[3])

                            Qvisspw.append(Qvis_temp)
                            Qsigmaspw.append(Qsigma_temp)
     
                            Uvis_temp = flag * (USpart[0] *\
                                data_temp[0] + USpart[1] *\
		                data_temp[1]+USpart[2] *\
		                data_temp[2]+ USpart[3] *\
		                data_temp[3])
                            Usigma_temp = flag *\
                                (USpart[0]*s_temp[0] +\
		                USpart[1] * s_temp[1] +\
                                USpart[2] * s_temp[2] +\
                                USpart[3] * s_temp[3])

                            Uvisspw.append(Uvis_temp)
                            Usigmaspw.append(Usigma_temp)
            
            if mode == 'tot':
	        nvis.append(len(visspw[0]))
	        visspw = np.swapaxes(visspw,0,1)
                visspw = visspw.reshape((np.shape(visspw)[0],-1))
		sigmaspw = np.swapaxes(sigmaspw,0,1)
                sigmaspw = sigmaspw.reshape((np.shape(sigmaspw)[0],-1))
	        vis.append(visspw)
	        sigma.append(sigmaspw)
	    if mode == 'pol':
	        Qvisspw = np.swapaxes(Qvisspw,0,1)\
                    .reshape((Qvisspw.shape[0],Qvisspw.shape[1],-1))
		Qsigmaspw = np.swapaxes(Qsigmaspw,0,1)\
                    .reshape((Qsigmaspw.shape[0],Qsigmaspw.shape[1],-1))
		Uvisspw = np.swapaxes(Uvisspw,0,1)\
                    .reshape((Uvisspw.shape[0],Uvisspw.shape[1],-1))
		Usigmaspw = np.swapaxes(Usigmaspw,0,1)\
                    .reshape((Usigmaspw.shape[0],Usigmaspw.shape[1],-1))
                nvis.append(len(Uvisspw[0]))
                Qvis.append(Qvisspw)
                Uvis.append(Uvisspw)
                Qsigma.append(Qsigmaspw)
                Usigma.append(Usigmaspw)
   
            ms.selectinit(datadescid=spw)
            
            #uvflat give all uv coordinates for 
            #the chosen spw in m
            uflat = ms.getdata(['u'])['u']
            vflat = ms.getdata(['v'])['v']

            freqtemp =info_temp["freq_axis"]\
	        ["chan_freq"]
            freq.append(freqtemp)
            lamb = C / freqtemp
            if mode == 'pol':
                lambs2.append(lamb**2 / PI)

            #calculates uv coordinates per 
            #channel in lambda 
            utemp = np.array([uflat/k for k in lamb])
            vtemp = np.array([vflat/k for k in lamb])
 
            #Reads the uv coordates into lists. 
            #Delete functions due to flags 

            u.append(utemp)
            v.append(vtemp)
                        

        try:
            summary = ms.summary()
        except:
            print "Warning: Could not create a summary"
            summary = None        
    
        ms.close()

        if mode =='tot':

            if noise_est == 'full':
                sigma[sigma==0.] = np.median(sigma)
            return vis, sigma, u, v, freq, nchan, nspw,\
                nvis, summary


        if mode == 'pol':
	  
            return Qvis, Qsigma, Uvis, Usigma, freq, lamb,\
                u, v, nchan, nspw, summary
  

def write_data_to_ms(msname, vis, u = None, v = None, weights = None, \
    freq=False, mode='tot'):
        
        
    """
    Function that writes simulated data, and/or uv-coverages to a measurement
    file. The function only works in a very limited fashion, i.e. it can only
    write data/u/v files to an existing ms-set of the exact same size as the
    arrays to be saved.
    ATTENTION: MS NEEDS to be in correlation mode I,Q,U,V for this routine to
    work!
    
    Args:
        msname: name of MS to write to.
        vis: for mode='tot' a Stokes I numpy array
             for mode='pol' a list [Q,U]
             for mode='full' a list [I,Q,U]
        freq: False for full wideband data writing.
              [spw,chan] for writing only specific frequencies. 
    """
    
    m = M.Messenger(2)

    if mode == 'pol':
        m.header2("Writing polarization data to the MeasurementSet...")
    if mode == 'tot':
        m.header2("Writing total intensity data to the MeasurementSet...")
    if mode == 'full':
        m.header2("Writing I,Q,U data to the MeasurementSet...")
    
    try:
        ms.open(msname,nomodify=False)
    except IOError:
        raise IOError('No Measurement Set with the given name')
        
    if np.any(vis):
        if freq:
            ms.selectinit(datadescid=freq[0])
            fullvis = ms.getdata('data')
            if mode == 'tot':
                fullvis['data'][0,freq[1]] = vis
                ms.putdata(fullvis)
            if mode == 'pol':
                fullvis['data'][1,freq[1]] = vis[0]
                fullvis['data'][2,freq[1]] = vis[1]
                ms.putdata(fullvis)
            if mode == 'full':
                fullvis['data'][0,freq[1]] = vis[0]
                fullvis['data'][1,freq[1]] = vis[1]
                fullvis['data'][2,freq[1]] = vis[2]
                ms.putdata(fullvis)
        else:
            fullvis = ms.getdata('data')
            if mode == 'tot':
                fullvis['data'][0] = vis
                ms.putdata(fullvis)
            if mode == 'pol':
                fullvis['data'][1] = vis[0]
                fullvis['data'][2] = vis[1]
                ms.putdata(fullvis)
            if mode == 'full':
                fullvis['data'][0] = vis[0]
                fullvis['data'][1] = vis[1]
                fullvis['data'][2] = vis[2]
                ms.putdata(fullvis)
                
    if u or v or weights:
        m.warning("u,v or weights writing not implemented yet")
        
    ms.close()
    m.header2("...done.")
        
            
        
    
def simulate_ms_file(modelimage, msname, instrumentmodel):
    """
    Routine that uses a model file defined as outlined on the webpage
    https://casaguides.nrao.edu/index.php/Antenna_List to produce a 
    ms-set of a fictious instrument with the specified uv-coverage. Note that
    properly defining the instrument model file takes some work. Note also that
    there are a lot more parameters available for simobserve, check the task
    description for details.
    
    Args:
        modelimage: string holding the path of the sky model casa image.
        msname: string output prefix.
        instrumentmodel: instrument model file for the fictious interferometer. 
    """

    ms.simobserve(project=msname,skymodel=modelimage,antennalist=instrumentmodel)

def computenoise(meta, nspw):
    
    ants = meta.antennaids()

    # loop over al baselines    
    for ant1 in ants:
        for ant2 in ants:
            if ant1 != ant2:
                ms.select({'antenna1':[ant1],'antenna2':[ant2]})
                for spw in nspw:
                    ms.selectinit(datadescid=spw)
                    vistemp = ms.getdata('data')['data']
                    sigma = np.std(np.abs(vistemp[0,29]-vistemp[0,30]))
                
    
def computenoise_martin(vis,datacolumn,m,minsamp=10,):

## estimate weights by inverse variance of visibilities in
## each baseline and spw, ## using all times
## works well for short observations where the rms is stable in time
## TBD: implement time binning for longer tracks
## NOTE: does not work well on data with bright sources
##       use uvsub to remove bright sources before running
##       then use uvsub(reverse=T) to put the sources back in
##
## inputs:   invis      ::   visibility ms
##           spwlist    ::   list of spectral windows to process
##                           default: '' (all)
##           minsamp    ::   minimum number of visibilities to estimate ##                           sample variance
##                           fewer visibilities means weight = 0
##           datacolumn ::   'corrected_data' (default)  or 'data'
##


    import numpy
    ms.open(vis,nomodify=False)

    
    
    spwlist = ms.getspectralwindowinfo().keys()

    ## loop over spectral windows
    for spw in spwlist:
        undersample = 0
        ms.msselect({'spw':spw})
        a = ms.getdata([datacolumn,"flag","sigma"],ifraxis=True)
        b = ms.getdata([datacolumn,"flag","weight"],ifraxis=True)
        d = a[datacolumn]
        f = a["flag"]
        m.message("old sigma: " + str(np.mean(a["sigma"])) + " in spw" \
            + str(spw))
        w = a["sigma"]
        ## loop over corr & baseline
        for corr in range(d.shape[0]):
            for base in range(d.shape[2]):
                dd = numpy.ravel(d[corr,:,base])
                ff = numpy.ravel(f[corr,:,base])
                gooddata = numpy.compress(ff==False,dd)
                if len(gooddata) < minsamp:
                    undersample += 1
                    m.failure('Cannot estimate sigma values from so few' \
                            + 'baseline values. Not performung estimate.')
                    break
                else:
                    if len(gooddata) <= 1:
                        m.failure('Cannot estimate sigma values from single' \
                            + 'baseline values. Not performing estimate.')
                        break
                    ww = numpy.std(gooddata)
                w[corr,base,:] = ww
        ## ugly trick to get the right data type:
        a["sigma"]=a["sigma"]*0.00+w
        m.message("new sigma: " + str(np.mean(a["sigma"]))+ " in spw" \
            + str(spw))
        b["weight"]=b["weight"]*0.00+w**-2 
        if undersample > 1:
           m.warn('Very low number of measurements per baseline in spw' + \
               str(spw) + ". Bad statistics expected.") 
        ms.putdata(a)
        ms.reset()

    ms.close()                              


         


