from casa import ms
import numpy as np
import Messenger as M

C = 299792458
PI = 3.14159265358979323846

#General purpose routine for MS-IO using CASA
def read_data_from_ms(msfn, viscol="DATA", noisecol='SIGMA',
                      mode='tot',noise_est = False):
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

    if noise_est:
        m.message("Performing simple noise estimate")
        computenoise(msfn,viscol,m,minsamp=10)

    ms.open(msfn)
    meta = ms.metadata()
    nspw = range(meta.nspw())
    nchan = []
    nvis = []
    u = []
    v = []
    freq = []
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

    
    for spw in nspw:
	  nchan.append(meta.nchan(spw))
          
	  ms.selectinit(datadescid=spw)
	  temp = ms.getdata([viscol,"axis_info"],ifraxis=False)
	  data_temp = temp[viscol]
	  info_temp = temp["axis_info"]
	  s_temp = ms.getdata(["sigma"], ifraxis=False)['sigma']
	  flags = 1. - ms.getdata(["flag"])['flag']

          if not(np.sum(flags[0])==np.sum(flags[1])==np.sum(flags[2])==np.sum(flags[3])):
              m.warn('Warning: Different flags for different correlations/channels.'+
            'Hard flag is applied: If any correlation is flagged, this gets'+ 
            'extended to all.')
          maximum = np.ones(np.shape(flags[0]))
          for i in range(4):
              if flags[i].sum() < maximum.sum():
                  maximum = flags[i]
          flag = maximum
                

          #Start reading data.

          if mode == 'tot': 
              vis_temp = flag * (Spart[0] * data_temp[0] + Spart[1] * data_temp[1] + Spart[2] *\
                        data_temp[2] + Spart[3] * data_temp[3])
              sigma_temp = flag * (Spart[0] * s_temp[0] + Spart[1] * s_temp[1] + Spart[2] * s_temp[2] +\
                        Spart[3] * s_temp[3])

              vis.append(vis_temp)
              sigma.append(sigma_temp)

              nvis.append(len(vis_temp[0]))


          if mode == 'pol':
              Qvis_temp = flag * (QSpart[0] * data_temp[0] + QSpart[1] * data_temp[1] + QSpart[2] *\
                         data_temp[2] + QSpart[3] * data_temp[3])
              Qsigma_temp = flag * (QSpart[0] * s_temp[0] + QSpart[1] * s_temp[1] + QSpart[2] * s_temp[2] +\
                         QSpart[3] * s_temp[3])

              Qvis.append(Qvis_temp)
              Qsigma.append(Qsigma_temp)
         
              Uvis_temp = flag * (USpart[0] * data_temp[0] + USpart[1] * data_temp[1] + USpart[2] *\
                         data_temp[2] + USpart[3] * data_temp[3])
              Usigma_temp = flag * (USpart[0] * s_temp[0] + USpart[1] * s_temp[1] + USpart[2] * s_temp[2] +\
                         USpart[3] * s_temp[3])

              Uvis.append(Uvis_temp)
              Usigma.append(Usigma_temp)

              
              nvis.append(len(Uvis_temp[0]))

       
          #uvflat give all uv coordinates for the chosen spw in m
          uflat = ms.getdata(['u'])['u']
          vflat = ms.getdata(['v'])['v']

          freqtemp = info_temp["freq_axis"]["chan_freq"]
          freq.append(freqtemp)
          lamb = C / freqtemp
          if mode == 'pol':
              lambs2.append(lamb**2 / PI)

         

          #calculates uv coordinates per channel in #lambda 
          utemp = np.array([uflat/k for k in lamb])
          vtemp = np.array([vflat/k for k in lamb])
 
          #Reads the uv coordates into lists. Delete functions take care of flags.
          u.append(utemp)
          v.append(vtemp)
    
    summary = ms.summary()
    ms.close()

    if mode =='tot':

	  return vis, sigma, u, v, freq, nchan, nspw, nvis, summary


    if mode == 'pol':
	  
	  return Qvis, Qsigma, Uvis, Usigma, freq, lamb, u, v, nchan, nspw, nvis


#routine to read pyrat data using CASA-ms IO
def read_pyratdata_from_ms(msfn, vis, noise, viscol="DATA", noisecol='SIGMA', mode='pol'):
    """
    Reads polarization or total intensity data in visibility and noise arrays.

    Args:
        msfn: Name of the MeasurementSet file from which to read the data
        visp: Pyrat data object
        noisep: Pyrat data object
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

    #Attention, in current setting, pyrat can't handle properly different flags on separate channels

    # Messenger object for displaying messages
    m = vis.m
    if vis._initialized and noise._initialized:
        m.warn("Requested data objects already exist. Using the " +
               "previously parsed data.")
        return

    if mode == 'pol':
        m.header2("Reading polarization data from the MeasurementSet...")
    if mode == 'tot':
        m.header2("Reading total intensity data from the MeasurementSet...")

    
    viscol = viscol.lower()
    noisecol = noisecol.lower()
    ms.open(msfn)
    meta = ms.metadata()
    nspw = range(meta.nspw())
    nchan = []
    u = []
    v = []
    freq = []
    if mode == 'pol':

      lambs = []
  
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

    m.message(corr_announce)

    
    for spw in nspw:

        nchan.append(meta.nchan(spw))
        ms.selectinit(datadescid=spw)
        temp = ms.getdata([viscol,"axis_info"],ifraxis=False)
        data_temp = temp[viscol]
        info_temp = temp["axis_info"]
        s_temp = ms.getdata(["sigma"], ifraxis=False)['sigma']
        flags = 1. - ms.getdata(["flag"])['flag']
          
        if not(np.sum(flags[0])==np.sum(flags[1])==np.sum(flags[2])==np.sum(flags[3])):
            m.warn('Warning: Different flags for different correlations/channels.\
            Hard flag is applied: If any correlation is flagged, this gets \
            extended to all.')
            maximum = np.ones(np.shape(flags[0]))
            for i in range(4):
                if flags[i].sum() < maximum.sum():
                    maximum = flags[i]
            flag = maximum

        if (np.sum(flag)==0):
            m.warn('Spw ' + str(spw) + ' is completely flagged!\n')
            continue

        #Start reading data. 
        if mode == 'tot': 
	    vis_temp = flag * (Spart[0] * data_temp[0] + Spart[1] * data_temp[1] + Spart[2] *\
                data_temp[2] + Spart[3] * data_temp[3])
	    sigma_temp = flag * (Spart[0] * s_temp[0] + Spart[1] * s_temp[1] + Spart[2] * s_temp[2] +\
                         Spart[3] * s_temp[3])
          

            vislist = np.array(vis_temp)
            sigmalist = np.array(sigma_temp)

        if mode == 'pol':

	    Qvis_temp = flag * (QSpart[0] * data_temp[0] + QSpart[1] * data_temp[1] + QSpart[2] *\
                data_temp[2] + QSpart[3] * data_temp[3])
	    Qsigma_temp = flag * (QSpart[0] * s_temp[0] + QSpart[1] * s_temp[1] + QSpart[2] * s_temp[2] +\
                         QSpart[3] * s_temp[3])
            
            Uvis_temp = flag * (USpart[0] * data_temp[0] + USpart[1] * data_temp[1] + USpart[2] *\
                data_temp[2] + USpart[3] * data_temp[3])
	    Usigma_temp = flag * (USpart[0] * s_temp[0] + USpart[1] * s_temp[1] + USpart[2] * s_temp[2] +\
                         USpart[3] * s_temp[3])
            
            Qvislist = np.array(Qvislist)
            Qsigmalist = np.array(Qsigmalist)
            Uvislist = np.array(Uvislist)
            Usigmalist = np.array(Usigmalist)
            
	  
        # uvflat give all uv coordinates for the chosen spw in m
        uflat = ms.getdata(['u'])['u']
        vflat = ms.getdata(['v'])['v']
          
        freqtemp = (info_temp["freq_axis"]["chan_freq"]).reshape(nchan[spw])
        freq.append(freqtemp)
        lamb = C / freqtemp
        if mode == 'pol':
              lambs.append(lamb)
          
        #calculates uv coordinates per channel in #lambda 
        utemp = np.array([uflat/k for k in lamb])
        vtemp = np.array([vflat/k for k in lamb])
          
        #Reads the uv coordates into lists. Delete functions take care of flags
        u.append(utemp)
        v.append(vtemp)
          
        if mode == 'tot':
            
                 vis.init_subgroup(spw, freqtemp, nrecs)
                 noise.init_subgroup(spw, freqtemp, nrecs)
                 
                 for k in range(nchan[spw]):
                     vis.store_records(vislist[k], spw, k)
                     noise.store_records(sigmalist[k], spw, k)
                     vis.coords.put_coords(uchanlist[k],vchanlist[k],spw,k)                

        if mode == 'pol':
                                  
                 vis.init_subgroup(spw, lamb**2 / PI, nrecs)                 
                 noise.init_subgroup(spw, lamb**2 / PI, nrecs)                 
                 noise_array = np.real(np.sqrt(Qsigmalist * Qsigmalist.conjugate() +
                                       Usigmalist * Usigmalist.conjugate()))
                                   
                 for h in range(nchan[spw]):
                     vis.store_records([Qvislist[h],Uvislist[h]], spw, h)
                     noise.store_records(noise_array[h], spw, h)
                     vis.coords.put_coords(uchanlist[h],vchanlist[h],spw,h)        
                 
    ms.close()


def computenoise(vis,datacolumn,m,minsamp=10,):

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


         


