# RESOLVE - A Bayesian algorithm for aperture synthesis imaging in radio astronomy

-------------------------------------------------------------------------------------------------------------------------------
## ATTENTION: part of the code is constantly being updated with new features, and so will be heavily non-tested and not well documented. 
## Please contact Henrik.Junklewitz@gmail.com for any details or problems. 
-------------------------------------------------------------------------------------------------------------------------------

RESOLVE is a Bayesian inference package for image reconstruction in radio interferometry.

### Stable features:

- parameter-free Bayesian image reconstruction of radio continuum data with a focus on extended and weak diffuse sources.
- approximate posterior uncertainty map for the image dependent on the reconstructed image and on estimated measurement noise.
- estimation of the spatial correlation structure of the radio astronomical source.
- fastResolve module which speeds up the computation heavy standard algorithm using a gridded likelihood function with minimized information loss.
- support for measurement sets.
- included simple radio interferometric simulation tool (if uv-coverage is provided).

### Experimental (less tested) features:

- PointResolve module, allowing for the simultaneous imaging and separation of point and extended sources. 
- rough noise estimation routine, additional inferring the visibility noise variance; 
possibly more robust in high noise cases, against outliers or wrongly measured noise
- usage of wsclean(LINK) functions for performance and functionality can be activated. Mostly gridding routines and w-stacking. Wsclean needs to be installed.

### Highly Experimental (only really tested in simulations):

- Multi-frequency RESOLVE, allowing for simultaneous reconstruction of total intensity and spectral index.

--------------------------------------------------------------------------------------------------------------------------------
## Quick installation guide
--------------------------------------------------------------------------------------------------------------------------------

### Prerequisite libraries:
- the full python package: numpy, scipy, pyfits, matplotlib, cython
- the statistical inference package [nifty](https://github.com/information-field-theory/nifty)
- the general Fourier transform package [gfft](https://github.com/mrbell/nifty)
- the radio astronomical data reduction package CASA for meausrement set support.
- optional: the python-casacore module for direct import of measurement sets. 
- optional: the radio imager wsclean for optional use of its gridding routines.
- optional: the astropy package for astronomical coordinate system support in numpy images. 

--------------------------------------------------------------------------------------------------------------------------------
## Quick start guide
--------------------------------------------------------------------------------------------------------------------------------

RESOLVE is a python package with much functionality actually being deferred to underlying C routines. For the end user it is 
simply started as a command line python tool using the main function resolve.py:

python resolve.py *data_pathname imsize cellsize resolve mode*

### Options:

-s custom save directory suffix

-p use python-casacore module for direct read-in of measurement sets (as opposed to the "casatools" procedure indicated below).

-v verbosity, range 1(only headers) to 5(diagnostic outputs); default:2

### Arguments:

*data_pathname* 

Path to the visibility data.

It is possible to directly read in measurement sets using the python-casacore module with the "-p" flag. The <data pathname> points to	   the measurememt set.

Since this assumes a full installation of casacore and python-casacore there is a fall-back alternative standard. For this RESOLVE assumes the data to be present in a specific numpy-save format. Measurement Sets can be read into this format with a supplied routine 'read_data_from_CASA' in the casa_tools package module which needs to be called from the CASA prompt. To do this, the following steps need to be taken:
1) Start CASA from the casatools directory and type 'import resolve_casa_functions'.
2) From the imported python module start the function 'read_data_withCASA',
e.g. as resolve_casa_functions.read_data_withCASA(<ms-filename>, save=<data directory where to read the RESOLVE-numpy format to>)
[the full function call with all keyword arguments is
read_data_withCASA(ms, viscol="DATA", noisecol='SIGMA',mode='tot', noise_est = False, save=None); more in the wiki]
In the end <data pathname> will be the directory that holds the numpy-save data.

If the wsclean functions are used, this needs to be again the pathname of the measurement set and no further procedure, neither with CASA or casacore is needed.

(A more generalized interface for RESOLVE to measurement sets might get developed at some point.)

*imsize* 

Size of image in numbers of pixels of one axis. Choose as in standard CLEAN imaging.

*cellsize* 

Size of one pixel in rad. Choose as in standard CLEAN imaging.

*resolve_mode* 

Activates one of the available RESOLVE modes, employing different features of the package with sets of 
(more or less) robust default parameters for the underlying inference and optimization routines. The idea is that for
'standard cases' and high SNR, the end user only has to choose from these and doesn't need to bother with the complex
numerical details. Every interested user can define custom modes and change virtually all parameters and combinations 
of sub-routines. This way the more experimental features like PointResolve or MF-Resolve can also be explored.
For details see the wiki. 
Current default modes:

- 'resolve': standard extended sources prior only (link paper). Full gridding and many changes between
 visibility and image space. In principle accurate but very slow. Bright point sorces will cause a problem and
 possibly be smoothed out

- 'resolve_fast': standard extended sources prior only (link paper). Uses the fastresolve approximation to speed up
computation time (uo to a few 100 times faster). Bright point sorces will cause a problem and
 possibly be smoothed out.

- 'standard_resolve': runs first fastresolve to obtain a good starting guess for a follow up, more accurate Resolve 
reconstruction. Conceptionally similar to classical major-minor cycles in CLEAN imaging. Should be the "standard blind" choice.

- '*_uncertainty': calculate an approximate posterior uncertainty for a given reconstructed image for the given setup in '*'. Stand-alone uncertainty calculation. Last image needs to be specified by hand in

- '*_simulation': run the chosen RESOLVE mode while simulating a data set. This mode needs a given uv-coverage to work.


--------------------------------------------------------------------------------------------------------------------------------
## Relevant Publications
--------------------------------------------------------------------------------------------------------------------------------

- RESOLVE: A new algorithm for aperture synthesis imaging of extended emission in radio astronomy
http://adsabs.harvard.edu/abs/2016A%26A...586A..76J

- A new approach to multifrequency synthesis in radio interferometry 
http://adsabs.harvard.edu/abs/2015A%26A...581A..59J

- fastRESOLVE: fast Bayesian imaging for aperture synthesis in radio astronomy
http://adsabs.harvard.edu/cgi-bin/nph-data_query?bibcode=2016arXiv160504317G&db_key=PRE&link_type=ABSTRACT

- NIFTY - Numerical Information Field Theory. A versatile PYTHON library for signal inference
http://adsabs.harvard.edu/cgi-bin/nph-data_query?bibcode=2013A%26A...554A..26S&db_key=AST&link_type=ABSTRACT

- Information field theory for cosmological perturbation reconstruction and nonlinear signal analysis
http://adsabs.harvard.edu/cgi-bin/nph-data_query?bibcode=2009PhRvD..80j5005E&db_key=AST&link_type=ABSTRACT&high=5508ca396122870


----------------------------------------------------------------------------------------------------------------------------

For more information, soon, you can refer to the [RESOLVE Wiki](https://github.com/henrikju/resolve/wiki).

RESOLVE is licensed under the [GPLv3](http://www.gnu.org/licenses/gpl.html).














