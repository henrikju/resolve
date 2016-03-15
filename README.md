# RESOLVE - A Bayesian algorithm for aperture synthesis imaging in radio astronomy

RESOLVE is a Bayesian inference algorithm for image reconstruction in radio interferometry. It is optimized for extended and diffuse sources. Features include:

- parameter-free Bayesian reconstruction of radio continuum data with a focus on extended and weak diffuse sources.
- reconstruction with uncertainty propagation dependent on measurement noise.
- estimation of the spatial correlation structure of the radio astronomical source.
- full support for measurement sets.
- included simulation tool (if uv-coverage is provided).

For the moment, the algorithm won't perform optimally under the presence of strong point sources and with wide field data.

ATTENTION: The newest release (March 15, 2016) included a major overhaul for which this description is not suitable any more. An update of the description will follow soon.

--------------------------------------------------------------------------------------------------------------------------------
Quick installation guide
--------------------------------------------------------------------------------------------------------------------------------

Prerequisite libraries:
- the statistical inference package [nifty](https://github.com/information-field-theory/nifty)
- the general Fourier transform package [gfft](https://github.com/mrbell/nifty)
- the radio astronomical data reduction package CASA

RESOLVE is started as a python function directly from an opened CASA shell by simply importing resolve.py as a python module. This is needed for the moment to ensure full measurement set support. To this end, it might be needed to make sure that CASA and nifty are using the same python installation (since CASA comes with its own python package). If needed, you can simply add your global python installation to the CASA library path. A more independent version with direct command line syntax is in work.

------------------------------------------------------------------------------------------------------------------------------

For more information, soon, you can refer to the [RESOLVE Wiki](https://github.com/henrikju/resolve/wiki).

RESOLVE is licensed under the [GPLv3](http://www.gnu.org/licenses/gpl.html).
