from distutils.core import setup, Extension
from Cython.Build import cythonize
#from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

"""
setup(ext_modules = cythonize(Extension(
           "grid_cython",                                # the extesion name
           sources=["grid_cython.pyx", "grid_C.c"], # the Cython source and
                                                  # additional C++ source files
           language="c",                        # generate and compile C++ code
      )),
      include_dirs=[numpy.get_include()]
)


"""
setup(
  name = 'grid_cython',
  ext_modules=[
    Extension('grid_cython',
              sources=["grid_cython.pyx", "grid_C.c"],
              extra_compile_args=["-O3", "-lm", "-std=c99"],
              language='c')
    ],
  include_dirs=[numpy.get_include()],
  cmdclass = {'build_ext': build_ext}
)

