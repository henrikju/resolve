"""
resolve.py

Written by Henrik Junklewitz based on scipts by Maksim Greiner

grid_cython.py provides the cython functions speeding up essential parts
of the fastresolve mode.

Copyright 2016 Henrik Junklewitz

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

# distutils: sources = grid_C.c
import numpy as np
cimport numpy as np
from cpython cimport array

cdef extern from "complex.h":
    double complex cexp(double complex)

cdef extern from "grid_C.h":
    void grid_complex(unsigned int Nk, unsigned int Nq,
                      double complex** grid,
                      double vk, double vq,
                      unsigned int Nuv,
                      double complex* inpoints,
                      double* u, double* v,
                      unsigned int precision)

    void grid_abs_squared(unsigned int Nk, unsigned int Nq,
                          double** grid,
                          double vk, double vq,
                          unsigned int Nuv,
                          double* inpoints,
                          double* u, double* v,
                          unsigned int precision)

    void grid_Martinc(unsigned int Nk, unsigned int Nq,
                      double complex** grid,
                      double vk, double vq,
                      unsigned int Nuv,
                      double complex* inpoints,
                      double* u, double* v,
                      unsigned int precision)


def grid_complex_cython(int Nk, int Nq, double vk, double vq,
                        np.ndarray[np.complex128_t, ndim=1] inpoints,
                        np.ndarray[np.float64_t, ndim=1] u,
                        np.ndarray[np.float64_t, ndim=1] v,
                        int precision):

    cdef np.ndarray[np.complex128_t, ndim=2] result = np.zeros((Nk,Nq), dtype=np.complex128)

    cdef int Nuv = len(inpoints)

    grid_complex(Nk, Nq,  <double complex**> result.data, vk, vq, Nuv,
                 <double complex*> inpoints.data, <double*> u.data,
                 <double*> v.data, precision)

    return result


def grid_abs_squared_cython(int Nk, int Nq, double vk, double vq,
                            np.ndarray[np.float64_t, ndim=1] inpoints,
                            np.ndarray[np.float64_t, ndim=1] u,
                            np.ndarray[np.float64_t, ndim=1] v,
                            int precision):

    cdef np.ndarray[np.float64_t, ndim=2] result = np.zeros((Nk,Nq), dtype=np.float64)

    cdef int Nuv = len(inpoints)

    grid_abs_squared(Nk, Nq,  <double**> result.data, vk, vq, Nuv,
                     <double*> inpoints.data, <double*> u.data,
                     <double*> v.data, precision)

    return result

def grid_complex_Martinc(int Nk, int Nq, double vk, double vq,
                        np.ndarray[np.complex128_t, ndim=1] inpoints,
                        np.ndarray[np.float64_t, ndim=1] u,
                        np.ndarray[np.float64_t, ndim=1] v,
                        int precision):

    cdef np.ndarray[np.complex128_t, ndim=2] result = np.zeros((Nk,Nq), dtype=np.complex128)

    cdef int Nuv = len(inpoints)

    grid_Martinc(Nk, Nq,  <double complex**> result.data, vk, vq, Nuv,
                 <double complex*> inpoints.data, <double*> u.data,
                 <double*> v.data, precision)

    return result
