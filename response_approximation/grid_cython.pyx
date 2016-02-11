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