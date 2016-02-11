#include <math.h>
#include <complex.h>
#include <stdio.h>
#include "grid_C.h"

double sinc(double x){
    double res;
    if(x == 0.) res = 1.;
    else{
        res = 3.1415926539*x;
        res = sin(res)/res;
    }
    return res;
}


void grid_complex(unsigned int Nk, unsigned int Nq,
                  double complex grid[Nk][Nq],
                  double vk, double vq,
                  unsigned int Nuv,
                  double complex inpoints[Nuv],
                  double u[Nuv], double v[Nuv],
                  unsigned int precision){

int i; int j; int l; //iteration variables

double vk_inv = 1./vk;
double vq_inv = 1./vq;

double V_k = Nk*vk;
double V_q = Nq*vq;

int shift_k = Nk/2;
double pos_k[Nk];
for(i=0; i<Nk; i++) pos_k[i] = (i-shift_k)*vk;

int shift_q = Nq/2;
double pos_q[Nq];
for(i=0; i<Nq; i++) pos_q[i] = (i-shift_q)*vq;

int ind_k;
int ind_q;
int max_ind_k;
int max_ind_q;
int min_ind_k;
int min_ind_q;

double tempdouble;
double complex tempcomplex;
double complex temp_k[precision*2+1];
double complex temp_q[precision*2+1];

double pi_overV_k = 3.1415926539 / V_k;
double pi_overV_q = 3.1415926539 / V_q;

for(i=0; i<Nuv; i++){

    ind_k = (int)(u[i]*vk_inv);
    ind_k += Nk/2;

    ind_q = (int)(v[i]*vq_inv);
    ind_q += Nq/2;

    min_ind_k = ind_k - precision;
    max_ind_k = ind_k + precision + 1;
    if(min_ind_k < 0) min_ind_k = 0;
    if(max_ind_k < 0) max_ind_k = 0;
    if(min_ind_k > Nk) min_ind_k = Nk;
    if(max_ind_k > Nk) max_ind_k = Nk;

    min_ind_q = ind_q - precision;
    max_ind_q = ind_q + precision + 1;

    if(min_ind_q < 0) min_ind_q = 0;
    if(max_ind_q < 0) max_ind_q = 0;
    if(min_ind_q > Nq) min_ind_q = Nq;
    if(max_ind_q > Nq) max_ind_q = Nq;

    for(j=min_ind_k; j<max_ind_k; j++){
            tempdouble = vk_inv*sinc((pos_k[j]-u[i])*vk_inv);
            tempcomplex = 0. + (pi_overV_k*(pos_k[j]-u[i])) * I;
            temp_k[j-min_ind_k] = cexp(tempcomplex) * tempdouble;
    }

    for(l=min_ind_q; l<max_ind_q; l++){
            tempdouble = vq_inv*sinc((pos_q[l]-v[i])*vq_inv);
            tempcomplex = 0. + (pi_overV_q*(pos_q[l]-v[i])) * I;
            temp_q[l-min_ind_q] = cexp(tempcomplex) * tempdouble;
    }

    for(j=min_ind_k; j<max_ind_k; j++){
        for(l=min_ind_q; l<max_ind_q; l++){
            grid[j][l] += temp_k[j-min_ind_k] * temp_q[l-min_ind_q] * inpoints[i];
        }
    }
}

}


void grid_abs_squared(unsigned int Nk, unsigned int Nq,
                      double grid[Nk][Nq],
                      double vk, double vq,
                      unsigned int Nuv,
                      double inpoints[Nuv],
                      double u[Nuv], double v[Nuv],
                      unsigned int precision){

int i; int j; int l; //iteration variables

double vk_inv = 1./vk;
double vq_inv = 1./vq;

int shift_k = Nk/2;
double pos_k[Nk];
for(i=0; i<Nk; i++) pos_k[i] = (i-shift_k)*vk;

int shift_q = Nq/2;
double pos_q[Nq];
for(i=0; i<Nq; i++) pos_q[i] = (i-shift_q)*vq;

int ind_k;
int ind_q;
int max_ind_k;
int max_ind_q;
int min_ind_k;
int min_ind_q;

double tempdouble;
double temp_k[precision*2+1];
double temp_q[precision*2+1];

for(i=0; i<Nuv; i++){

    ind_k = (int)(u[i]*vk_inv);
    ind_k += Nk/2;

    ind_q = (int)(v[i]*vq_inv);
    ind_q += Nq/2;

    min_ind_k = ind_k - precision;
    max_ind_k = ind_k + precision + 1;
    if(min_ind_k < 0) min_ind_k = 0;
    if(max_ind_k < 0) max_ind_k = 0;
    if(min_ind_k > Nk) min_ind_k = Nk;
    if(max_ind_k > Nk) max_ind_k = Nk;

    min_ind_q = ind_q - precision;
    max_ind_q = ind_q + precision + 1;

    if(min_ind_q < 0) min_ind_q = 0;
    if(max_ind_q < 0) max_ind_q = 0;
    if(min_ind_q > Nq) min_ind_q = Nq;
    if(max_ind_q > Nq) max_ind_q = Nq;

    for(j=min_ind_k; j<max_ind_k; j++){
            tempdouble = vk_inv*sinc((pos_k[j]-u[i])*vk_inv);
            temp_k[j-min_ind_k] = tempdouble * tempdouble;
    }

    for(l=min_ind_q; l<max_ind_q; l++){
            tempdouble = vq_inv*sinc((pos_q[l]-v[i])*vq_inv);
            temp_q[l-min_ind_q] = tempdouble * tempdouble;
    }

    for(j=min_ind_k; j<max_ind_k; j++){
        for(l=min_ind_q; l<max_ind_q; l++){
            grid[j][l] += temp_k[j-min_ind_k] * temp_q[l-min_ind_q] * inpoints[i];
        }
    }
}

}


const double pi=3.141592653589793238462643383279502884197;

void grid_Martinc(unsigned int Nk, unsigned int Nq,
                  double complex grid[Nk][Nq],
                  double vk, double vq,
                  unsigned int Nuv,
                  double complex inpoints[Nuv],
                  double u[Nuv], double v[Nuv],
                  unsigned int precision){

double vk_inv = 1./vk;
double vq_inv = 1./vq;

int shift_k = Nk/2;
int shift_q = Nq/2;

double complex temp_q[precision*2+1];

for(int i=0; i<Nuv; i++){
  int ind_k = (int)(u[i]*vk_inv) + shift_k;
  int ind_q = (int)(v[i]*vq_inv) + shift_q;

  int min_ind_k = ind_k - precision;
  int max_ind_k = ind_k + precision + 1;
  min_ind_k = (min_ind_k<0) ? 0 : ((min_ind_k>Nk) ? Nk : min_ind_k);
  max_ind_k = (max_ind_k<0) ? 0 : ((max_ind_k>Nk) ? Nk : max_ind_k);

  int min_ind_q = ind_q - precision;
  int max_ind_q = ind_q + precision + 1;
  min_ind_q = (min_ind_q<0) ? 0 : ((min_ind_q>Nq) ? Nq : min_ind_q);
  max_ind_q = (max_ind_q<0) ? 0 : ((max_ind_q>Nq) ? Nq : max_ind_q);

{
  double pos1=((min_ind_q-shift_q)*vq-v[i])*vq_inv*pi;
  double sinp1=sin(pos1);
  double pos2=pi/(Nq*vq)*((min_ind_q-shift_q)*vq-v[i]);
  complex double exppos2=cos(pos2) + I*sin(pos2);
  complex double expdpos2=cos(pi/Nq) + I*sin(pi/Nq);
  for(int l=min_ind_q; l<max_ind_q; l++){
    double tempdouble = (pos1==0.) ? vq_inv : vq_inv*sinp1/pos1;
    temp_q[l-min_ind_q] = exppos2 * tempdouble;
    exppos2*=expdpos2;
    pos1+=pi;
    sinp1=-sinp1;
  }
}

{
  double pos1=((min_ind_k-shift_k)*vk-u[i])*vk_inv*pi;
  double sinp1=sin(pos1);
  double pos2=pi/(Nk*vk)*((min_ind_k-shift_k)*vk-u[i]);
  complex double exppos2=cos(pos2) + I*sin(pos2);
  complex double expdpos2=cos(pi/Nk) + I*sin(pi/Nk);
  for(int j=min_ind_k; j<max_ind_k; j++){
    double tempdouble = (pos1==0.) ? vk_inv : vk_inv*sinp1/pos1;
    double complex t = exppos2 * tempdouble * inpoints[i];
    for(int l=min_ind_q; l<max_ind_q; l++){
      grid[j][l] += t * temp_q[l-min_ind_q];
    }
    exppos2*=expdpos2;
    pos1+=pi;
    sinp1=-sinp1;
  }
}

}
}
