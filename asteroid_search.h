#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>
#include <fitsio.h>
#include <string.h>

#ifndef ASTEROID_SEARCH_H
  #define ASTEROID_SEARCH_H
  #include "cuda_errors.h"

#ifdef _OPENMP
  // OpenMP is enabled
  #include <omp.h>
#endif


//void fits_error(int status);

void Is_GPU_present();

int timeval_subtract (double *result, struct timeval *x, struct timeval *y);

void image_bw(float *image, long Npix, int Nc);

void crop(float *buf0, int *Nx, int *Ny, long *Npix, float crop_fraction);

void rebin(int i_image, float *buf0, int *Nx, int *Ny, long *Npix, float** h_image);

void subtract_background(int i_image, float *img, int Nx, int Ny, int NTx, int NTy);

int date2mjd (int yr, int mn, int dy);

__global__ void motion_search_cuda (float **d_image, int N_images, int Ix1, int Iy1, int Jx, int Jy, float MQ, float p_min);

#endif
