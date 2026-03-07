#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>
#include <fitsio.h>
#include <string.h>
#ifdef _OPENMP
  // OpenMP is enabled
  #include <omp.h>
#endif

#define IDX(x,y,Ny) ((x)*(Ny) + (y))



void fits_error(int status);

void fft_image(int Nx, int Ny,
                double *img0,
                fftw_complex *F0);

void derive_kernel_ft(int N, fftw_complex *F1,
                      fftw_complex *F0,
                      fftw_complex *K,
                      double eps);
					  
void ifft_kernel(int Nx, int Ny,
                 fftw_complex *K,
                 fftw_complex *k_spatial);
				 
void ifft_kernel(int Nx, int Ny,
                 fftw_complex *K,
                 fftw_complex *k_spatial);
				 
void truncate_kernel(int Nx, int Ny,
                     fftw_complex *k,
                     double R);

void fft_kernel(int Nx, int Ny,
                fftw_complex *k_spatial,
                fftw_complex *K);

void convolve_image(int Nx, int Ny,
                    fftw_complex *F0,
                    fftw_complex *K,
                    double *out);

void fft_images_padded(int Nx, int Ny,
                       int Px, int Py,
                       double *img0,
                       fftw_complex *F0,
					   int Pad);

void crop_image_centered(int Nx, int Ny,
                         int Px, int Py,
                         double *padded,
                         double *out);

void sigma_clipping(double *image, long plane_pixels, double Nsigma, double *p0, double *sgm, long *Npix, int *k);

double scaling (double *image1, double *master, long plane_pixels, double p1_low, int *k);

void gauss_blur(int Nx, int Ny, double* img, double sgm);

int timeval_subtract (double *result, struct timeval *x, struct timeval *y);

#ifdef _OPENMP
void init_all_locks();
#endif	

#ifndef MYMAIN_H
  #define MYMAIN_H
  extern int verbose;
#endif


