#ifndef RECONV_H
#define RECONV_H


#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>
#include <fitsio.h>
#include <string.h>
#include <assert.h>

// CLoseness parameter for cluster members during cluster analysis
// Possible range : 1...4
// 1: the tightest clusters possible (only one out of 4 dimensions differs by 1)
// 4: the loosest clusters possible (all 4 dimensions may differ by 1)
// 1 is the best when minimizing the chance that two closely located clusters merge into one
// 4 is the best when minimizing the chance that the part of a cluster
// is mistakenly assigned to another cluster
#define CL_MAX 4

// Mask test value:
#define MASK -100.0  // for if() statements
#define MASK0 -101.0 // for assigning

#define IDX(x,y,Ny) ((x)*(Ny) + (y))

#ifdef __cplusplus
extern "C" {
#endif

void my_alloc_init (int P);
float* my_alloc_real ();
fftw_complex* my_alloc_complex ();
void my_free_real(float *p);
void my_free_complex(fftw_complex *p);
void my_alloc_destroy();

void fits_error(int status);

void fft_image(int Nx, int Ny,
                float *img0,
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
                    float *out);

void fft_images_padded(int Nx, int Ny,
                       int Px, int Py,
                       float *img0,
                       fftw_complex *F0,
					   int Pad);

void crop_image_centered(int Nx, int Ny,
                         int Px, int Py,
                         float *padded,
                         float *out);

void sigma_clipping(float *image, long plane_pixels, double Nsigma, double *p0, double *sgm, long *Npix, int *k);

float scaling (float *image1, float *master, long plane_pixels, double p1_low, int *k);

void gauss_blur(int Nx, int Ny, float* img, float *img_out, float sgm);

int timeval_subtract (double *result, struct timeval *x, struct timeval *y);

void dump_fits (int Nx, int Ny, int Nc, float *img, const char *name);

void highpass_filter(const float *input, float *output,
                     int rows, int cols, double cutoff);

void image_bw(float *image, long Npix, int Nc);

void crop(float *buf0, int *Nx, int *Ny, long *Npix, float crop_fraction);

void rebinning(float *buf0, int Nx_ini, int Ny_ini, int Nx, int Ny, double bias, int rebin, float *image);

void subtract_background(int i_image, int N_images, float *img, int Nx, int Ny, int NTx, int NTy, float bias);

int date2mjd (int yr, int mn, int dy);

//void compute_histogram(float *image, long Npix, float sgm, float *p_min_std, long *hist);
	
void borders(float *img, int Nx, int Ny, int BW);	

void grow_masked_stars(float *img, int Nx, int Ny, float mask_sgm, int *N_excluded);

void erase_hot_pixels(float *img, int Nx, int Ny, float p_max);

void string_is_a_float (const char *str);

#ifdef __cplusplus
}
#endif

#endif