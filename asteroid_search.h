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


//#define TEST

#define MALLOCHOST  // If defined, use cudaMlalocHost in place of malloc

#ifdef _OPENMP
  // OpenMP is enabled
  #include <omp.h>
#endif

// Base tiles have NBxNB pixels size 31x31.
// It is one less than the 32x32 tiles used in CUDA kernel for image stacking
// 32x32 fits perfectly GPU vector parallelism, also can be handled by 1024
// threads (largest block size allowed).
// Image tiles need to bhave +1 dimensions to account for the neighbouring row and column
// of pixels in the direction of the motion.
#define NB 31

// Maximum number of stored brighest pixels
#define MAX_PIXELS 100000000


// CLoseness parameter for cluster members during cluster analysis
// Possible range : 1...4
// 1: the tightest clusters possible (only one out of 4 dimensions differs by 1)
// 4: the loosest clusters possible (all 4 dimensions may differ by 1)
// 1 is the best when minimizing the chance that two closely located clusters merge into one
// 4 is the best when minimizing the chance that the halo of pixels from another cluster
// is mistakenly assigned to a new cluster
#define CL_MAX 4

struct List
{
	int Jx; // Motion vector
	int Jy;
	int ix; // Base pixel coordinates
	int iy;
	float p; // Stacked pixel brightness
};


//void fits_error(int status);

void Is_GPU_present();

int timeval_subtract (double *result, struct timeval *x, struct timeval *y);

void image_bw(float *image, long Npix, int Nc);

void crop(float *buf0, int *Nx, int *Ny, long *Npix, float crop_fraction);

void rebin(int i_image, float *buf0, int *Nx, int *Ny, long *Npix, float** h_image);

void subtract_background(int i_image, float *img, int Nx, int Ny, int NTx, int NTy);

int date2mjd (int yr, int mn, int dy);

__global__ void motion_search_cuda (float **d_image, int N_images, size_t pitch, int Ix1, int Iy1, int Jx, int Jy, float MQ, float p_min, float *d_dt, float *d_test_image, struct List *d_list, int save_image, unsigned int *d_Pixel_counter, int Nx, int Ny);

void find_kernel_parameters(int Jx, int Jy, float MQ, int Nx, int Ny, dim3 *Grid_size, int *Ix1, int *Iy1);

void cluster_analysis(struct List *list, unsigned int Pixel_counter, int *Cluster_index);


#endif
