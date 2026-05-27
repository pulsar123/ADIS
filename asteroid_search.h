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

#define PI 3.14159265358979323846

// Base tiles have NBxNB pixels size 31x31.
// It is one less than the 32x32 tiles used in CUDA kernel for image stacking
// 32x32 fits perfectly GPU vector parallelism, also can be handled by 1024
// threads (largest block size allowed).
// Image tiles need to bhave +1 dimensions to account for the neighbouring row and column
// of pixels in the direction of the motion.
#define NB 31

// Maximum number of stored brighest pixels
#define MAX_PIXELS 10000000

#define FIND_MAXIMUM_BS 512 // block size for the find_maximum kernel

#define ICLOUD_STATS_MAX 1000  // Number of the top clouds to compute statistics for

// Histogram parameters:
#define NVECTORS 1000 // Number of motion vectors used to figure out p_min
#define DEL_SGM 0.1 // Bin width, in sgm units
#define NBIN 10000


//__device__ int i_free_pixel;
__device__ unsigned int d_N_members1; 

struct List
{
	int *Jx; // Motion vector
	int *Jy;
	int *ix; // Base pixel coordinates
	int *iy;
	float *p; // Stacked pixel brightness
};

struct Cloud
{
	float pmax;
	int imax;
	int ix;
	int iy;
	int Jx;
	int Jy;
	int ix_min;
	int iy_min;
	int Jx_min;
	int Jy_min;
	int ix_max;
	int iy_max;
	int Jx_max;
	int Jy_max;
	int N;
	float mass;	
};


//void fits_error(int status);

void Is_GPU_present();

int timeval_subtract (double *result, struct timeval *x, struct timeval *y);

__global__ void motion_search_cuda (float **d_image, int N_images, size_t pitch, int Ix1, int Iy1, int Jx, int Jy, float MQ, float p_min, float *d_dt, float *d_test_image, List d_list, int save_image, unsigned int *d_Pixel_counter, int Nx, int Ny);

__global__ void subtract_master_image (float **d_image, int N_images, size_t pitch, int Ix1, int Iy1, float *master_image, int Nx, int Ny);

void find_kernel_parameters(int Jx, int Jy, float MQ, int Nx, int Ny, dim3 *Grid_size, int *Ix1, int *Iy1);

void cluster_analysis(List h_list, unsigned int Pixel_counter, int *Cluster_index, int *N_cloud);

void cluster_analysis_cuda(List d_list, unsigned int Pixel_counter, int *h_cloud, int *N_cloud);

__global__ void init_d_cloud(int *d_cloud, unsigned int Pixel_counter);

//__global__ void find_free_pixel(int *d_cloud, unsigned int Pixel_counter, int N_cloud, int *d_members);

__global__ void find_neighbours(int N_members, int N_cloud, unsigned int Pixel_counter, List d_list, int *d_cloud, int *d_members);

__global__ void find_maximum (int step, int N, float *vec, int *index, int *d_cloud, float *vec_out, int *index_out, int N_cloud, int *d_members);

void cloud_stats (List h_list, unsigned int h_Pixel_counter, int N_cloud, int *Cluster_index, 
	Cloud *cloud, float sgm, float MQ, int finetune, int rebin, int d_rebin);

void save_cloud_fits (int Nx_ini, int Ny_ini, int Nx, int Ny, int Nc, float *img, const char *name,
 const char *name0, Cloud *cloud, int i_cloud, float sgm, int rebin, int d_rebin);

__global__ void erase_image (float *image, size_t pitch, int Nx, int Ny, double bias);

void create_mosaic (int Nx, int Ny, List list, unsigned int Pixel_counter, int *Cluster_index, int NCmax, const char *name0, Cloud *cloud);

__global__ void compute_histogram (List d_list, unsigned int Pixel_counter, float p_min, float del_sgm, int *d_hist);

#endif
