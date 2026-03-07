#include "asteroid_search.h"

int timeval_subtract (double *result, struct timeval *x, struct timeval *y)
{
  struct timeval result0;

  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_usec < y->tv_usec) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
    y->tv_usec -= 1000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_usec - y->tv_usec > 1000000) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000;
    y->tv_usec += 1000000 * nsec;
    y->tv_sec -= nsec;
  }

  /* Compute the time remaining to wait.
     tv_usec is certainly positive. */
  result0.tv_sec = x->tv_sec - y->tv_sec;
  result0.tv_usec = x->tv_usec - y->tv_usec;
  *result = ((double)result0.tv_usec)/1e6 + (double)result0.tv_sec;

  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}


void Is_GPU_present()
{
  int devid, devcount;
  /* find number of device in current "context" */
  cudaGetDevice(&devid);
  /* find how many devices are available */
  if (cudaGetDeviceCount(&devcount) || devcount==0)
    {
      printf ("No CUDA devices!\n");
      exit (1);
    }
  else
    {
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties (&deviceProp, devid);
      printf ("Device count, devid: %d %d\n", devcount, devid);
      printf ("Device: %s\n", deviceProp.name);
      printf ("Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
      printf ("Maximum treads per block: %d\n", deviceProp.maxThreadsPerBlock);
      printf ("Maximum block grid dimensions: %d x %d x %d\n\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    }
  return;
}



/* ------------------------------------------------ */
void fits_error(int status)
{
    if (status) {
        fits_report_error(stderr, status);
        exit(EXIT_FAILURE);
    }
}

/* ------------------------------------------------ */

void image_bw(float *image, long Npix, int Nc)
{
    for (long i = 0; i < Npix; i++)
    {
		// The colors are RGB; storing the B&W image into the red channel
		image[i] = 0.25*(image[i] + 2*image[Npix + i] + image[2*Npix + i]);
	}

	return;
}
/* ------------------------------------------------ */


void crop_and_rebin(int i_image, float *buf0, int *Nx, int *Ny, long *Npix, float** h_image)
{
	if (i_image == 0)
	{
		// We use cudaMallocHost instead of malloc to put the array in the pinned host memory
		ERR( cudaMallocHost((void**)h_image, sizeof(float)* *Npix) )
	}
printf("5\n");	
	for (long i=0; i<*Npix; i++)
	{
		(*h_image)[i] = buf0[i];
//printf("%d %f\n",i,buf0[i]);
	}
	
	return;
	
}
/* ------------------------------------------------ */
