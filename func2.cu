#include "reconv.h"
#include "asteroid_search.h"


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
/*
void fits_error(int status)
{
    if (status) {
        fits_report_error(stderr, status);
        exit(EXIT_FAILURE);
    }
}
*/
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
	
	for (long i=0; i<*Npix; i++)
	{
		(*h_image)[i] = buf0[i];
	}
	
	return;
	
}
/* ------------------------------------------------ */

	void subtract_background(int i_image, float *img, int Nx, int Ny, float sgm)
	{
	if (i_image == 0)
	{

	}

		long N = Nx * Ny;		
		float *img_bkg = (float *)malloc(sizeof(float) * N);

//		gauss_blur(Nx, Ny, img, img, 5*sgm); // background

	
		gauss_blur(Nx, Ny, img, img_bkg, 5*sgm); // background
		gauss_blur(Nx, Ny, img, img, sgm); // image with noise blurred
		
		// Subtracting the background from the image:
		for (long i=0; i<N; i++)
			img[i] = img[i] - img_bkg[i];
	
	
		free(img_bkg);
	}

/* ------------------------------------------------ */


/*
	void gauss_blur_cuda(int i_image, int Nx, int Ny, float* img_in, float* img_out, float sgm)
	{
	if (i_image == 0)
	{
//		cufftPlan2D()
	}
		
		// cufftExecR2C() 
		
	}
	*/
	
	
	void dump_fits (int Nx, int Ny, int Nc, float *img, const char *name)
	// Dump a 2D image into a FITS file (for debugging)
	{
		int status=0; 
		
		fitsfile *fk;
		fits_create_file(&fk, name, &status);
		fits_error(status);
		long nelem1  = (long)Nx * Ny * Nc;
//		long naxes[3] = {Ny, Nx, Nc};
//		fits_create_img(fk, FLOAT_IMG, 3, naxes, &status);
		long naxes[2] = {Ny, Nx};
		fits_create_img(fk, FLOAT_IMG, 2, naxes, &status);
		fits_error(status);
		fits_write_img(fk, TFLOAT, 1, nelem1, img, &status);
		fits_error(status);
		fits_close_file(fk, &status);		
		
		return;
	}
	