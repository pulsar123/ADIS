/* The main program in Asteroid_detector package. Searches for moving star-like objects
in the imaging sequence which was pre-processed with "reconv". Uses GPU acceleration (CUDA).

To compile:

nvcc -O3 asteroid_search.cu func.c func2.cu -lfftw3 -lcfitsio -lm -lz -o asteroid_search

*/

#include "reconv.h"
#include "asteroid_search.h"

int main(int argc, char **argv)
{
    struct timeval  tdr0, tdr1, tdr;
    double restime;
	int status = 0, naxis;
    long naxes[3];
	long Npix;
	int Nx, Ny, Nc, Nx0, Ny0;
	float *buf0; // host
	float *d_img1; // device
	float *d_img2; // device
	size_t pitch;
	char date_obs[30];
	double mjd, mjd0;
	int year, month, day, hour, minute;
    double second, exposure;	
	
	float FWHM = 5.0;

	// Processing command line arguments
	
	if (argc == 1)
	{
		printf("\n Syntax (any order):\n\n");
		printf(" %s  List_of_images\n\n", argv[0]);
		printf("\n");		
		exit(0);
	}
	
	Is_GPU_present();
	
	int N_images = argc - 1;
	
	float **d_image = NULL;
	ERR( cudaMallocHost(&d_image, N_images*sizeof(float*)) )
	float *h_image = NULL;
	double *h_dt = NULL;
	ERR( cudaMallocHost(&h_dt, N_images*sizeof(double*)) )
	double *d_dt = NULL;
	ERR( cudaMalloc(&d_dt, N_images*sizeof(double*)) )
	
	float sgm_Gauss = FWHM / sqrt(8*log(2.0));
	printf("sgm_Gauss = %f\n", sgm_Gauss);
	
	// Reading input FITS images one by one, using cudaMemcpyAsync to do concurrent copying to the GPU
	for (int i_image=0; i_image<N_images; i_image++)
	{		
		// Reading the next input FITS file
		printf("\nReading file %s\n", argv[1+i_image]);
		fitsfile *f0;
		char *file0 = argv[1+i_image];
		fits_open_file(&f0, file0, READONLY, &status);
		fits_error(status);
		
		// Computing time offsets from the first image, in seconds:
		fits_read_key(f0, TSTRING, "DATE-OBS", date_obs, NULL, &status);
		fits_read_key(f0, TDOUBLE, "EXPTIME", &exposure, NULL, &status);
		fits_str2time(date_obs, &year, &month, &day,
                  &hour, &minute, &second, &status);
		if (status) { fits_report_error(stderr, status); return status; }
		mjd = exposure/2.0/86400 + date2mjd (year, month, day) + ((second/60.0+minute)/60.0+hour)/24.0; // Modified Julian Date (UT) of the middle of exposure
		if (i_image == 0)
			mjd0 = mjd;
		h_dt[i_image] = (mjd - mjd0)*86400; //  in seconds
		printf("DATE-OBS: %s : %lf\n", date_obs, h_dt[i_image]);

		
		fits_get_img_dim(f0, &naxis, &status);
		fits_get_img_size(f0, 3, naxes, &status);
		fits_error(status);
		
		if (naxis != 3 || naxes[2] != 3) 
		{
			fprintf(stderr, "Error: expected RGB FITS image\n");
			return EXIT_FAILURE;
		}

	
		if (i_image == 0)
		{
			Nx = naxes[1];
			Ny = naxes[0];
			Nx0 = Nx;
			Ny0 = Ny;
			Nc = 3;
			Npix = (long)Nx * Ny;
			buf0 = (float *)malloc(sizeof(float) * Npix * Nc);
			// These will be used for gaussian blur and background subtraction:
//			ERR( cudaMallocPitch(&d_img1, &pitch, (size_t)(Ny*sizeof(float)), Nx) )
//			ERR( cudaMallocPitch(&d_img2, &pitch, (size_t)(Ny*sizeof(float)), Nx) )
		}
		else
		{
			Nx = naxes[1];
			Ny = naxes[0];
			if (naxes[1] != Nx0 || naxes[0] != Ny0)
			{
				fprintf(stderr, "Error: mismatching input images\n");
				return EXIT_FAILURE;
			}
		}
		
		fits_read_img(f0, TFLOAT, 1, Npix * Nc, NULL, buf0, NULL, &status);
		fits_error(status);
		fits_close_file(f0, &status);		

		image_bw(buf0, Npix, Nc);  // Turn the image into black and white

		float crop_fraction = 1.0;
		crop(buf0, &Nx, &Ny, &Npix, crop_fraction);

		// Background model has these many tiles along each dimension:
		int NTx = 5;
		int NTy = (int)((float)NTx / (float)Nx * (float)Ny + 0.5);
		printf("Background model: %d x %d tiles, %d x %d pixels each\n", NTy, NTx, Ny/NTy, Nx/NTx);
		subtract_background(i_image, buf0, Nx, Ny, NTx, NTy);
//		dump_fits(Nx, Ny, 1, buf0, "image.fits");
		
		rebin(i_image, buf0, &Nx, &Ny, &Npix, &h_image);  // allocates h_image on first call
//		printf("a\n");
		
		// x is for the rows, y is for columns
		// Assuming pitch will be the same for all images
		if (cudaMallocPitch(&d_image[i_image], &pitch, (size_t)(Ny*sizeof(float)), Nx))
			printf("Error in cudaMallocPitch!\n");
		printf("Nx=%d Ny=%d pitch=%d\n", Nx, Ny, (int)(pitch/sizeof(float)));
		
		// cudaDeviceSynchronize();  // Not necessary, as the prior cudaMalloc will block until cudaMemcpyAsync finishes
		// At this point, the prior image (i_image-1) has been copied to the GPU; we can process it
		
		// The memcopy command is asynchronous, and will run concurrently with the next host operation - 
		// reading the next image, and preprocessing it
		
		if (cudaMemcpy2DAsync(d_image[i_image], pitch, h_image, Ny*sizeof(float), Ny*sizeof(float), Nx, cudaMemcpyHostToDevice))
			printf("Error in cudaMemcpy!\n");
//		cudaDeviceSynchronize();
//		cudaError_t err = cudaGetLastError();
//		if (err != cudaSuccess) {
//			printf("CUDA Error: %s\n", cudaGetErrorString(err));
//}
		
		
	} // i_image loop

	ERR( cudaMemcpy(d_dt, h_dt, N_images*sizeof(double), cudaMemcpyHostToDevice) )


	free(buf0);
	ERR( cudaFreeHost(h_image) )
	ERR( cudaFree(d_img1) )
	ERR( cudaFree(d_img2) )

	return 0;
}