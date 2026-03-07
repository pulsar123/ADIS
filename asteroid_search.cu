/* The main program in Asteroid_detector package. Searches for moving star-like objects
in the imaging sequence which was pre-processed with "reconv". Uses GPU acceleration (CUDA).

To compile:

nvcc -O3 asteroid_search.cu func2.cu -lfftw3 -lcfitsio -lm -lz -o asteroid_search

*/

#include "asteroid_search.h"

int main(int argc, char **argv)
{
    struct timeval  tdr0, tdr1, tdr;
    double restime;
	int status = 0, naxis;
    long naxes[3];
	long Npix;
	int Nx, Ny, Nc;
	float *buf0;
	size_t pitch;

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
	
	
	// Reading input FITS images one by one, using cudaMemcpyAsync to do concurrent copying to the GPU
	for (int i_image=0; i_image<N_images; i_image++)
	{		
		// Reading the next input FITS file
		printf("\nReading file %s\n", argv[1+i_image]);
		fitsfile *f0;
		char *file0 = argv[1+i_image];
		fits_open_file(&f0, file0, READONLY, &status);
		fits_error(status);
		
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
			Nc = 3;
			Npix = (long)Nx * Ny;
			buf0 = (float *)malloc(sizeof(float) * Npix * Nc);
		}
		else
		{
			if (naxes[1] != Nx || naxes[0] != Ny)
			{
				fprintf(stderr, "Error: mismatching input images\n");
				return EXIT_FAILURE;
			}
		}
		
		fits_read_img(f0, TFLOAT, 1, Npix * Nc, NULL, buf0, NULL, &status);
		fits_error(status);
		fits_close_file(f0, &status);		
printf("1\n");				
		image_bw(buf0, Npix, Nc);  // Turn the image into black and white
printf("2\n");				
		
		crop_and_rebin(i_image, buf0, &Nx, &Ny, &Npix, &h_image);  // allocates h_image on first call

printf("3\n");				
		
		// x is for the rows, y is for columns
		// Assuming pitch will be the same for all images
		ERR( cudaMallocPitch(&d_image[i_image], &pitch, (size_t)(Ny*sizeof(float)), Nx) )
printf("4\n");				
		
		// cudaDeviceSynchronize();  // Not necessary, as the prior cudaMalloc will block until cudaMemcpyAsync finishes
		
		// The memcopy command is asynchronous, and will run concurrently with the next host operations - 
		// reading the next image, and preprocessing it
		ERR( cudaMemcpyAsync(d_image[i_image], h_image, Npix*sizeof(float), cudaMemcpyHostToDevice) )
		
	} // i_image loop

	free(buf0);
	ERR( cudaFreeHost(h_image) )

	return 0;
}