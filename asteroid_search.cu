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
	
	float FWHM = 5.0; // Needs to be provided via images' FITS headers

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
		
		// Computing time offsets from the first image:
		fits_read_key(f0, TSTRING, "DATE-OBS", date_obs, NULL, &status);
		fits_read_key(f0, TDOUBLE, "EXPTIME", &exposure, NULL, &status);
		fits_str2time(date_obs, &year, &month, &day,
                  &hour, &minute, &second, &status);
		if (status) { fits_report_error(stderr, status); return status; }
		mjd = exposure/2.0/86400 + date2mjd (year, month, day) + ((second/60.0+minute)/60.0+hour)/24.0; // Modified Julian Date (UT) of the middle of exposure
		if (i_image == 0)
			mjd0 = mjd;
		h_dt[i_image] = mjd - mjd0; //  in days
//		printf("DATE-OBS: %s : %lf\n", date_obs, h_dt[i_image]);

		
		fits_get_img_dim(f0, &naxis, &status);
		fits_get_img_size(f0, 3, naxes, &status);
		fits_error(status);
		
		if (naxis != 3 || naxes[2] != 3) 
		{
			fprintf(stderr, "Error: expected RGB FITS image\n");
			return EXIT_FAILURE;
		}
	
		Nx = naxes[1];
		Ny = naxes[0];
		if (i_image == 0)
		{
			Nx0 = Nx;
			Ny0 = Ny;
			Nc = 3;
			Npix = (long)Nx * Ny;
			buf0 = (float *)malloc(sizeof(float) * Npix * Nc);
		}
		else
		{
			if (Nx != Nx0 || Ny != Ny0)
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
		
		// x is for the rows, y is for columns
		// This is the host-device sync point
		if (cudaMallocPitch(&d_image[i_image], &pitch, (size_t)(Ny*sizeof(float)), Nx))
			printf("Error in cudaMallocPitch!\n");
		printf("Nx=%d Ny=%d pitch=%d\n", Nx, Ny, (int)(pitch/sizeof(float)));
				
		// The memcopy command is asynchronous, and will run concurrently with the next host operation - 
		// reading the next image, and preprocessing it		
		if (cudaMemcpy2DAsync(d_image[i_image], pitch, h_image, Ny*sizeof(float), Ny*sizeof(float), Nx, cudaMemcpyHostToDevice))
			printf("Error in cudaMemcpy!\n");		
		
	} // i_image loop

	// Normalizing the time shift for the last image to 1:
	for (int i=0; i<N_images; i++)
		h_dt[i] = h_dt[i]/h_dt[N_images-1];
	ERR( cudaMemcpy(d_dt, h_dt, N_images*sizeof(double), cudaMemcpyHostToDevice) )
	
	// Base tiles have NBxNB pixels size:
	int NB = 31;
	// Number of base tiles along both axes (no incomplete tiles are allowed):
	int Nxb = Nx/NB;
	int Nyb = Ny/NB;
	// Base tile (Ixb,Iyb) covers this range of pixels (inclusive): ix=NB*Ixb..NB*(Ixb+1)-1, iy=NB*Iyb..NB*(Iyb+1)-1 
	
	// End of motion vectors are quantized using this step in FWHM units:
	float Step = 0.5;
	// The step (Motion Quantum) in pixels:
	float MQ = Step * FWHM; // The multipler Step should be provided as a command line argument
	
	// The motion vector space is between radii RMIN and RMAX (both are in MQ units)
	// RMIN and RMX should come as command line arguments
	int RMIN = 10;
	int RMAX = 20;
	// RMAX is limited by the image diagonal length (distance between the farthest tiles):
	int diag = floor(NB * sqrt(pow(Nxb-1,2) + pow(Nyb-1,2)) / MQ);
	if (RMAX > diag)
		RMAX = diag;
	
	// Minimum pixel brightness (in global std units) to qualify for a detection
	// Should be input parameter:
	float p_min = 5.0;
	
	// Double loop going over all possible motion vectors:
	// Jx, Jy are in MQ units
	for (int Jx=-RMAX; Jx>=RMAX; Jx++)
		for (int Jy=-RMAX; Jy>=RMAX; Jy++)
		{
			// Length of the motion vector in MQ units:
			float R = sqrt(float(Jx*Jx + Jy*Jy));
			if (R > RMIN && R <= RMAX)
			{
				// Motion vector in pixels:
				float jx = Jx*MQ;
				float jy = Jy*MQ;
				
				// Maximum range of base image pixels usable for this motion vector:
				// ix_min...ix_max, iy_min...iy_max (inclusive)
				int ix_min = 0;
				int ix_max = Nx - 1;
				int iy_min = 0;
				int iy_max = Ny - 1;
				if (jx < 0)
					ix_min = ceil(-jx);
				else
					ix_max = ceil(jx);
				if (jy < 0)
					iy_min = ceil(-jy);
				else
					iy_max = ceil(jy);
				
				// Now we can get the ranges for the base tile indexes (inclusive):
				int Ix1 = ix_min / NB;
				int Ix2 = (ix_max-NB+1) / NB;
				int Iy1 = iy_min / NB;
				int Iy2 = (iy_max-NB+1) / NB;
				
				// The grid of blocks is for Ix and Iy parameters:
				dim3 Grid_size(Ix2-Ix1+1, Iy2-Iy1+1);
				
				// The main compute kernel: searches for moving objects over all images,
				// all allowed base image tiles, for the given motion vector (Jx,Jy) in MQ units
				// Using the maximum 1024 threads per block, to cover 32x32 pixel tiles
				motion_search_cuda <<<Grid_size, 1024>>> (d_image,N_images,Ix1,Iy1,Jx,Jy,MQ,p_min);								
			}			
		}


	free(buf0);
	ERR( cudaFreeHost(h_image) )
	ERR( cudaFree(d_img1) )
	ERR( cudaFree(d_img2) )

	return 0;
}