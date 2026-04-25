/* The main program in Asteroid_detector package. Searches for moving star-like objects
in the imaging sequence which was pre-processed with "reconv". Uses GPU acceleration (CUDA).

To compile:

nvcc -O3 asteroid_search.cu func.c func2.cu -lfftw3 -lcfitsio -lm -lz -o asteroid_search

*/

#include "reconv.h"
#include "asteroid_search.h"


int main(int argc, char **argv)
{
    struct timeval  tdr0, tdr1, tdr, tdr2, tdr3;
    double restime;
	int status = 0, naxis;
    long naxes[3];
	long Npix;
	int Nx, Ny, Nc, Nx0, Ny0;
	float *buf0; // host
	size_t pitch;
	char date_obs[30];
	double mjd, mjd0;
	int year, month, day, hour, minute;
    double second, exposure;	
	
	float FWHM = 7.7; // Needs to be provided via images' FITS headers; 7.7

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
	
	float **h_image = NULL;
	h_image = (float **)malloc(N_images*sizeof(float*));
	float **d_image = NULL;
	ERR( cudaMalloc(&d_image, N_images*sizeof(float*)) )

	float *h_image1 = NULL;
	float *h_dt = NULL;
//	ERR( cudaMallocHost(&h_dt, N_images*sizeof(float*)) )
	h_dt = (float *)malloc(N_images*sizeof(float*));
	float *d_dt = NULL;
	ERR( cudaMalloc(&d_dt, N_images*sizeof(float*)) )
	
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
		
#ifndef TEST
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
		printf("DATE-OBS: %s : %lf\n", date_obs, h_dt[i_image]);
#endif		
		
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
		
		gettimeofday (&tdr2, NULL);  		
		gauss_blur(i_image, N_images, Nx, Ny, buf0, buf0, sgm_Gauss);
		gettimeofday (&tdr3, NULL);  
		tdr = tdr2;
		timeval_subtract (&restime, &tdr3, &tdr);
		printf("Convolution time: %e\n", restime);
		
//		dump_fits(Nx, Ny, 1, buf0, "image.fits");
		
		if (i_image == 0)
//			ERR( cudaMallocHost(&h_image1, sizeof(float)*Npix) )
			h_image1 = (float *)malloc(sizeof(float)*Npix);
		
		rebin(i_image, buf0, &Nx, &Ny, &Npix, &h_image1);
		
		// x is for the rows, y is for columns
		// This is the host-device sync point
		if (cudaMallocPitch(&h_image[i_image], &pitch, (size_t)(Ny*sizeof(float)), Nx))
			printf("Error in cudaMallocPitch!\n");
		printf("Nx=%d Ny=%d pitch=%d\n", Nx, Ny, (int)(pitch/sizeof(float)));
				
		// The memcopy command is asynchronous, and will run concurrently with the next host operation - 
		// reading the next image, and preprocessing it		
		if (cudaMemcpy2DAsync(h_image[i_image], pitch, h_image1, Ny*sizeof(float), Ny*sizeof(float), Nx, cudaMemcpyHostToDevice))
			printf("Error in cudaMemcpy!\n");		
		
	} // i_image loop

//exit(0);

#ifndef TEST
	// Normalizing the time shift for the last image to 1:
	for (int i=0; i<N_images; i++)
		h_dt[i] = h_dt[i]/h_dt[N_images-1];
#else	
	for (int i=0; i<N_images; i++)
		h_dt[i] = (float)i / (float)(N_images - 1);
#endif	

	ERR( cudaMemcpy(d_dt, h_dt, N_images*sizeof(float), cudaMemcpyHostToDevice) )
	ERR( cudaMemcpy(d_image, h_image, N_images*sizeof(float*), cudaMemcpyHostToDevice) )
	
	// Number of base tiles along both axes (no incomplete tiles are allowed):
	int Nxb = Nx/NB;
	int Nyb = Ny/NB;
	// Base tile (Ixb,Iyb) covers this range of pixels (inclusive): ix=NB*Ixb..NB*(Ixb+1)-1, iy=NB*Iyb..NB*(Iyb+1)-1 
	
	// End of motion vectors are quantized on a mesh with this step in FWHM units:
	// The multipler Step should be provided as a command line argument
	float Step = 0.5;
	// The mesh step (Motion Quantum) in pixels:
	float MQ = Step * FWHM; 
	
	// The motion vector space is between radii RMIN and RMAX (both are in MQ units)
	// RMIN and RMAX should come as command line arguments
	int RMIN = 1;
	int RMAX = 500;
	
	// RMAX is limited by the image diagonal length (distance between the farthest tiles):
	int diag = floor(NB * sqrt(pow(Nxb-1,2) + pow(Nyb-1,2)) / MQ);
	if (RMAX > diag)
		RMAX = diag;
	if (RMAX < 0)
		RMAX = 0;
	if (RMIN < 0)
		RMIN = 0;
	
	// Minimum pixel brightness (in master image std units) to qualify for a detection
	// Should be input parameter:  25, 5.0
	float p_min_std = 1.5;
	
	// Master image std (should come from FITS headers, or command line) 0.0028, 3.722e-4
	float std_master = 3.722e-4;
	
	// Signal detection theshold for image stacks (assuming summation stacking):
	float p_min = p_min_std * std_master * N_images;

	float *d_test_image = NULL;
	ERR( cudaMallocPitch(&d_test_image, &pitch, (size_t)(Ny*sizeof(float)), Nx) )
	float *h_test_image = NULL;
//	ERR( cudaMallocHost(&h_test_image, (size_t)(Nx*Ny*sizeof(float))) )
	h_test_image = (float *)malloc(Npix*sizeof(float));
	struct List *h_list = NULL;
	h_list = (struct List *)malloc(MAX_PIXELS*sizeof(struct List));
	struct List *d_list = NULL;
	ERR (cudaMalloc(&d_list, MAX_PIXELS*sizeof(struct List)))
	
	unsigned int *d_Pixel_counter = NULL;
	ERR (cudaMalloc(&d_Pixel_counter, sizeof(unsigned int)))
	unsigned int h_Pixel_counter = 0;
	cudaMemcpy(d_Pixel_counter, &h_Pixel_counter, sizeof(unsigned int), cudaMemcpyHostToDevice);

	
	cudaDeviceSynchronize();
    gettimeofday (&tdr0, NULL);  
	
	int Jcount = 0;
	dim3 Grid_size;
	dim3 Block_size(32,32);
	int Ix1, Iy1;
	// Double loop going over all possible motion vectors:
	// Jx, Jy are in MQ units
	for (int Jx=-RMAX; Jx<=RMAX; Jx++)		
	{
		printf("Jx = %d\n", Jx);
		for (int Jy=-RMAX; Jy<=RMAX; Jy++)
		{
			// Length of the motion vector in MQ units:
			float R = sqrt(float(Jx*Jx + Jy*Jy));
			if (R >= RMIN && R <= RMAX)
			{
				find_kernel_parameters(Jx, Jy, MQ, Nx, Ny, &Grid_size, &Ix1, &Iy1);
								
				// The main compute kernel: searches for moving objects over all images,
				// all allowed base image tiles, for the given motion vector (Jx,Jy) in MQ units
				// Using the maximum 1024 threads per block, to cover 32x32 pixel tiles
				motion_search_cuda <<<Grid_size, Block_size>>> (d_image,N_images,pitch,Ix1,Iy1,Jx,Jy,MQ,p_min,d_dt,d_test_image,d_list,0,d_Pixel_counter,Nx,Ny);
				
				Jcount++;
			}			
		}
		}

	ERR( cudaDeviceSynchronize() )
    gettimeofday (&tdr1, NULL);  
    tdr = tdr0;
    timeval_subtract (&restime, &tdr1, &tdr);

			
	if (Jcount == 0)
		printf("No motion vectors in the given range of R=RMIN...RMAX\n");
	else
	{
		printf("Processed %d motion vectors\n", Jcount);
		printf ("time per motion vector: %e\n", restime/Jcount);
		
		ERR( cudaMemcpy(&h_Pixel_counter, d_Pixel_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost))
		cudaDeviceSynchronize();
		printf("Number of bright pixels: %d\n", h_Pixel_counter);
		if (h_Pixel_counter > 0)
		{
			ERR( cudaMemcpy(h_list, d_list, h_Pixel_counter*sizeof(struct List), cudaMemcpyDeviceToHost))
			cudaDeviceSynchronize();
			float pmax = -1e30;
			int imax = -1;
			for (int i=0; i<h_Pixel_counter; i++)
			{
				if (h_list[i].p > pmax)
				{
					pmax = h_list[i].p;
					imax = i;
				}
			}
			printf("Motion vector: (%d, %d); pixel coords: (%d, %d); p=%f (%f std)\n",
			h_list[imax].Jx, h_list[imax].Jy, h_list[imax].ix, h_list[imax].iy, h_list[imax].p,
			h_list[imax].p/(std_master*N_images));
			
			int *Cluster_index = (int *)malloc(h_Pixel_counter*sizeof(int));			
			cluster_analysis(h_list, h_Pixel_counter, Cluster_index);
			
			FILE *fp = fopen("list.dat", "w");
			for (int i=0; i<h_Pixel_counter; i++)
			{
				fprintf(fp, "%d %d %d %d %f %d\n", h_list[i].Jx, h_list[i].Jy, h_list[i].ix, h_list[i].iy,
				h_list[i].p/(std_master*N_images), Cluster_index[i]);
			}
			fclose(fp);
			free(Cluster_index);

			find_kernel_parameters(h_list[imax].Jx, h_list[imax].Jy, MQ, Nx, Ny, &Grid_size, &Ix1, &Iy1);

			motion_search_cuda <<<Grid_size, Block_size>>> (d_image,N_images,pitch,Ix1,Iy1,
			h_list[imax].Jx,h_list[imax].Jy,MQ,p_min,d_dt,d_test_image,d_list,1,d_Pixel_counter,Nx,Ny);

			ERR( cudaMemcpy2D(h_test_image, Ny*sizeof(float), d_test_image, pitch, Ny*sizeof(float), Nx, cudaMemcpyDeviceToHost) )
			cudaDeviceSynchronize();
			dump_fits(Nx, Ny, 1, h_test_image, "output.fits");
			
		}
	}

	free(buf0);	
//	ERR( cudaFreeHost(h_image) )
	free(h_image1);
	free(h_image);
//	ERR( cudaFreeHost(h_test_image) )
	free(h_test_image);

	return 0;
}