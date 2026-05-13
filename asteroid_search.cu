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

	// The motion vector space is between radii RMIN and RMAX (both are in MQ units)
	// RMIN and RMAX should come as command line arguments
	int RMIN = 100;
	int RMAX = 101;

	// Minimum pixel brightness (in master image std units) to qualify for a detection
	// Should be input parameter:  25, 5.0
	float p_min_std = 25;
	
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
	size_t free_mem, total_mem;
	cudaMemGetInfo(&free_mem, &total_mem);
	printf("GPU memory: Free: %f GB, Total: %f GB\n\n", (float)free_mem/(1024*1024*1024), (float)total_mem/(1024*1024*1024));		
	
	int N_images = argc - 1;
	
	float **h_image = NULL;
	h_image = (float **)malloc(N_images*sizeof(float*));
	float **d_image = NULL;
	ERR( cudaMalloc(&d_image, N_images*sizeof(float*)) )

	float *h_image1 = NULL;
	float *h_dt = NULL;
	
	#ifdef MALLOCHOST	
	ERR( cudaMallocHost(&h_dt, N_images*sizeof(float*)) )
	#else	
	h_dt = (float *)malloc(N_images*sizeof(float*));
	#endif
	
	float *d_dt = NULL;
	ERR( cudaMalloc(&d_dt, N_images*sizeof(float*)) )
	
	float sgm_Gauss = FWHM / sqrt(8*log(2.0));
	printf("sgm_Gauss = %f\n", sgm_Gauss);

	gettimeofday (&tdr2, NULL);  		
	
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

//		float crop_fraction = 1.0;
//		crop(buf0, &Nx, &Ny, &Npix, crop_fraction);

		// Background model has these many tiles along each dimension:
		int NTx = 5;
		int NTy = (int)((float)NTx / (float)Nx * (float)Ny + 0.5);
//		printf("Background model: %d x %d tiles, %d x %d pixels each\n", NTy, NTx, Ny/NTy, Nx/NTx);
		subtract_background(i_image, buf0, Nx, Ny, NTx, NTy);
		
		gauss_blur(i_image, N_images, Nx, Ny, buf0, buf0, sgm_Gauss);

//		dump_fits(Nx, Ny, 1, buf0, "image.fits");
		
		if (i_image == 0)
			#ifdef MALLOCHOST
			ERR( cudaMallocHost(&h_image1, sizeof(float)*Npix) )
			#else
			h_image1 = (float *)malloc(sizeof(float)*Npix);
			#endif
		
		rebin(i_image, buf0, &Nx, &Ny, &Npix, &h_image1);
		
		// x is for the rows, y is for columns
		// This is the host-device sync point
		if (cudaMallocPitch(&h_image[i_image], &pitch, (size_t)(Ny*sizeof(float)), Nx))
			printf("Error in cudaMallocPitch!\n");
		printf("Nx=%d, Ny=%d, pitch=%d, image size=%f GB\n", Nx, Ny, (int)(pitch/sizeof(float)),
		(float)(pitch*Nx)/(1024*1024*1024));
		
		if (i_image == 0)
		{
			size_t free_mem1, total_mem1;
			cudaMemGetInfo(&free_mem1, &total_mem1);
			if ((N_images-1)*pitch*Nx > free_mem1)
			{
				printf("\nNot enough of free GPU memory to fit all the images!\n");
				printf("Maximum %d images can be loaded\n", (int)((float)free_mem/(float)(pitch*Nx)));
				printf("Exiting\n\n");
				exit(1);
			}
		}
				
		// The memcopy command is asynchronous, and will run concurrently with the next host operation - 
		// reading the next image, and preprocessing it		
		if (cudaMemcpy2DAsync(h_image[i_image], pitch, h_image1, Ny*sizeof(float), Ny*sizeof(float), Nx, cudaMemcpyHostToDevice))
			printf("Error in cudaMemcpy!\n");		
		
	} // i_image loop

	gettimeofday (&tdr3, NULL);  
	tdr = tdr2;
	timeval_subtract (&restime, &tdr3, &tdr);
	printf("\nImage processing time, per image: %e seconds\n\n", restime/N_images);
		

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
	
	
	// RMAX is limited by the image diagonal length (distance between the farthest tiles):
	int diag = floor(NB * sqrt(pow(Nxb-1,2) + pow(Nyb-1,2)) / MQ);
	if (RMAX > diag)
		RMAX = diag;
	if (RMAX < 0)
		RMAX = 0;
	if (RMIN < 0)
		RMIN = 0;
	
	
	// Master image std (should come from FITS headers, or command line) 0.0028, 3.722e-4
//	float std_master = 3.722e-4;
	
	// Signal detection theshold for image stacks (assuming summation stacking):
//	float p_min = p_min_std * std_master * N_images;

	float *d_test_image = NULL;
	ERR( cudaMallocPitch(&d_test_image, &pitch, (size_t)(Ny*sizeof(float)), Nx) )
	float *h_test_image = NULL;

	#ifdef MALLOCHOST
	ERR( cudaMallocHost(&h_test_image, (size_t)(Nx*Ny*sizeof(float))) )
	#else
	h_test_image = (float *)malloc(Npix*sizeof(float));
	#endif

	unsigned int *d_Pixel_counter = NULL;
	ERR (cudaMalloc(&d_Pixel_counter, sizeof(unsigned int)))
	unsigned int h_Pixel_counter = 0;
	cudaMemcpy(d_Pixel_counter, &h_Pixel_counter, sizeof(unsigned int), cudaMemcpyHostToDevice);
	
	int *h_ix;
	cudaMallocHost(&h_ix, MAX_PIXELS*sizeof(int));
	int *h_iy;
	cudaMallocHost(&h_iy, MAX_PIXELS*sizeof(int));
	int *h_Jx;
	cudaMallocHost(&h_Jx, MAX_PIXELS*sizeof(int));
	int *h_Jy;
	cudaMallocHost(&h_Jy, MAX_PIXELS*sizeof(int));
	float *h_p;
	cudaMallocHost(&h_p, MAX_PIXELS*sizeof(float));
	int *d_ix;
	cudaMalloc(&d_ix, MAX_PIXELS*sizeof(int));
	int *d_iy;
	cudaMalloc(&d_iy, MAX_PIXELS*sizeof(int));
	int *d_Jx;
	cudaMalloc(&d_Jx, MAX_PIXELS*sizeof(int));
	int *d_Jy;
	cudaMalloc(&d_Jy, MAX_PIXELS*sizeof(int));
	float *d_p;
	cudaMalloc(&d_p, MAX_PIXELS*sizeof(float));
	
	cudaMemGetInfo(&free_mem, &total_mem);
	printf("GPU memory: Free: %f GB, Total: %f GB\n\n", (float)free_mem/(1024*1024*1024), (float)total_mem/(1024*1024*1024));
	
	
	// Computing the value of sgm for a zero-shift stack
	dim3 Block_size(32,32);
	dim3 Grid_size;
	int Ix1, Iy1;
	find_kernel_parameters(0, 0, MQ, Nx, Ny, &Grid_size, &Ix1, &Iy1);
	float p_min = 1e30;
	// Zero-offset stacking:
	motion_search_cuda <<<Grid_size, Block_size>>> (d_image,N_images,pitch,Ix1,Iy1,0,0,MQ,p_min,d_dt,d_test_image,d_ix,d_iy,d_Jx,d_Jy,d_p,1,d_Pixel_counter,Nx,Ny);
	
	
	ERR( cudaMemcpy2D(h_test_image, Ny*sizeof(float), d_test_image, pitch, Ny*sizeof(float), Nx, cudaMemcpyDeviceToHost) )
		
	double p0, sgm;
	long Npix2;
	int kk;
	sigma_clipping(h_test_image, Npix, 3.0, &p0, &sgm, &Npix2, &kk);
	dump_fits(Nx, Ny, 1, h_test_image, "zero_shift.fits");
	printf("\n Zero shift stack: p=%e, sgm=%e\n\n", p0, sgm);
	long *hist = (long *)malloc(sizeof(long)*(BIN_MAX-BIN_MIN+1));
	compute_histogram(h_test_image, Npix, sgm, &p_min_std, hist);
	printf("(0,0) vector: p_min_std=%f\n", p_min_std);


	find_kernel_parameters(RMIN, RMIN, MQ, Nx, Ny, &Grid_size, &Ix1, &Iy1);
	p_min = 1e30;
	motion_search_cuda <<<Grid_size, Block_size>>> (d_image,N_images,pitch,Ix1,Iy1,RMIN,RMIN,MQ,p_min,d_dt,d_test_image,d_ix,d_iy,d_Jx,d_Jy,d_p,1,d_Pixel_counter,Nx,Ny);
	ERR( cudaMemcpy2D(h_test_image, Ny*sizeof(float), d_test_image, pitch, Ny*sizeof(float), Nx, cudaMemcpyDeviceToHost) )
	sigma_clipping(h_test_image, Npix, 3.0, &p0, &sgm, &Npix2, &kk);
	printf("\n Rmin shift stack: p=%e, sgm=%e\n\n", p0, sgm);
	compute_histogram(h_test_image, Npix, sgm, &p_min_std, hist);
	printf("(Rmin,Rmin) vector: p_min_std=%f\n", p_min_std);

	find_kernel_parameters(RMAX, RMAX, MQ, Nx, Ny, &Grid_size, &Ix1, &Iy1);
	p_min = 1e30;
	motion_search_cuda <<<Grid_size, Block_size>>> (d_image,N_images,pitch,Ix1,Iy1,RMAX,RMAX,MQ,p_min,d_dt,d_test_image,d_ix,d_iy,d_Jx,d_Jy,d_p,1,d_Pixel_counter,Nx,Ny);
	ERR( cudaMemcpy2D(h_test_image, Ny*sizeof(float), d_test_image, pitch, Ny*sizeof(float), Nx, cudaMemcpyDeviceToHost) )
	sigma_clipping(h_test_image, Npix, 3.0, &p0, &sgm, &Npix2, &kk);
	printf("\n Rmax shift stack: p=%e, sgm=%e\n\n", p0, sgm);
	compute_histogram(h_test_image, Npix, sgm, &p_min_std, hist);
	printf("(Rmax,Rmax) vector: p_min_std=%f\n", p_min_std);

	free(hist);
//	exit(0);

//	p_min = p_min_std * sgm;
	p_min = 3.722e-4 * 2.5;


	cudaDeviceSynchronize();
    gettimeofday (&tdr0, NULL);  
		
//--------------  The longest part of the code: image stacking for different motion vectors on GPU -----	
	int Jcount = 0;
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
				motion_search_cuda <<<Grid_size, Block_size>>> (d_image,N_images,pitch,Ix1,Iy1,Jx,Jy,MQ,p_min,d_dt,d_test_image,d_ix,d_iy,d_Jx,d_Jy,d_p,0,d_Pixel_counter,Nx,Ny);
				
				Jcount++;
			}			
		}
		}		
//-------------------------------------------------------------------------------------------------------		
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
			cudaMemcpy(h_ix, d_ix, h_Pixel_counter*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_iy, d_iy, h_Pixel_counter*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_Jx, d_Jx, h_Pixel_counter*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_Jy, d_Jy, h_Pixel_counter*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_p, d_p, h_Pixel_counter*sizeof(float), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();

			float pmax = -1e30;
			int imax = -1;
			for (int i=0; i<h_Pixel_counter; i++)
			{
				if (h_p[i] > pmax)
				{
					pmax = h_p[i];
					imax = i;
				}
			}
			printf("Motion vector: (%d, %d); pixel coords: (%d, %d); p=%f (%f std), i=%d\n",
			h_Jx[imax], h_Jy[imax], h_ix[imax], h_iy[imax], h_p[imax],
			h_p[imax]/sgm, imax);
			
			int *Cluster_index = (int *)malloc(h_Pixel_counter*sizeof(int));			
			
			cudaDeviceSynchronize();
			gettimeofday (&tdr0, NULL);  
//			cluster_analysis(h_ix,h_iy,h_Jx,h_Jy,h_p, h_Pixel_counter, Cluster_index);
			cudaDeviceSynchronize();
			gettimeofday (&tdr1, NULL);  			
			cluster_analysis_cuda(d_ix,d_iy,d_Jx,d_Jy,d_p, h_Pixel_counter, Cluster_index);
			cudaDeviceSynchronize();
			gettimeofday (&tdr2, NULL);  

			tdr = tdr0;
			timeval_subtract (&restime, &tdr1, &tdr);
			printf("Cluster analysis CPU: %e s\n", restime);
			tdr = tdr1;
			timeval_subtract (&restime, &tdr2, &tdr);
			printf("Cluster analysis GPU: %e s\n", restime);
			

			FILE *fp = fopen("list.dat", "w");
			for (int i=0; i<h_Pixel_counter; i++)
			{
				fprintf(fp, "%d %d %d %d %f %d\n", h_Jx[i], h_Jy[i], h_ix[i], h_iy[i],
				h_p[i]/sgm, Cluster_index[i]);
			}
			fclose(fp);
			free(Cluster_index);

			find_kernel_parameters(h_Jx[imax], h_Jy[imax], MQ, Nx, Ny, &Grid_size, &Ix1, &Iy1);

			motion_search_cuda <<<Grid_size, Block_size>>> (d_image,N_images,pitch,Ix1,Iy1,
			h_Jx[imax],h_Jy[imax],MQ,p_min,d_dt,d_test_image,d_ix,d_iy,d_Jx,d_Jy,d_p,1,d_Pixel_counter,Nx,Ny);

			ERR( cudaMemcpy2D(h_test_image, Ny*sizeof(float), d_test_image, pitch, Ny*sizeof(float), Nx, cudaMemcpyDeviceToHost) )
			cudaDeviceSynchronize();
			dump_fits(Nx, Ny, 1, h_test_image, "output.fits");
			
		}
	}

	#ifdef MALLOCHOST
	ERR( cudaFreeHost(h_image1) )
	ERR( cudaFreeHost(h_dt) )
	ERR( cudaFreeHost(h_test_image) )
	cudaFree(h_ix);
	cudaFree(h_iy);
	cudaFree(h_Jx);
	cudaFree(h_Jy);
	#else
	free(h_image1);
	free(h_dt);
	free(h_test_image);
	#endif

	free(h_image);
	free(buf0);	
	cudaFree(d_ix);
	cudaFree(d_iy);
	cudaFree(d_Jx);
	cudaFree(d_Jy);

	return 0;
}