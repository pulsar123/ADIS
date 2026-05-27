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
    long naxes[2];
	long Npix, Npix_ini;
	int Nx, Ny, Nx0, Ny0, Nx_ini, Ny_ini;
	float *buf0; // host
	size_t pitch;
	char date_obs[30];
	double mjd, mjd0;
	int year, month, day, hour, minute;
    double second, exposure;	
	float p_min;
	char name0[100];


	// Processing command line arguments
	
	if (argc == 1)
	{
		printf("\n Syntax (any order):\n\n");
		printf(" %s  -RMAX value  -Step value  image1 image2 ... \n\n", argv[0]);
		printf("\n Obligatory arguments:\n\n");
		printf(" -RMAX value        : longest motion vector (in FWHM units)\n");
		printf(" -Step value        : the mesh step for motion vector lengths (in FWHM units)\n");
		printf(" image1  ...        : list of B&W input FITS images (preprocessed with reconv)\n");
		printf("\n Optional arguments [default value]:\n\n");
		printf(" -finetune x y Jx Jy: do finetuning search for one cluster described by the 4D coords (pixels)\n");
//		printf(" -list file         : instead of motion object detection, use the list generated in a prior detection run\n");
		printf(" -N_cloud value     : maximum number of cloud fits files to save [10]\n");
		printf(" -N_noise value     : parameter used during the histogram step (in std units) [10]\n");
		printf(" -rebin value       : image rebinning factor for x and y, in pixels [1]\n");
		printf(" -RMIN value        : shortest motion vector (in FWHM units) [1]\n");
		printf("\n In the finetune mode, the meaning of RMIN and RMAX changes:\n");
		printf(" RMIN: ignored\n");
		printf(" RMAX: largest distance from the cloud in (Jx,Jy) coordinates, using FWHM units\n");
		printf("\n");		
		exit(0);
	}

	float RMINf = 1;
	float RMAXf = 10;
	float Step = 0.5;
	int N_noise = 10;
	int N_cloud_max = 10;
	int Center_pix_ini[4];
	int Center_pix[4];
	int finetune = 0;
	int rebin = 1;
	
	int N_obligatory = 2; // Number of obligatory arguments
	int j = 1; // j counts all arguments
	int j0 = 1; // j0 counts only known arguments
	int error = 0;
	int iob = 0;
	while (j<argc)
	{		
		// Obligatory switches:
		if (strcmp(argv[j],"-RMAX") == 0)
		{
			j++;
			j0 = j0 + 2;
			if (argv[j][0] == '-' || j>=argc)
			{
				error = j-1;
				break;
			}
			RMAXf = atof(argv[j]);
			assert(RMAXf > 0);
			iob++;
		}
		
		if (strcmp(argv[j],"-Step") == 0)
		{
			j++;
			j0 = j0 + 2;
			if (argv[j][0] == '-' || j>=argc)
			{
				error = j-1;
				break;
			}
			Step = atof(argv[j]);
			assert(Step > 0);
			iob++;
		}		
		
		// Optional switches:
		
		if (strcmp(argv[j],"-finetune") == 0)
		{
			j0 = j0 + 5;
			finetune = 1;
			for (int i=0; i<4; i++)
			{
				j++;
				if (j>=argc)
				{
					error = j-1;
					break;
				}
				Center_pix_ini[i] = atoi(argv[j]);
			}
		}		

		if (strcmp(argv[j],"-N_cloud") == 0)
		{
			j++;
			j0 = j0 + 2;
			if (argv[j][0] == '-' || j>=argc)
			{
				error = j-1;
				break;
			}
			N_cloud_max = atoi(argv[j]);
			assert(N_cloud_max > 0);
		}		

		if (strcmp(argv[j],"-N_noise") == 0)
		{
			j++;
			j0 = j0 + 2;
			if (argv[j][0] == '-' || j>=argc)
			{
				error = j-1;
				break;
			}
			N_noise = atoi(argv[j]);
			assert(N_noise > 0);
		}		

		if (strcmp(argv[j],"-rebin") == 0)
		{
			j++;
			j0 = j0 + 2;
			if (argv[j][0] == '-' || j>=argc)
			{
				error = j-1;
				break;
			}
			rebin = atoi(argv[j]);
			assert(rebin > 0);
		}
		
		if (strcmp(argv[j],"-RMIN") == 0)
		{
			j++;
			j0 = j0 + 2;
			if (argv[j][0] == '-' || j>=argc)
			{
				error = j-1;
				break;
			}
			RMINf = atof(argv[j]);
			assert(RMINf > 0);
		}
		
		j++;
	}

	if (error)
	{
		printf("\nWrong argument: %s\n\n", argv[error]);
		exit(1);
	}
	if (iob < N_obligatory)
	{
		printf("\nSome obligatory arguments are missing!\n\n");
		exit(1);		
	}			
	if (j0 > argc - 2)
	{
		printf("\nThere should be at least 2 images in the stack. Exiting\n\n");
		exit(1);
	}
	
	// Now j0 points to the first image name argument

	// Switching from FWHM to MQ=FWHM*Step units
	int RMIN = (int)(RMINf/Step + 0.5);
	int RMAX = (int)(RMAXf/Step + 0.5);
	
	// Offsets (used when converting rebinned image pixel coordinates to non-rebin pixels image)
	int d_rebin = (rebin - 1) / 2;	
	
	// Switching to rebinned pixel coordinates:
	for (int j=0; j<4; j++)
	{
		if (j < 2)
			// For ix, iy coordinates we need both recenter and scale:
			Center_pix[j] = (Center_pix_ini[j]-d_rebin) / rebin;
		else
			// For Jx, Jy coordinates (which are differential by nature), we only need to rescale:
			Center_pix[j] = Center_pix_ini[j] / rebin;
	}
	
	Is_GPU_present();
	size_t free_mem, total_mem;
	cudaMemGetInfo(&free_mem, &total_mem);
	printf("GPU memory: Free: %f GB, Total: %f GB\n\n", (float)free_mem/(1024*1024*1024), (float)total_mem/(1024*1024*1024));		
	
	int N_images = argc - j0;

	List h_list;
	cudaMallocHost(&h_list.ix, MAX_PIXELS*sizeof(int));
	cudaMallocHost(&h_list.iy, MAX_PIXELS*sizeof(int));
	cudaMallocHost(&h_list.Jx, MAX_PIXELS*sizeof(int));
	cudaMallocHost(&h_list.Jy, MAX_PIXELS*sizeof(int));
	cudaMallocHost(&h_list.p, MAX_PIXELS*sizeof(float));
	List d_list;
	cudaMalloc(&d_list.ix, MAX_PIXELS*sizeof(int));
	cudaMalloc(&d_list.iy, MAX_PIXELS*sizeof(int));
	cudaMalloc(&d_list.Jx, MAX_PIXELS*sizeof(int));
	cudaMalloc(&d_list.Jy, MAX_PIXELS*sizeof(int));
	cudaMalloc(&d_list.p, MAX_PIXELS*sizeof(float));	
	
	float **h_image = NULL;
	h_image = (float **)malloc(N_images*sizeof(float*));
	float **d_image = NULL;
	ERR( cudaMalloc(&d_image, N_images*sizeof(float*)) )

	float *h_image1 = NULL;
	float *h_dt = NULL;
	
	ERR( cudaMallocHost(&h_dt, N_images*sizeof(float*)) )
	
	float *d_dt = NULL;
	ERR( cudaMalloc(&d_dt, N_images*sizeof(float*)) )
	
	gettimeofday (&tdr2, NULL); 

	double bias = 0.0;
	float FWHM;
	printf("\n");
	
	// Reading input FITS images one by one, using cudaMemcpyAsync to do concurrent copying to the GPU
	for (int i_image=0; i_image<N_images; i_image++)
	{		
		if (i_image == 0)
			sprintf(name0,"%s",argv[j0]);
		// Reading the next input FITS file
		fitsfile *f0;
		char *file0 = argv[j0+i_image];
		fits_open_file(&f0, file0, READONLY, &status);
		fits_error(status);
		
		if (i_image == 0)
		{
			// FWHM is read from the first input image, it is for the original (not-rebinned) image, in pixels
			fits_read_key(f0, TFLOAT, "FWHM", &FWHM, NULL, &status);
			fits_read_key(f0, TDOUBLE, "BIAS", &bias, NULL, &status);
		}
		
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
//		printf("DATE-OBS: %s : %lf\n", date_obs, h_dt[i_image]);
#endif		
		
		fits_get_img_dim(f0, &naxis, &status);
		fits_get_img_size(f0, naxis, naxes, &status);
		fits_error(status);
		
		if (naxis != 2) 
		{
			fprintf(stderr, "Error: expected B&W FITS image\n");
			return EXIT_FAILURE;
		}
	
		// Original image size (before rebinning):
		Nx_ini = naxes[1];
		Ny_ini = naxes[0];
		// Size after rebinning (including incomplete bins at the end):
		Nx = (Nx_ini+rebin-1)/rebin;
		Ny = (Ny_ini+rebin-1)/rebin;
		Npix_ini = (long)Nx_ini * Ny_ini;
		Npix = (long)Nx * Ny;
		if (i_image == 0)
		{
			Nx0 = Nx;
			Ny0 = Ny;
			buf0 = (float *)malloc(sizeof(float) * Npix_ini);
			ERR( cudaMallocHost(&h_image1, sizeof(float)*Npix) )
		}
		else
		{
			if (Nx != Nx0 || Ny != Ny0)
			{
				fprintf(stderr, "\nError: mismatching input images\n");
				exit(1);
			}
		}
		
		fits_read_img(f0, TFLOAT, 1, Npix_ini, NULL, buf0, NULL, &status);
		fits_error(status);
		fits_close_file(f0, &status);		
		
		// Subtracting bias from each image, and optional rebinning:
		rebinning(buf0, Nx_ini, Ny_ini, Nx, Ny, bias, rebin, h_image1);
		
		// x is for the rows, y is for columns
		// This is the host-device sync point
		if (cudaMallocPitch(&h_image[i_image], &pitch, (size_t)(Ny*sizeof(float)), Nx))
			printf("Error in cudaMallocPitch!\n");

		cudaMemGetInfo(&free_mem, &total_mem);

		printf("Reading file %s; ", argv[j0+i_image]);
		printf("Nx=%d, Ny=%d, free GPU memory=%f GB\n", Nx, Ny, (float)(free_mem)/(1024*1024*1024));
		
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
		
	// The mesh step (Motion Quantum) in rebinned pixels:
	float MQ = Step * FWHM/rebin; 

	float R0;
	if (finetune)
	{
		// Central radius in finetune mode (in MQ units):
		R0 = sqrt(pow(Center_pix[2],2) + pow(Center_pix[3]/rebin,2)) / MQ;
	}
	
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
	
	
	if (finetune == 0)
	{
		// RMAX is limited by the image diagonal length (distance between the farthest tiles):
		int diag = floor(NB * sqrt(pow(Nxb-1,2) + pow(Nyb-1,2)) / MQ);
		if (RMAX > diag)
		{
			RMAX = diag;
			printf("\nLimiting the search to RMAX=%f in FWHM units\n", RMAX*Step);
		}
		if (RMAX < 0)
			RMAX = 0;
		if (RMIN < 0)
			RMIN = 0;
	}
	
	float *d_test_image = NULL;
	ERR( cudaMallocPitch(&d_test_image, &pitch, (size_t)(Ny*sizeof(float)), Nx) )
	float *h_test_image = NULL;

	ERR( cudaMallocHost(&h_test_image, (size_t)(Nx*Ny*sizeof(float))) )

	unsigned int *d_Pixel_counter = NULL;
	ERR (cudaMalloc(&d_Pixel_counter, sizeof(unsigned int)))
	unsigned int h_Pixel_counter = 0;
	cudaMemcpy(d_Pixel_counter, &h_Pixel_counter, sizeof(unsigned int), cudaMemcpyHostToDevice);
	
	cudaMemGetInfo(&free_mem, &total_mem);
//	printf("GPU memory: Free: %f GB, Total: %f GB\n\n", (float)free_mem/(1024*1024*1024), (float)total_mem/(1024*1024*1024));

	dim3 Block_size(32,32);
	// Used for image erase
	dim3 Grid_size2 = {(Nx+Block_size.x-1)/Block_size.x, (Ny+Block_size.y-1)/Block_size.y};	
	dim3 Grid_size;
	
	// Computing the value of sgm for a zero-shift stack
	int Ix1, Iy1;
	find_kernel_parameters(0, 0, MQ, Nx, Ny, &Grid_size, &Ix1, &Iy1);
	p_min = 1e30;
	erase_image <<<Grid_size2, Block_size>>> (d_test_image, pitch, Nx, Ny, 0.0);
	// Zero-offset stacking:
	motion_search_cuda <<<Grid_size, Block_size>>> (d_image,N_images,pitch,Ix1,Iy1,0,0,MQ,p_min,d_dt,d_test_image,d_list,1,d_Pixel_counter,Nx,Ny);
	ERR( cudaMemcpy2D(h_test_image, Ny*sizeof(float), d_test_image, pitch, Ny*sizeof(float), Nx, cudaMemcpyDeviceToHost) )		
	double p0, sgm;
	long Npix2;
	int kk;
	sigma_clipping(h_test_image, Npix, 3.0, &p0, &sgm, &Npix2, &kk);
	dump_fits(Nx, Ny, 1, h_test_image, "zero_shift.fit");
	printf("\nZero shift stack: p=%e, sgm=%e\n\n", p0, sgm);
	
//++++++++++++++++++++++++++   Histogram step +++++++++++++++++++++++++++++++

	printf("\nProcessing %d motion vectors to compute the histogram\n", NVECTORS);
	
	// Testing NVECTORS motion_search vectors in the RMIN..RMAX range, computing the cumulative histogram of bright pixels,
	// which we use to place p_min threshold
	float delta_p = 3.0; // Initial offset when computing the histogram, in sgm units
	p_min = p0 + delta_p*sgm;  // Initial guess for p_min
	h_Pixel_counter = 0;
	cudaMemcpy(d_Pixel_counter, &h_Pixel_counter, sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
    gettimeofday (&tdr0, NULL);  
	float R, phi;
	int Jx, Jy;
	int iv = 0;
	int iv0 = 0;
	float R1, R2;
	if (finetune)
	{
		// In finetune mode, we compute the histogram from the range of the motion vectors
		// centered around the cloud center motion vector R0:
		R1 = R0 - RMAX;
		R2 = R0 + RMAX;
	}
	else
	{
		R1 = RMIN;
		R2 = RMAX;
	}
	
	do
	{
		iv0++;
		
		R = (R2-R1)*(float)rand() / (float)RAND_MAX + R1; // Random in R1..R2 range
		phi = 2*PI * (float)rand() / (float)RAND_MAX; // Random in 0..2*Pi range
		Jx = (int)(R * cos(phi) + 0.5);  // Random Jx
		Jy = (int)(R * sin(phi) + 0.5);  // Random Jy
		if (finetune)
		{
			// Distance from the cloud center (MQ units):
			float dR = sqrt(pow(Jx-Center_pix[2]/MQ,2) + pow(Jy-Center_pix[3]/MQ,2));
			// In finetuning mode, we exclude the neighbourhood of the cluster when computing the histogram:
			if (dR < RMAX)
				continue;
		}
		
		find_kernel_parameters(Jx, Jy, MQ, Nx, Ny, &Grid_size, &Ix1, &Iy1);
		motion_search_cuda <<<Grid_size, Block_size>>> (d_image,N_images,pitch,Ix1,Iy1,Jx,Jy,MQ,p_min,
		   d_dt,d_test_image,d_list,0,d_Pixel_counter,Nx,Ny);		
		
		iv++;
	}
	while (iv < NVECTORS && iv0 < 10*NVECTORS);
	
	if (iv == 0)
	{
		printf("\nSomething wrong in histogram computation!\n");
		if (finetune)
			printf("Likely RMAX is too large, in finetune mode\n");
		exit(1);
	}
	
	ERR( cudaDeviceSynchronize() )
    gettimeofday (&tdr1, NULL);  
    tdr = tdr0;
    timeval_subtract (&restime, &tdr1, &tdr);
	float dt_motion_search = restime / NVECTORS; // time per one motion_search_cude run, in seconds

	ERR( cudaMemcpy(&h_Pixel_counter, d_Pixel_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost))
	cudaDeviceSynchronize();
	
	if (h_Pixel_counter == 0)
	{
		// Decrease the initial guess for p_min above
		printf("\n No pixels detected during histogram calculation!\n\n");
		exit(1);
	}

	int *h_hist = (int *)malloc(NBIN * sizeof(int));
	int *d_hist = NULL;
	cudaMalloc(&d_hist, NBIN * sizeof(int));		
	cudaMemset(d_hist, 0, NBIN * sizeof(int));
	float del_sgm = sgm * DEL_SGM;
	int BS = 256;
	int NBlocks = (h_Pixel_counter + BS - 1) / BS;
	compute_histogram <<<NBlocks, BS>>> (d_list, h_Pixel_counter, p_min, del_sgm, d_hist);
	ERR( cudaMemcpy(h_hist, d_hist, NBIN*sizeof(int), cudaMemcpyDeviceToHost))

	if (h_hist[NBIN-1] > N_noise)
	{
		printf("\nToo much noise in the last bin!\n");
		printf("Increase N_noise, increase NBIN, or increase DEL_SGM.\n");
		printf("This might be a sign of a bright moving object in the images.\n\n");
		exit(1);
	}

	int pix_sum = 0;
	int bin = 0;
	for (bin=NBIN-1; bin>=0; bin--)
	{
		pix_sum = pix_sum + h_hist[bin];
		if (pix_sum > N_noise)
			break;
	}

	// New value of p_min based on the analysis of the histogram:
	// It is such that there are no more than N_noise detected pixels cumulatively
	// for NVECTORS random motion vectors.
	p_min = p_min + del_sgm*(bin+1);

	printf("\nHistogram: p_min=%f, offset=%f std\n\n", p_min, delta_p+(bin+1)*DEL_SGM);
	
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++	

	h_Pixel_counter = 0;
	cudaMemcpy(d_Pixel_counter, &h_Pixel_counter, sizeof(unsigned int), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
    gettimeofday (&tdr0, NULL);  
		
//--------------  The longest part of the code: image stacking for different motion vectors on GPU -----	

	// Estimating how long the motion search will take:
	int NMS = PI*(RMAX*RMAX-RMIN*RMIN); // Estimated number of motion vectors
	float dt_estimate = NMS * dt_motion_search;
	printf("\nEstimated search time: %f seconds; N = %d\n\n", dt_estimate, NMS);

	printf("\n\n === Motion search ===\n\n");

	int Jx0 = 0;
	int Jy0 = 0;
	// In the finetune mode, we are searching in the vicinity of the cloud
	if (finetune)
	{
		Jx0 = (int)(Center_pix[2]/MQ + 0.5);
		Jy0 = (int)(Center_pix[3]/MQ + 0.5);
	}

	int Jcount = 0;
	// Double loop going over all possible motion vectors:
	// Jx, Jy are in MQ units
	for (int Jx=Jx0-RMAX; Jx<=Jx0+RMAX; Jx++)		
	{
		for (int Jy=Jy0-RMAX; Jy<=Jy0+RMAX; Jy++)
		{
			float dR;
			if (finetune)
				// Distance from the cloud center (MQ units):
				dR = sqrt(pow(Jx-Center_pix[2]/MQ,2) + pow(Jy-Center_pix[3]/MQ,2));
				else
				// Length of the motion vector in MQ units:
				dR = sqrt(float(Jx*Jx + Jy*Jy));
				
			if (finetune==0 && (dR >= RMIN && dR <= RMAX) || finetune==1 && (dR <= RMAX))
			{
				find_kernel_parameters(Jx, Jy, MQ, Nx, Ny, &Grid_size, &Ix1, &Iy1);
								
				// The main compute kernel: searches for moving objects over all images,
				// all allowed base image tiles, for the given motion vector (Jx,Jy) in MQ units
				// Using the maximum 1024 threads per block, to cover 32x32 pixel tiles
				motion_search_cuda <<<Grid_size, Block_size>>> (d_image,N_images,pitch,Ix1,Iy1,Jx,Jy,MQ,p_min,d_dt,d_test_image,d_list,0,d_Pixel_counter,Nx,Ny);
				
				Jcount++;
				if (Jcount % 10000 == 0)
					printf("Processed %d motion vectors\n", Jcount);
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
		printf("Processed %d motion vectors\n\n", Jcount);
		printf ("Time per motion vector: %e\n", restime/Jcount);
		
		ERR( cudaMemcpy(&h_Pixel_counter, d_Pixel_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost))
		cudaDeviceSynchronize();
		printf("Number of bright pixels: %d\n", h_Pixel_counter);
		if (h_Pixel_counter > 0)
		{
			int N_cloud;
			
			cudaMemcpy(h_list.ix, d_list.ix, h_Pixel_counter*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_list.iy, d_list.iy, h_Pixel_counter*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_list.Jx, d_list.Jx, h_Pixel_counter*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_list.Jy, d_list.Jy, h_Pixel_counter*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_list.p, d_list.p, h_Pixel_counter*sizeof(float), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();

			float pmax = -1e30;
			int imax = -1;
			for (int i=0; i<h_Pixel_counter; i++)
			{
				if (h_list.p[i] > pmax)
				{
					pmax = h_list.p[i];
					imax = i;
				}
			}
			printf("Motion vector for the brightest pixel: (%8.2f, %8.2f); pixel coords: (%d, %d); p=%f (%f std)\n",
			rebin*MQ*h_list.Jx[imax], rebin*MQ*h_list.Jy[imax], rebin*h_list.ix[imax]+d_rebin, 
			rebin*h_list.iy[imax]+d_rebin, h_list.p[imax],
			h_list.p[imax]/sgm);
			
			int *Cluster_index = (int *)malloc(h_Pixel_counter*sizeof(int));			
			
			cudaDeviceSynchronize();
			gettimeofday (&tdr0, NULL);  

			// Cluster analysis on the CPU:
//			cluster_analysis(h_list, h_Pixel_counter, Cluster_index, &N_cloud);
			
			cudaDeviceSynchronize();
			gettimeofday (&tdr1, NULL);  			
			
			// CLuster analysis on the GPU (much faster when Pixel_counter>>100k)
			cluster_analysis_cuda(d_list, h_Pixel_counter, Cluster_index, &N_cloud);
			
			cudaDeviceSynchronize();
			gettimeofday (&tdr2, NULL);  

			tdr = tdr0;
			timeval_subtract (&restime, &tdr1, &tdr);
//			printf("Cluster analysis CPU: %e s\n", restime);
			tdr = tdr1;
			timeval_subtract (&restime, &tdr2, &tdr);
			printf("Cluster analysis GPU: %e s\n", restime);
			

			FILE *fp = fopen("list.dat", "w");
			for (int i=0; i<h_Pixel_counter; i++)
			{
				// Jx, Jy are converted to the original (non-rebinned) pixels:
				fprintf(fp, "%d %f %f %d %d %11e %d\n", i, rebin*MQ*h_list.Jx[i], rebin*MQ*h_list.Jy[i],
				rebin*h_list.ix[i]+d_rebin,
				rebin*h_list.iy[i]+d_rebin, h_list.p[i]/sgm, Cluster_index[i]);
			}
			fclose(fp);

			int NC;
			if (N_cloud < N_cloud_max)
				NC = N_cloud;
			else
				NC = N_cloud_max;

			Cloud *cloud = (Cloud *)malloc((N_cloud+1)*sizeof(Cloud));

			// Computing stats for the N_cloud brightest clouds			
			cloud_stats(h_list, h_Pixel_counter, N_cloud, Cluster_index, cloud, sgm, MQ, finetune,
				rebin, d_rebin);

// 			Not sure how useful the mosaic is
//			create_mosaic(Nx, Ny, h_list, h_Pixel_counter, Cluster_index, NC, name0, cloud);

			// Saving fits stacks for the most significant motion detections (clouds)
			char fits_name[100];
			for (int icloud=0; icloud<NC; icloud++)
			{
				for (int i=0; i<Nx*Ny; i++)
					h_test_image[i] = MASK0;
				cudaMemcpy2DAsync(d_test_image, pitch, h_test_image, Ny*sizeof(float), Ny*sizeof(float), Nx, cudaMemcpyHostToDevice);

				int imax = cloud[icloud].imax;
				int Jx = h_list.Jx[imax];
				int Jy = h_list.Jy[imax];
				find_kernel_parameters(Jx, Jy, MQ, Nx, Ny, &Grid_size, &Ix1, &Iy1);

				erase_image <<<Grid_size2, Block_size>>> (d_test_image, pitch, Nx, Ny, 0.0);

				motion_search_cuda <<<Grid_size, Block_size>>> (d_image,N_images,pitch,Ix1,Iy1,
				Jx,Jy,MQ,p_min,d_dt,d_test_image,d_list,1,d_Pixel_counter,Nx,Ny);
				
				ERR( cudaMemcpy2D(h_test_image, Ny*sizeof(float), d_test_image, pitch, Ny*sizeof(float), Nx, cudaMemcpyDeviceToHost) )
				cudaDeviceSynchronize();
				sprintf(fits_name,"cloud_%03d.fit", icloud);
				
				save_cloud_fits(Nx_ini, Ny_ini, Nx, Ny, 1, h_test_image, fits_name, name0, cloud, icloud,
					sgm, rebin, d_rebin);
			}
			free(Cluster_index);
	
		}
	}

	ERR( cudaFreeHost(h_image1) )
	ERR( cudaFreeHost(h_dt) )
	ERR( cudaFreeHost(h_test_image) )
	cudaFree(h_list.ix);
	cudaFree(h_list.iy);
	cudaFree(h_list.Jx);
	cudaFree(h_list.Jy);
	cudaFree(h_list.p);

	free(h_image);
	free(buf0);	
	cudaFree(d_list.ix);
	cudaFree(d_list.iy);
	cudaFree(d_list.Jx);
	cudaFree(d_list.Jy);
	cudaFree(d_list.p);

	return 0;
}