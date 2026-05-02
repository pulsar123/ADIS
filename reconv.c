#include "reconv.h"

// To compile:
//   gcc -O2 reconv.c func.c -lfftw3 -lcfitsio -lm -lz -o reconv

int	verbose = 0;

const int BUFFER_SIZE=1000;    // Make buffer size large enough for all print statements


int main(int argc, char **argv)
{
    struct timeval  tdr0, tdr1, tdr;
    double restime;
	
	// Processing command line arguments
	
	if (argc == 1)
	{
		printf("\n Syntax (any order):\n\n");
		printf(" %s  -i input_image  -m master_image  -o output_image  -R kernel_radius\n\n", argv[0]);
		printf(" Optional arguments [default value]:\n\n");
		printf(" -bias value    :  bias for the output image [0]\n");
		printf(" -blur FWHM     :  FWHM for the Gaussian blur for the input images [0]\n");
		printf(" -bmax value    :  erase image pixels above this many master std [0]\n");
		printf(" -k image_name  :  output kernel image\n");
		printf(" -mask          :  use a negative number mask with -bmax\n");
		printf(" -no_rescale    :  do not rescale brightness\n");
		printf(" -subtract_only :  just subtract the master from the image\n");
		printf(" -v             :  verbose\n");
		printf("\n");		
		exit(0);
	}

	char *file1 = "";
	char *file0 = "";
    char *file_out = "";
    double R = 1.0;
	double outbias = 0.0;
	char *fkernel = "";
	int dump_kernel = 0;
	float FWHM = 0.0;
	int perform_blur = 0;
	int no_rescale = 0;
	double bmax = -1.0;
	int subtract_only = 0;
	int mask = 0;
	
	int N_obligatory = 4; // Number of obligatory arguments
	int j = 1;
	int error = 0;
	int iob = 0;
	while (j<argc)
	{
		
		// Name of the input individual image
		if (strcmp(argv[j],"-i") == 0)
		{
			j++;
			if (argv[j][0] == '-' || j>=argc)
			{
				error = j-1;
				break;
			}
			file1 = argv[j];
			iob++;
		}
		
		// Name of the input master image
		if (strcmp(argv[j],"-m") == 0)
		{
			j++;
			if (argv[j][0] == '-' || j>=argc)
			{
				error = j-1;
				break;
			}
			file0 = argv[j];
			iob++;
		}
		
		// Name of the output image
		if (strcmp(argv[j],"-o") == 0)
		{
			j++;
			if (argv[j][0] == '-' || j>=argc)
			{
				error = j-1;
				break;
			}
			file_out = argv[j];
			iob++;
		}
				
		// Maximum kernel radius in pixels:
		if (strcmp(argv[j],"-R") == 0)
		{
			j++;
			if (argv[j][0] == '-' || j>=argc)
			{
				error = j-1;
				break;
			}
			R = atof(argv[j]);
			iob++;
		}						
		
		// Output image bias (0...1 scale), 0 by default:
		if (strcmp(argv[j],"-bias") == 0)
		{
			j++;
			if (argv[j][0] == '-' || j>=argc)
			{
				error = j-1;
				break;
			}
			outbias = atof(argv[j]);
		}

		// Only subtract the master, no convolution etc (0 by default):
		if (strcmp(argv[j],"-subtract_only") == 0)
		{
			subtract_only = 1;
		}						
		
		// Verbose (0 by default):
		if (strcmp(argv[j],"-v") == 0)
		{
			verbose = 1;
		}						
		
		// Use a negative number mask with -bmax:
		if (strcmp(argv[j],"-mask") == 0)
		{
			mask = 1;
		}						

		// Name of the output kernel image (optional)
		if (strcmp(argv[j],"-k") == 0)
		{
			j++;
			if (argv[j][0] == '-' || j>=argc)
			{
				error = j-1;
				break;
			}
			fkernel = argv[j];
			dump_kernel = 1;
		}		

		// Gaussian blur sigma in pixels:
		if (strcmp(argv[j],"-blur") == 0)
		{
			j++;
			if (argv[j][0] == '-' || j>=argc)
			{
				error = j-1;
				break;
			}
			FWHM = atof(argv[j]);
			perform_blur = 1;
		}						

		// No brightness rescale (0 by default):
		if (strcmp(argv[j],"-no_rescale") == 0)
		{
			no_rescale = 1;
		}						
		
		if (strcmp(argv[j],"-bmax") == 0)
		{
			j++;
			if (argv[j][0] == '-' || j>=argc)
			{
				error = j-1;
				break;
			}
			bmax = atof(argv[j]);
		}						

		j++;
	} // /end of argv while loop
	
	
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
	
	double Pad = R;

    fitsfile *f0, *f1, *fout;
    int status = 0, naxis;
    long naxes[3];

// Number of OpenMP therads to use:
	#ifdef _OPENMP
	int NOMP = 3;
	omp_set_num_threads(3);
    init_all_locks();
	if (verbose)
		printf("\nOpenMP: using %d threads\n",NOMP);
	#else
	int NOMP = 1;
	#endif	

    /* ---------- Open Master ---------- */
    fits_open_file(&f0, file0, READONLY, &status);
    fits_error(status);

    fits_get_img_dim(f0, &naxis, &status);
    fits_get_img_size(f0, 3, naxes, &status);
    fits_error(status);

    if (naxis != 3 || naxes[2] != 3) {
        fprintf(stderr, "Error: expected RGB FITS image\n");
        return EXIT_FAILURE;
    }

    int Nx = naxes[1];
    int Ny = naxes[0];
    int Nc = 3;
    long plane_pixels = (long)Nx * Ny;

    fits_open_file(&f1, file1, READONLY, &status);
    fits_error(status);	
	
	float *buf0;
	float *buf1;

	// In OmpenMP, reading the Master and Input files in parallel
	// (requires the Cfitsio library to be compiled with ./configure --enable-reentrant option)
	#pragma omp parallel sections firstprivate(status)
	{
	#pragma omp section
	{
    buf0 = (float *)malloc(sizeof(float) * plane_pixels * Nc);
    fits_read_img(f0, TFLOAT, 1,
                  plane_pixels * Nc, NULL, buf0, NULL, &status);
    fits_error(status);
    fits_close_file(f0, &status);		
	}
	
	#pragma omp section
	{
    buf1 = (float *)malloc(sizeof(float) * plane_pixels * Nc);
    fits_read_img(f1, TFLOAT, 1,
                  plane_pixels * Nc, NULL, buf1, NULL, &status);
    fits_error(status);				
	}		
	} // pragma section


	char buffer[50];
	sprintf(buffer, "rm -f %s >/dev/null", file_out);
	if (system(buffer))
		printf("Could not delete the file %s\n", file_out);
    fits_create_file(&fout, file_out, &status);
    fits_error(status);
	fits_copy_header(f1, fout, &status);
	fits_error(status);
	
    /* ---------- Padding ---------- */
    int Px = Nx + (int)(2*Pad);
    int Py = Ny + (int)(2*Pad);
    int P  = Px * Py;

    gettimeofday (&tdr0, NULL);  

    float *img0 = malloc(sizeof(float) * plane_pixels * NOMP);
    float *img1 = malloc(sizeof(float) * plane_pixels * NOMP);

    fftw_complex *F0 = fftw_malloc(sizeof(fftw_complex)*P * NOMP);
    fftw_complex *F1 = fftw_malloc(sizeof(fftw_complex)*P * NOMP);
    fftw_complex *K  = fftw_malloc(sizeof(fftw_complex)*P * NOMP);
    fftw_complex *k_spatial = fftw_malloc(sizeof(fftw_complex)*P * NOMP);

    float *img = malloc(sizeof(float) * plane_pixels * NOMP);
    fftw_complex *F = fftw_malloc(sizeof(fftw_complex)*P * NOMP);
    float *padded_out = malloc(sizeof(float)*P * NOMP);
    float *cropped = malloc(sizeof(float)*plane_pixels * NOMP);

    float *outbuf = malloc(sizeof(float) * plane_pixels * Nc);
	float *kernel = NULL;
	if (dump_kernel)
	{
		kernel = (float*)malloc(sizeof(float) * Px*Py * Nc);
	}


    gettimeofday (&tdr1, NULL);  
    tdr = tdr0;
    timeval_subtract (&restime, &tdr1, &tdr);
//    printf ("time: %e\n", restime);

    double eps = 1e-8;
	double Nsigma = 3.0;
	// Low value cutoff for the input image, in sgm1 units:
	double Nsigma_cutoff = 5;
    int cx = Px/2;
    int cy = Py/2;

	double Sinc = 0.0;	
	double sgm_master[3];
	
	float sgm_blur = FWHM / sqrt(8*log(2.0));
	printf("sgm_blur = %f\n", sgm_blur);


	// Full processing, separately for each channel (R, G, B)
	#pragma omp parallel
	#pragma omp for reduction(+:Sinc) ordered
	for (int c=0; c<3; c++) 
	
	{
		double S = 1;

		#ifdef _OPENMP
		int ithread = omp_get_thread_num();
		#else
		int ithread = 0;
		#endif		
		
		int k_master, k1;
		int k_scale = 0;
		
		// Extracting the channel for the master image
	    for (long i=0;i<plane_pixels;i++)
			img0[ithread*plane_pixels + i] = buf0[c*plane_pixels + i];
		// Extracting the channel for the individual image
	    for (long i=0;i<plane_pixels;i++)
			img1[ithread*plane_pixels + i] = buf1[c*plane_pixels + i];

		if (perform_blur)
		{
			gauss_blur(0, 2, Nx, Ny, &img0[ithread*plane_pixels], &img0[ithread*plane_pixels], sgm_blur);
			gauss_blur(1, 2, Nx, Ny, &img1[ithread*plane_pixels], &img1[ithread*plane_pixels], sgm_blur);
		}

		double p1, sgm1, p_master;
		long Npix1, Npix_master;

		//3-sigma clipping and bias removal for the master image
		sigma_clipping(&img0[ithread*plane_pixels], plane_pixels, Nsigma, &p_master, &sgm_master[c], &Npix_master, &k_master);

		//3-sigma clipping and bias removal for the individual image
		sigma_clipping(&img1[ithread*plane_pixels], plane_pixels, Nsigma, &p1, &sgm1, &Npix1, &k1);
			
		if (subtract_only == 0)
		{
			// Direct 2D FFT transform of padded img0/img1 with zero imaginary part -> F0/F1
			fft_images_padded(0, 2, Nx, Ny, Px, Py, &img0[ithread*plane_pixels], &F0[ithread*P], Pad);
			fft_images_padded(1, 2, Nx, Ny, Px, Py, &img1[ithread*plane_pixels], &F1[ithread*P], Pad);

			/* ---------- Kernel estimation ---------- */
			// Complex division F1/F0 = K
			derive_kernel_ft(P, &F1[ithread*P], &F0[ithread*P], &K[ithread*P], eps);
		
			// Inverse FFT: K -> k_spatial (the kernel in real space)
			ifft_kernel(Px, Py, &K[ithread*P], &k_spatial[ithread*P]);
			
			// Truncating the kernel beyond R radius
			truncate_kernel(Px, Py, &k_spatial[ithread*P], R);

			if (dump_kernel)
			{
				/* circular shift */
				for (int x=0;x<Px;x++) {
					for (int y=0;y<Py;y++) {
						int xs = (x + cx) % Px;
						int ys = (y + cy) % Py;
						long i = IDX(x,y,Py);
						kernel[c*Px*Py + IDX(xs,ys,Py)] = outbias + sqrt(pow(k_spatial[ithread*P+i][0],2)+pow(k_spatial[ithread*P+i][1],2));
					}
				}
			}

			// Direct FFT: k_spatial -> K
			fft_kernel(Px, Py, &k_spatial[ithread*P], &K[ithread*P]);

			convolve_image(0, 1, Px, Py, &F0[ithread*P], &K[ithread*P], &padded_out[ithread*P]);

			/* crop */
			crop_image_centered(Nx, Ny, Px, Py, &padded_out[ithread*P], &cropped[ithread*plane_pixels]);

			// The low cutoff value:
			double p1_low = Nsigma_cutoff * sgm1;
	
			// Computing scaling coeficient S between the convolved master cropped and input image img1:
			if (no_rescale)
				S = 1.0;
			else
			{
				S = scaling(&img1[ithread*plane_pixels], &cropped[ithread*plane_pixels], plane_pixels, p1_low, &k_scale);
			}
			
			for (long i=0; i<plane_pixels; i++)
			{
				// Applying the scaling to the convolved master, and subtracting the result from the input image:
				outbuf[c*plane_pixels + i] = img1[ithread*plane_pixels+i] - S*cropped[ithread*plane_pixels+i];	
			}
		}
		else
		// if subtract_only
		{
			for (long i=0; i<plane_pixels; i++)
			{
				outbuf[c*plane_pixels + i] = outbias + img1[ithread*plane_pixels+i] - img0[ithread*plane_pixels+i];	
			}
		}
		
		#pragma omp ordered
		if (verbose)
		{
			printf("\n=== Channel %d ===\n",c);
			printf("Sigma clipping:\n");		
			printf("Master : %d %e %e %ld\n", k_master, sgm_master[c], p_master, Npix_master);		
			printf("Image  : %d %e %e %ld\n", k1, sgm1, p1, Npix1);		
			printf("Scaling: %d %e\n", k_scale, S);		
		}

	}  // color channels loop
	
	if (subtract_only == 0)
	{
		for (long i=0;i<plane_pixels;i++)
		{
			double w = 1.0;
			
			if (bmax > 0)
			{
				// p=1 when the pixel brightness =bmax in sgm_master units, in the blurred master image
				float p = sqrt(pow(cropped[i]/sgm_master[0],2)
					+pow(cropped[plane_pixels + i]/sgm_master[1],2)
					+pow(cropped[2*plane_pixels + i]/sgm_master[2],2)) / bmax;

				// Transitional brightness range for the master image:
				double PM_DELTA = 0.1; // Determines the half-width of the transitional brightness range
				double pm_min = 1-PM_DELTA;
				double pm_max = 1+PM_DELTA;
				
				// Optional brightness cutoff
				if (p < pm_min)
				{
					w = 1.0;
				}
				else if (p > pm_max)
				{
					w = 0.0;
				}
				else
				{
					double t = (p - pm_min)/(pm_max - pm_min);
					// w(t) = 1 at t=0, =0.5 at t=0.5, =0 at t=1, and zero derivatives on both ends
					w = 1.0 + t*t*(-3.0 + 2.0*t);
				}
			}
			
			
			// Adding fixed bias outbias:
			for (int c=0; c<3; c++)
			{
				if (mask && w<0.5)
					outbuf[c*plane_pixels + i] = MASK0;  // masked pixel value = -101
				else
					outbuf[c*plane_pixels + i] = outbias + w*outbuf[c*plane_pixels + i];
			}
			
			Sinc = Sinc + w;
		}
		

		if (verbose && bmax >= 0.0)
		{
			printf("\nExcluded pixels fraction: %f\n",(plane_pixels-Sinc)/plane_pixels);
		}
	} // if not subtract_only

    fftw_free(F);
    free(padded_out);
    free(cropped);
    free(img);
    free(buf0);


    /* ---------- Write output FITS ---------- */
	
	if (dump_kernel)
	{
		dump_fits(Px, Py, 3, kernel, fkernel);
	}

	long fpixel = 1;
	long nelem  = (long)Nx * Ny * 3;

	fits_write_img(fout, TFLOAT, fpixel, nelem, outbuf, &status);
    fits_error(status);

    fits_close_file(fout, &status);
	fits_close_file(f1, &status);

    /* ---------- Cleanup ---------- */
    free(buf1);
    free(outbuf);
    free(img1);
	free(img0);

    fftw_free(F0);
    fftw_free(F1);
    fftw_free(K);
    fftw_free(k_spatial);
	if (dump_kernel)
	{
		free(kernel);
	}

    fftw_cleanup();
	

    return EXIT_SUCCESS;
}
