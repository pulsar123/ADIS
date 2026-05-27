#include "reconv.h"
#include "reconv.h"

/*  ADIS: Asteroid Discovery in Image Sequences
     
	The first step in the two-step procedure. 
	
	Inputs: an image sequence, plus the stacked version of it (master image).
	This should be RGB FITS files, calibrated, registered, background corrected, 
	and cropped as needed.
	Only the first image in the sequence needs to be plate solved.
	
	Outputs: the processed image sequence, B&W FITS files. All static objects
	(stars, galaxies etc) are removed, images are Gaussian smoothed to reduce
	the noise level, bilinear background removal applied at the end. The omages should look
	like pure noise, with some blask masked areas ()to hide bright stars artifacts).

	You need to install FFTW3 and CFITSIO libraries.
	
	Uses the CPU only (not GPU).

*/

// To compile:
//   gcc -O2 reconv.c func.c -lfftw3 -lcfitsio -lm -lz -o reconv

int	verbose = 0;

const int BUFFER_SIZE=1000;    // Make buffer size large enough for all print statements


int main(int argc, char **argv)
{
    struct timeval  tdr0, tdr1, tdr;
    double restime;

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++		
	// Processing command line arguments
	
	if (argc == 1)
	{
		printf("\n          *** ADIS: Asteroid Discovery in Image Sequences ***          \n\n");
		printf(" Stage 1 (out of 2): pre-processing the image sequence\n");
		
		printf("\n Syntax (any order):\n\n");
		printf(" %s  -m master_image  -o output_images_prefix  -FWHM value  -R kernel_radius  image1 image2 ... \n\n", argv[0]);
		printf(" -m master_image   :  name of the master image (the stacked version of all input images) \n");
		printf(" -o name           :  prefix for output images \n");
		printf(" -FWHM value       :  FWHM in pixels (measured from the master image) \n");
		printf(" -R kernel_radius  :  kernel radius in FWHM units \n");
		printf(" image1  ...       :  list of input FITS images (RGB only)\n");
		printf("\n Optional arguments [default value]:\n\n");
		printf(" -bg value         :  number of vertical tiles for bilinear background subtraction [5]\n");
		printf(" -bias value       :  bias for the output image [0]\n");
		printf(" -border width     :  mask out the output image border, the width is in kernel_radius units [0]\n");
		printf(" -grow_mask sgm    :  grow star masks using gaussian with sgm size (FWHM units)\n");
		printf(" -hot_pixels value :  mask pixels this mant std above the floor\n");
		printf(" -k image_name     :  output kernel image\n");
		printf(" -mask value       :  mask image pixels above this many master std\n");
		printf(" -no_rescale       :  do not rescale brightness\n");
		printf(" -v                :  verbose\n");
		printf("\n");		
		exit(0);
	}

// Default values for input parameters:
	char *file1 = "";
	char *file0 = "";
    char prefix[100];
    double R = 1.0;
	double outbias = 0.0;
	char *fkernel = "";
	int dump_kernel = 0;
	float FWHM = 0.0;
	int no_rescale = 0;
	double bmax = -1.0;
	int mask = 0;
	float border_width = 0.0;
	int mask_borders = 0;
	int grow_mask = 0;
	float mask_sgm = 0.0;
	int NTx = 5;
	int hot_pixels = 0;
	float hot_pixels_std = 0.0;
	
	int N_obligatory = 4; // Number of obligatory arguments
	int j = 1; // j counts all arguments
	int j0 = 1; // j0 counts only known arguments
	int error = 0;
	int iob = 0;
	while (j<argc)
	{
	// -------- Obligatory : --------------	
		// Name of the input master image
		if (strcmp(argv[j],"-m") == 0)
		{
			j++;
			j0 = j0 + 2;
			if (argv[j][0] == '-' || j>=argc)
			{
				error = j-1;
				break;
			}
			file0 = argv[j];
			iob++;
		}
		
		// Prefix for output images
		if (strcmp(argv[j],"-o") == 0)
		{
			j++;
			j0 = j0 + 2;
			if (argv[j][0] == '-' || j>=argc)
			{
				error = j-1;
				break;
			}
			strcpy(prefix, argv[j]);
			iob++;
		}
				
		// Kernel radius in FWHM units:
		if (strcmp(argv[j],"-R") == 0)
		{
			j++;
			j0 = j0 + 2;
			if (argv[j][0] == '-' || j>=argc)
			{
				error = j-1;
				break;
			}
			R = atof(argv[j]);
			iob++;
			assert(R>0);
		}						
		
		// FWHM for the master image in pixels (use Siril etc to measure it):
		if (strcmp(argv[j],"-FWHM") == 0)
		{
			j++;
			j0 = j0 + 2;
			if (argv[j][0] == '-' || j>=argc)
			{
				error = j-1;
				break;
			}
			FWHM = atof(argv[j]);
			iob++;
			assert(FWHM>0);
		}						

	// -------- Optional : --------------	
		// Number of vertical tiles in bilinear background removal
		// Number of horizontal tiles will be computed based on this value
		if (strcmp(argv[j],"-bg") == 0)
		{
			j++;
			j0 = j0 + 2;
			if (argv[j][0] == '-' || j>=argc)
			{
				error = j-1;
				break;
			}
			NTx = atoi(argv[j]);
			assert(NTx>0);
		}

		// Output image bias (0...1 scale), 0 by default:
		// This is only for visualization, not important for the atseroid search
		if (strcmp(argv[j],"-bias") == 0)
		{
			j++;
			j0 = j0 + 2;
			if (argv[j][0] == '-' || j>=argc)
			{
				error = j-1;
				break;
			}
			outbias = atof(argv[j]);
			assert(outbias>=0.0 && outbias<=1.0);
		}

		// Verbose (0 by default):
		if (strcmp(argv[j],"-v") == 0)
		{
			j0++;
			verbose = 1;
		}						
		
		// Name of the output kernel image (optional)
		if (strcmp(argv[j],"-k") == 0)
		{
			j++;
			j0 = j0 + 2;
			if (argv[j][0] == '-' || j>=argc)
			{
				error = j-1;
				break;
			}
			fkernel = argv[j];
			dump_kernel = 1;
		}		

		// Border width in R units:
		if (strcmp(argv[j],"-border") == 0)
		{
			j++;
			j0 = j0 + 2;
			if (argv[j][0] == '-' || j>=argc)
			{
				error = j-1;
				break;
			}
			border_width = atof(argv[j]);
			mask_borders = 1;
			assert(border_width>0);
		}						

		// No brightness rescale (0 by default):
		if (strcmp(argv[j],"-no_rescale") == 0)
		{
			j0++;
			no_rescale = 1;
		}						
		
		// Number of std units above the bias (in the master image) where all brighter pixels
		// will be masked out. To deal with bright star artifacts
		if (strcmp(argv[j],"-mask") == 0)
		{
			j++;
			j0 = j0 + 2;
			if (argv[j][0] == '-' || j>=argc)
			{
				error = j-1;
				break;
			}
			bmax = atof(argv[j]);
			mask = 1;
			assert(bmax>0);
		}						

		// This will make masked areas gow larger. To deal with bright star artifacts
		if (strcmp(argv[j],"-grow_mask") == 0)
		{
			j++;
			j0 = j0 + 2;
			if (argv[j][0] == '-' || j>=argc)
			{
				error = j-1;
				break;
			}
			mask_sgm = atof(argv[j]);
			grow_mask = 1;
			assert(mask_sgm>0);
		}						

		// Erasing hot pixels from the smoothed image
		if (strcmp(argv[j],"-hot_pixels") == 0)
		{
			j++;
			j0 = j0 + 2;
			if (argv[j][0] == '-' || j>=argc)
			{
				error = j-1;
				break;
			}
			hot_pixels_std = atof(argv[j]);
			hot_pixels = 1;
			assert(hot_pixels_std>0);
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

	int N_images = argc - j0;
	if (N_images < 1)
	{
		printf("\nThere should be at least 1 image in the stack. Exiting\n\n");
		exit(1);
	}	
	
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++	

	// For Gaussian smoothing of both the master and individual images:
	float sgm_blur = FWHM / sqrt(8*log(2.0));  // assuming perfecly gaussian PSF

	// Switching to pixel units:
	R = R * FWHM;
	double Pad = R;
	
	// For my memory management I find the largest image size (number of pixels). For that I need
	// to find the largest pad size for each dimension:
	int largest_pad = (int)(2*R);
	int pad1 = 2*(int)(10*sgm_blur);
	if (pad1 > largest_pad)
		largest_pad = pad1;
	pad1 = 2*(int)(10*FWHM*mask_sgm);
	if (pad1 > largest_pad)
		largest_pad = pad1;
	
    fitsfile *f0, *f1, *fout;
    int status = 0, naxis;
    long naxes[3];
	char buffer[150];
	char file_out[100];

    /* ---------- Read Master image ---------- */
    fits_open_file(&f0, file0, READONLY, &status);
    fits_error(status);

    fits_get_img_dim(f0, &naxis, &status);
    fits_error(status);
    fits_get_img_size(f0, 3, naxes, &status);
    fits_error(status);

    if (naxis != 3 || naxes[2] != 3) {
        fprintf(stderr, "Error: expected RGB FITS image\n");
        return EXIT_FAILURE;
    }

    int Nx = naxes[1]; // X axis is vertical
    int Ny = naxes[0]; // Y axis is horizontal
    int Nc = 3;
    long plane_pixels = (long)Nx * Ny;

	// Initializing my memory management:
	// (it's only used for 2D images of either float or fftw_complex types)
	my_alloc_init((Nx+largest_pad)*(Ny+largest_pad));

    float *img0 = malloc(sizeof(float) * plane_pixels * Nc);
    float *img1 = malloc(sizeof(float) * plane_pixels * Nc);
    fits_read_img(f0, TFLOAT, 1, plane_pixels * Nc, NULL, img0, NULL, &status);
    fits_error(status);
    fits_close_file(f0, &status);		
    fits_error(status);
	image_bw(img0, plane_pixels, Nc); // converting to B&W

    /* ---------- Padding ---------- */
    int Px = Nx + (int)(2*Pad);
    int Py = Ny + (int)(2*Pad);
    int P  = Px * Py;
    fftw_complex *F0 = fftw_malloc(sizeof(fftw_complex)*P);
    fftw_complex *F1 = fftw_malloc(sizeof(fftw_complex)*P);
    fftw_complex *K  = fftw_malloc(sizeof(fftw_complex)*P);
    fftw_complex *k_spatial = fftw_malloc(sizeof(fftw_complex)*P);
    float *img = malloc(sizeof(float) * plane_pixels);
    fftw_complex *F = fftw_malloc(sizeof(fftw_complex)*P);
    float *padded_out = malloc(sizeof(float)*P);
    float *cropped = malloc(sizeof(float)*plane_pixels);
    float *outbuf = malloc(sizeof(float) * plane_pixels);
	float *kernel = NULL;
	if (dump_kernel)
	{
		kernel = (float*)malloc(sizeof(float) * P);
	}
    double eps = 1e-8; // Used during kernel estimation
	double Nsigma = 3.0; // Used during sigma clipping
	// Low value cutoff for the input image, in sgm1 units:
	double Nsigma_cutoff = 5; // used during scaling
    int cx = Px/2;
    int cy = Py/2;
	double sgm_master;
	int k_master, k1;
	double p1, sgm1, p_master;
	long Npix1, Npix_master;


	gauss_blur(Nx, Ny, img0, img0, sgm_blur);

	//3-sigma clipping and bias removal for the master image
	sigma_clipping(img0, plane_pixels, Nsigma, &p_master, &sgm_master, &Npix_master, &k_master);

	if (verbose)
	{
		printf("\nMaster sgm clipping : %d, sgm=%e, p=%e, Npix=%ld\n", k_master, sgm_master, p_master, Npix_master);		
	}
	
	// Direct 2D FFT transform of padded img0/img1 with zero imaginary part -> F0/F1
	fft_images_padded(Nx, Ny, Px, Py, img0, F0, Pad);
	
	// Reading all input images one by one	
	for (int i_image=0; i_image<N_images; i_image++)
	{		
		file1 = argv[j0+i_image];
		printf("Reading %s\n", file1);
		
		fits_open_file(&f1, file1, READONLY, &status);
		fits_error(status);	
		
		fits_read_img(f1, TFLOAT, 1, plane_pixels * Nc, NULL, img1, NULL, &status);
		fits_error(status);
		
		image_bw(img1, plane_pixels, Nc);  // converting to B&W, storing to the R channel of img1

		sprintf(file_out, "%s%s", prefix, file1);
		sprintf(buffer, "rm -f %s >/dev/null", file_out);
		if (system(buffer))
			printf("Could not delete the file %s\n", file_out);
		fits_create_file(&fout, file_out, &status);
		fits_error(status);
		fits_copy_header(f1, fout, &status); // We copy all header info from input to output images
		fits_error(status);

			// We add +1 to avoid allocating scratch arrays internally again
		gauss_blur(Nx, Ny, img1, img1, sgm_blur);

		//3-sigma clipping and bias removal for the individual image
		sigma_clipping(img1, plane_pixels, Nsigma, &p1, &sgm1, &Npix1, &k1);
			
		// Direct 2D FFT transform of padded img0/img1 with zero imaginary part -> F1
		fft_images_padded(Nx, Ny, Px, Py, img1, F1, Pad);

		/* ---------- Kernel estimation ---------- */
		// Complex division F1/F0 = K
		derive_kernel_ft(P, F1, F0, K, eps);
	
		// Inverse FFT: K -> k_spatial (the kernel in real space)
		ifft_kernel(Px, Py, K, k_spatial);
		
		// Truncating the kernel beyond R radius
		truncate_kernel(Px, Py, k_spatial, R);

		if (dump_kernel)
		{
			/* circular shift */
			for (int x=0;x<Px;x++) {
				for (int y=0;y<Py;y++) {
					int xs = (x + cx) % Px;
					int ys = (y + cy) % Py;
					long i = x*Py+y;
					kernel[xs*Py+ys] = outbias + sqrt(pow(k_spatial[i][0],2)+pow(k_spatial[i][1],2));
				}
			}
		}

		// Direct FFT: k_spatial -> K
		fft_kernel(Px, Py, k_spatial, K);

		// Convoling the master image with the kernel derived from image_i
		convolve_image(Px, Py, F0, K, padded_out);

		// Cropping the convolved master image
		crop_image_centered(Nx, Ny, Px, Py, padded_out, cropped);

		// The low cutoff value:
		double p1_low = Nsigma_cutoff * sgm1;
		double S;
		int k_scale = 0;
		
		// Computing scaling coeficient S between the convolved master cropped and input image img1:
		if (no_rescale)
			S = 1.0;
		else
		{
			S = scaling(img1, cropped, plane_pixels, p1_low, &k_scale);
		}
		
		for (long i=0; i<plane_pixels; i++)
		{
			// Applying the scaling to the convolved master, and subtracting the result from the input image:
			outbuf[i] = img1[i] - S*cropped[i];	
		}
		
		if (verbose)
		{
			printf("\nImage sgm clipping  : %d %e %e %ld\n", k1, sgm1, p1, Npix1);		
			printf("Scaling: %d %e\n", k_scale, S);		
		}

		int N_excluded = 0;
		
		// Masking artifacts from bright stars:
		if (mask)
			for (long i=0;i<plane_pixels;i++)
			{
				float p = cropped[i]/sgm_master / bmax;
							
				if (p >= 1.0)
					outbuf[i] = MASK0;  // masked pixel value = -101
				
				N_excluded++;
			}

		// Bilinear background subtraction, and adding the bias:
		// Background model has these many tiles along each dimension:
		int NTy = (int)((float)NTx / (float)Nx * (float)Ny + 0.5);
		if(verbose)
			printf("\nBackground model: %d x %d tiles, %d x %d pixels each\n", NTy, NTx, Ny/NTy, Nx/NTx);
		subtract_background(0, 1, outbuf, Nx, Ny, NTx, NTy, outbias);

		if (hot_pixels)
			erase_hot_pixels(outbuf, Nx, Ny, hot_pixels_std*sgm1+outbias);

		// Grow masked stars if requested:
		if (mask && grow_mask)
		{
			N_excluded = 0;
			grow_masked_stars(outbuf, Nx, Ny, FWHM*mask_sgm, &N_excluded);
		}
		
		if (verbose && mask)
		{
			printf("\nExcluded pixels fraction: %f\n", (float)N_excluded/plane_pixels);
		}


		if (mask_borders)
			borders(outbuf, Nx, Ny, (int)(border_width*R));

		/* ---------- Write output FITS ---------- */
		
		if (dump_kernel)
		{
			dump_fits(Px, Py, 1, kernel, fkernel);
		}

		// Writing a BW output image

		// Storing important code parameters as additional FITS keywords:
		fits_write_key(fout, TFLOAT, "FWHM", &FWHM, "asteroid_search", &status);
		fits_write_key(fout, TDOUBLE, "R", &R, "asteroid_search", &status);
		fits_write_key(fout, TDOUBLE, "BMAX", &bmax, "asteroid_search", &status);
		fits_write_key(fout, TDOUBLE, "BIAS", &outbias, "asteroid_search", &status);

		int naxis_out = 2;
		fits_update_key(fout, TINT, "NAXIS", &naxis_out,
						"number of array dimensions", &status);
			fits_error(status);
		long NY = Ny;
		fits_update_key(fout, TLONG, "NAXIS1", &NY,
						"length of x axis", &status);
			fits_error(status);
		long NX = Nx;
		fits_update_key(fout, TLONG, "NAXIS2", &NX,
						"length of y axis", &status);
			fits_error(status);
		fits_delete_key(fout, "NAXIS3", &status);
			fits_error(status);
		long fpixel = 1;
		long nelem1  = (long)Nx * Ny;	
		fits_write_img(fout, TFLOAT, fpixel, nelem1, outbuf, &status);
		fits_error(status);

		fits_close_file(fout, &status);
		fits_close_file(f1, &status);
	
	}  // i_image cycle

    /* ---------- Cleanup ---------- */
	my_alloc_destroy();

    fftw_free(F);
    free(padded_out);
    free(cropped);
    free(img);

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
