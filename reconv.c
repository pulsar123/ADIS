// reconv 5: includes rescaling
// 7: full processing for each channel; removing biases fort each channel; command line handler
// 8: optional kernel output

#include "reconv.h"


int main(int argc, char **argv)
{
	
	// Processing command line arguments
	
	if (argc == 1)
	{
		printf("\n Syntax (any order):\n\n");
		printf(" %s  -i input_image  -m master_image  -o output_image  -R kernel_radius\n\n", argv[0]);
		printf(" Optional arguments [default value]:\n\n");
		printf(" -bias value  :  bias for the output image [0]\n");
		printf(" -blur sigma  :  sigma for the Gaussian blur for the input image [0]\n");
		printf(" -k image_name:  output kernel image\n");
		printf(" -v           :  verbose\n");
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
	double sgm_blur = 0.0;
	int perform_blur = 0;
	
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

		// Verbose (0 by default):
		if (strcmp(argv[j],"-v") == 0)
		{
			verbose = 1;
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
			sgm_blur = atof(argv[j]);
			perform_blur = 1;
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

    /* ---------- Read Master RGB ---------- */
    float *buf0 = malloc(sizeof(float) * plane_pixels * Nc);
    fits_read_img(f0, TFLOAT, 1,
                  plane_pixels * Nc, NULL, buf0, NULL, &status);
    fits_error(status);
    fits_close_file(f0, &status);
	
    /* ---------- Read Image1 RGB ---------- */
    fits_open_file(&f1, file1, READONLY, &status);
    fits_error(status);

    float *buf1 = malloc(sizeof(float) * plane_pixels * Nc);
    fits_read_img(f1, TFLOAT, 1,
                  plane_pixels * Nc, NULL, buf1, NULL, &status);
    fits_error(status);
    fits_create_file(&fout, file_out, &status);
    fits_error(status);
	fits_copy_header(f1, fout, &status);
	fits_error(status);
	
    /* ---------- Padding ---------- */
    int Px = Nx + (int)(2*Pad);
    int Py = Ny + (int)(2*Pad);
    int P  = Px * Py;

    double *img0 = malloc(sizeof(double) * plane_pixels);
    double *img1 = malloc(sizeof(double) * plane_pixels);

    fftw_complex *F0 = fftw_malloc(sizeof(fftw_complex)*P);
    fftw_complex *F1 = fftw_malloc(sizeof(fftw_complex)*P);
    fftw_complex *K  = fftw_malloc(sizeof(fftw_complex)*P);
    fftw_complex *k_spatial = fftw_malloc(sizeof(fftw_complex)*P);

    float *outbuf = malloc(sizeof(float) * plane_pixels * Nc);
    double *img = malloc(sizeof(double) * plane_pixels);
    fftw_complex *F = fftw_malloc(sizeof(fftw_complex)*P);
    double *padded_out = malloc(sizeof(double)*P);
    double *cropped = malloc(sizeof(double)*plane_pixels);
	float *kernel;
	if (dump_kernel)
	{
		kernel = (float*)malloc(sizeof(float) * Px*Py * Nc);
	}

    double eps = 1e-8;
	double Nsigma = 3.0;
	// Low value cutoff for the input image, in sgm1 units:
	double Nsigma_cutoff = 5;
    int cx = Px/2;
    int cy = Py/2;


	// Full processing, separately for each channel (R, G, B)
    for (int c=0;c<3;c++) {
		if (verbose)
			printf("\n=== Channel %d ===\n",c);


		// Extracting the channel for the master image
	    for (long i=0;i<plane_pixels;i++)
			img0[i] = buf0[c*plane_pixels + i];
		// Extracting the channel for the individual image
	    for (long i=0;i<plane_pixels;i++)
			img1[i] = buf1[c*plane_pixels + i];

		if (perform_blur)
		{
			gauss_blur(Nx, Ny, img1, sgm_blur);
		}

		double p1, sgm1, p_master, sgm_master;
		long Npix1, Npix_master;

		//3-sigma clipping and bias removal for the master image
		sigma_clipping(img0, plane_pixels, Nsigma, &p_master, &sgm_master, &Npix_master);
		//3-sigma clipping and bias removal for the individual image
		sigma_clipping(img1, plane_pixels, Nsigma, &p1, &sgm1, &Npix1);
			
		// Direct 2D FFT transform of padded img0/img1 with zero imaginary part -> F0/F1
		fft_images_padded(Nx, Ny, Px, Py, img0, F0, Pad);
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
					long i = IDX(x,y,Py);
					kernel[c*Px*Py + IDX(xs,ys,Py)] = outbias + sqrt(pow(k_spatial[i][0],2)+pow(k_spatial[i][1],2));
				}
			}
		}

		// Direct FFT: k_spatial -> K
		fft_kernel(Px, Py, k_spatial, K);

        convolve_image(Px, Py, F0, K, padded_out);

        /* crop */
		crop_image_centered(Nx, Ny, Px, Py, padded_out, cropped);

		// The low cutoff value:
		double p1_low = Nsigma_cutoff * sgm1;
	
		// Computing scaling coeficient S between the convolved master cropped and input image img1:
		double S = scaling(img1, cropped, plane_pixels, p1_low);

		// Applying the scaling to the convolved master, and subtracting the result from the input image:
		for (long i=0; i<plane_pixels; i++)
		{
			outbuf[c*plane_pixels + i] = outbias + img1[i] - S*cropped[i];
		}

	}  // color channels loop

    fftw_free(F);
    free(padded_out);
    free(cropped);
    free(img);
    free(buf0);


    /* ---------- Write output FITS ---------- */
	
	if (dump_kernel)
	{
		fitsfile *fk;
		fits_create_file(&fk, fkernel, &status);
		fits_error(status);
		long nelem1  = (long)Px * Py * 3;
		long naxes[3] = {Py, Px, 3};
		fits_create_img(fk, FLOAT_IMG, 3, naxes, &status);
		fits_error(status);
		fits_write_img(fk, TFLOAT, 1, nelem1, kernel, &status);
		fits_error(status);
		fits_close_file(fk, &status);
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
