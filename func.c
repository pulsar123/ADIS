#include "reconv.h"

#ifdef _OPENMP
#define NLOCKS 10
omp_lock_t my_lock[NLOCKS];
#endif

int timeval_subtract (double *result, struct timeval *x, struct timeval *y)
{
  struct timeval result0;

  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_usec < y->tv_usec) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
    y->tv_usec -= 1000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_usec - y->tv_usec > 1000000) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000;
    y->tv_usec += 1000000 * nsec;
    y->tv_sec -= nsec;
  }

  /* Compute the time remaining to wait.
     tv_usec is certainly positive. */
  result0.tv_sec = x->tv_sec - y->tv_sec;
  result0.tv_usec = x->tv_usec - y->tv_usec;
  *result = ((double)result0.tv_usec)/1e6 + (double)result0.tv_sec;

  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}


/* ------------------------------------------------ */

#ifdef _OPENMP
void init_all_locks()
{
	for (int i=0; i<NLOCKS; i++)
		omp_init_lock(&my_lock[i]);
}
#endif

/* ------------------------------------------------ */


void fft_image(int Nx, int Ny,
                float *img0,
                fftw_complex *F0)
{
    fftw_complex *in = fftw_malloc(sizeof(fftw_complex)*Nx*Ny);
    fftw_plan p;

    /* Image0 */
    for (int i=0;i<Nx*Ny;i++) {
        in[i][0] = img0[i];
        in[i][1] = 0.0;
    }

    #ifdef _OPENMP
	omp_set_lock(&my_lock[0]);
	#endif
    p = fftw_plan_dft_2d(Nx, Ny, in, F0, FFTW_FORWARD, FFTW_ESTIMATE);
    #ifdef _OPENMP
	omp_unset_lock(&my_lock[0]);
	#endif

    fftw_execute(p);

    #ifdef _OPENMP
	omp_set_lock(&my_lock[1]);
	#endif
    fftw_destroy_plan(p);
    #ifdef _OPENMP
	omp_unset_lock(&my_lock[1]);
	#endif
	

    fftw_free(in);
}

/* ------------------------------------------------ */


void derive_kernel_ft(int N, fftw_complex *F1,
                      fftw_complex *F0,
                      fftw_complex *K,
                      double eps)
{
	
	// Complex division F1/F0
	
    for (int i=0;i<N;i++) {
        double a = F1[i][0];
        double b = F1[i][1];
        double c = F0[i][0];
        double d = F0[i][1];

// Small number eps is to prevent division by zero:
        double denom = c*c + d*d + eps;

        K[i][0] = ( a*c + b*d ) / denom;
        K[i][1] = ( b*c - a*d ) / denom;
    }
}

/* ------------------------------------------------ */

void ifft_kernel(int Nx, int Ny,
                 fftw_complex *K,
                 fftw_complex *k_spatial)
{
	// Inverse FFT: K -> k_spatial
	fftw_plan p;

    #ifdef _OPENMP
	omp_set_lock(&my_lock[2]);
	#endif
	p = fftw_plan_dft_2d(Nx, Ny, K, k_spatial, FFTW_BACKWARD, FFTW_ESTIMATE);
    #ifdef _OPENMP
	omp_unset_lock(&my_lock[2]);
	#endif
	
    fftw_execute(p);
	
    #ifdef _OPENMP
	omp_set_lock(&my_lock[3]);
	#endif
    fftw_destroy_plan(p);
    #ifdef _OPENMP
	omp_unset_lock(&my_lock[3]);
	#endif

    double norm = 1.0 / (Nx * Ny);
    for (int i=0;i<Nx*Ny;i++) {
        k_spatial[i][0] *= norm;
        k_spatial[i][1] *= norm;
    }
}

/* ------------------------------------------------ */

void truncate_kernel(int Nx, int Ny,
                     fftw_complex *k,
                     double R)
{
	// Truncating the kernel spatially, 
	
    int cx = Nx/2;
    int cy = Ny/2;
    double R0 = 0.5 * R;

    fftw_complex *tmp = fftw_malloc(sizeof(fftw_complex)*Nx*Ny);

    /* circular shift */
    for (int x=0;x<Nx;x++) {
        for (int y=0;y<Ny;y++) {
            int xs = (x + cx) % Nx;
            int ys = (y + cy) % Ny;
            tmp[IDX(xs,ys,Ny)][0] = k[IDX(x,y,Ny)][0];
            tmp[IDX(xs,ys,Ny)][1] = k[IDX(x,y,Ny)][1];
        }
    }

    /* radial taper */
    for (int x=0;x<Nx;x++) {
        for (int y=0;y<Ny;y++) {
            double dx = x - cx;
            double dy = y - cy;
            double r = sqrt(dx*dx + dy*dy);

            double w;
            if (r <= R0)
                w = 1.0;
            else if (r >= R)
                w = 0.0;
            else
// Cubic spline:
{
				double t = (r-R0)/(R-R0);
				// w(t) = 1 at t=0, =0.5 at t=0.5, =0 at t=1, and zero derivatives on both ends
				w = 1.0 + t*t*(-3.0 + 2.0*t);
// Gaussian:				
//			w = exp(-0.5*(r*r/(R*R)));
}

            tmp[IDX(x,y,Ny)][0] *= w;
            tmp[IDX(x,y,Ny)][1] *= w;
        }
    }

	// Circular shift back
    for (int x=0; x<Nx; x++) {
        for (int y=0; y<Ny; y++) {
            int xs = (x + cx) % Nx;
            int ys = (y + cy) % Ny;

            k[IDX(x,y,Ny)][0] = tmp[IDX(xs,ys,Ny)][0];
            k[IDX(x,y,Ny)][1] = tmp[IDX(xs,ys,Ny)][1];
        }
    }

    fftw_free(tmp);
	return;
}

/* ------------------------------------------------ */


void fft_kernel(int Nx, int Ny,
                fftw_complex *k_spatial,
                fftw_complex *K)
{
	fftw_plan p;
	
    #ifdef _OPENMP
	omp_set_lock(&my_lock[4]);
	#endif
    p = fftw_plan_dft_2d(Nx, Ny, k_spatial, K, FFTW_FORWARD, FFTW_ESTIMATE);
    #ifdef _OPENMP
	omp_unset_lock(&my_lock[4]);
	#endif
	
    fftw_execute(p);

    #ifdef _OPENMP
	omp_set_lock(&my_lock[5]);
	#endif
    fftw_destroy_plan(p);
    #ifdef _OPENMP
	omp_unset_lock(&my_lock[5]);
	#endif
}

/* ------------------------------------------------ */

void convolve_image(int i_image, int N_images, int Nx, int Ny,
                    fftw_complex *F0,
                    fftw_complex *K,
                    float *out)
{
	static fftw_complex *Fout;

	if (i_image == 0)
		Fout = fftw_malloc(sizeof(fftw_complex)*Nx*Ny);

    /* multiply in Fourier space */
    for (int i=0;i<Nx*Ny;i++) {
        double a = F0[i][0];
        double b = F0[i][1];
        double c = K[i][0];
        double d = K[i][1];

        Fout[i][0] = a*c - b*d;
        Fout[i][1] = a*d + b*c;
    }
	
    /* inverse FFT */
	fftw_plan p;
    #ifdef _OPENMP
	omp_set_lock(&my_lock[6]);
	#endif
    p = fftw_plan_dft_2d(Nx, Ny, Fout, Fout, FFTW_BACKWARD, FFTW_ESTIMATE);
    #ifdef _OPENMP
	omp_unset_lock(&my_lock[6]);
	#endif
	
    fftw_execute(p);

    #ifdef _OPENMP
	omp_set_lock(&my_lock[7]);
	#endif
    fftw_destroy_plan(p);
    #ifdef _OPENMP
	omp_unset_lock(&my_lock[7]);
	#endif

    double norm = 1.0 / (Nx * Ny);
    for (int i=0;i<Nx*Ny;i++)
        out[i] = Fout[i][0] * norm;

	if (i_image == N_images-1)
		fftw_free(Fout);
}

/* ------------------------------------------------ */


void fft_images_padded(int i_image, int N_images,int Nx, int Ny,
                       int Px, int Py,
                       float *img0,
                       fftw_complex *F0,
					   int Pad)
{
	// First pad the image with Pad width, then direct FFT -> F0
	static float *p0;
	static fftw_complex *in;
	
    int P = Px * Py;

	if (i_image == 0)
	{
		p0 = calloc(P, sizeof(float));
		in = fftw_malloc(sizeof(fftw_complex)*P);
	}

    /* zero-fill */
    for (int i = 0; i < Px*Py; i++)
        p0[i] = 0.0;

    /* copy image into center */
    for (int x = 0; x < Nx; x++) {
        for (int y = 0; y < Ny; y++) {
            p0[(x + Pad)*Py + (y + Pad)] =
                img0[x*Ny + y];
        }
    }
	
    fftw_plan plan;

    /* Image0 */
    for (int i=0;i<P;i++) {
        in[i][0] = p0[i];
        in[i][1] = 0.0;
    }
	// Direct 2D FFT transform in (padded img0 with zero imaginary part) -> F0.
    #ifdef _OPENMP
	omp_set_lock(&my_lock[8]);
	#endif
    plan = fftw_plan_dft_2d(Px, Py, in, F0, FFTW_FORWARD, FFTW_ESTIMATE);
    #ifdef _OPENMP
	omp_unset_lock(&my_lock[8]);
	#endif

    fftw_execute(plan);

    #ifdef _OPENMP
	omp_set_lock(&my_lock[9]);
	#endif
    fftw_destroy_plan(plan);
    #ifdef _OPENMP
	omp_unset_lock(&my_lock[9]);
	#endif

	if (i_image == N_images-1)
	{
		fftw_free(in);
		free(p0);
	}
}


/* ------------------------------------------------ */


void crop_image_centered(int Nx, int Ny,
                         int Px, int Py,
                         float *padded,
                         float *out)
{
    int x0 = Px/2 - Nx/2;
    int y0 = Py/2 - Ny/2;

    for (int x = 0; x < Nx; x++) {
        for (int y = 0; y < Ny; y++) {
            out[x*Ny + y] =
                padded[(x + x0)*Py + (y + y0)];
        }
    }
}


/* ------------------------------------------------ */

void fits_error(int status)
{
    if (status) {
        fits_report_error(stderr, status);
        exit(EXIT_FAILURE);
    }
}

/* ------------------------------------------------ */


void sigma_clipping(float *image, long plane_pixels, double Nsigma, double *p0, double *sgm, long *Npix, int *k)
// Doing sigma-cliipping for the input monochrome image. Outputs: offset p0, std sgm, number of good pixels Npix.
{
  *p0 = 0.0;
  *sgm = 1e12;
  *Npix = -1;
  long Npix_old = -2;
  *k = 0;

  while (*Npix != Npix_old)
  {

    *k = *k + 1;
    Npix_old = *Npix;
    double sum = 0.0;
    double sum2 = 0.0;
    *Npix = 0;

    for (long i = 0; i < plane_pixels; i++)
    {
		   double p = image[i];
           if (fabs(p - *p0) < Nsigma * *sgm)
             {
              sum += p;
              sum2 += p * p;
              (*Npix)++;
             }
	}
	
    *p0 = sum / *Npix;
    *sgm = sqrt(sum2 / *Npix - *p0 * *p0);	

  }

	// Bias subtraction
    for (long i = 0; i < plane_pixels; i++)
    {
		image[i] = image[i] - *p0;
	}  

  return;
}


/* ------------------------------------------------ */

  float scaling (float *image1, float *master, long plane_pixels, double p1_low, int *k)
  {
	  // Computing the scaling between the convolved master image and the individual image1.
	  // Both images need their bias removed in advance

	// Counting the number of pixels to be used for scaling computations:
	long Nj = 0;
	double p1_max = -1e30;
    for (long i = 0; i < plane_pixels; i++)
	{
		if (image1[i] > p1_low)
			Nj++;
		if (image1[i] > p1_max)
			p1_max = image1[i];
	}
	  
	double *X = (double *)malloc(Nj*sizeof(double));
	double *Y = (double *)malloc(Nj*sizeof(double));
	double *W = (double *)malloc(Nj*sizeof(double));

	long j = -1;
    for (long i = 0; i < plane_pixels; i++)
	{
		// Pixels in image1 dimmer than p1_low are not used for scaling computations:
		if (image1[i] > p1_low)
		{
			j++;
			X[j] = master[i];
			Y[j] = image1[i];
			// Scaling will use this weight; going linearly from 0 at image1=p1_low,
			// to 1 at image1=p1_max.
			// This is to minimize the impact of the noisiest and by far the most numerous
			// dim pixels.
			W[j] = (Y[j]-p1_low) / (p1_max-p1_low);
			//W[j] = 1.0;
		}
	}

	// Finding scaling S
	double S = 1.0;
	double sgm = 1e12;
	long Npix = -1;
	long Npix_old;
	*k=0;
	// 3-sigma clipping
	do
	{
		*k = *k + 1;
		double sum = 0.0;
		double sum2 = 0.0;
		double sumW = 0.0;
		Npix_old = Npix;
		Npix = 0;
		for (long j=0; j<Nj; j++)
		{
			double Sj = log10(Y[j]/X[j]);  // Explicitly using the fact that both images should have bias=0
			if (fabs(Sj - S) < 3*sgm)
			{
				sum = sum + W[j]*Sj;
				sum2 = sum2 + W[j]*Sj*Sj;
				sumW = sumW + W[j];		
				Npix++;
			}
		}
		
		S = sum / sumW;
        sgm = sqrt(sum2 / sumW - S*S);	
	} 
	while(Npix != Npix_old);
	  
	return pow(10.0,S);
  }

/* ------------------------------------------------ */

	void gauss_blur(int i_image, int N_images, int Nx, int Ny, float* img, float *img_out, float sgm)
	// Applying gaussian blur to img, with sgm radius
	// i_image: image index
	// N_images: number of images in the sequence
	{
		static float *G;
		static float *R;
		static fftw_complex *FG;
		static fftw_complex *FI;
		
		int Pad = (int)(10*sgm);
		int Px = Nx + 2*Pad;
		int Py = Ny + 2*Pad;
		long P = Px * Py;		
		long N = Nx * Ny;

		if (i_image == 0)
		{
			// Fill an image with a Gaussian, using circular shifts and padding:
			G = (float *)malloc(sizeof(float) * P);
			double sum = 0.0;
			double sgm2 = sgm*sgm;
			double cutoff2 = (double)(Pad*Pad);
			for (int x=0; x<Px; x++)
			{
				int dx = (x < Px / 2) ? x : x - Px;
				for (int y=0; y<Py; y++)
				{
					int dy = (y < Py / 2) ? y : y - Py;
					double r2 = (double)(dx*dx + dy*dy);
					if (r2 < cutoff2)
					{
						double val = exp(-r2 / (2.0 * sgm2));
						G[x*Py + y] = val;
						sum += val;
					}
					else
						G[x*Py + y] = 0.0;
				}
			}
			
			for (long i = 0; i < P; i++)
				G[i] /= sum;


			FG = fftw_malloc(sizeof(fftw_complex)*P);
			FI = fftw_malloc(sizeof(fftw_complex)*P);
			fft_image(Px, Py, G, FG);					
			R = (float *)malloc(sizeof(float) * P);
		}

		
		
		fft_images_padded(i_image, N_images, Nx, Ny, Px, Py, img, FI, Pad);
		
		convolve_image(i_image, N_images, Px, Py, FG, FI, R);
		
		// Cropping the convolved result, storing in img_out:
		crop_image_centered(Nx, Ny, Px, Py, R, img_out);
		

		if (i_image == N_images-1)
		{
			free(G);
			free(R);
			fftw_free(FG);
			fftw_free(FI);
		}
		
		return;		
	}


/* ------------------------------------------------ */


	
	void dump_fits (int Nx, int Ny, int Nc, float *img, const char *name)
	// Dump a 2D image into a FITS file (for debugging)
	{
		char buffer[50];
		int status=0; 
		float bias = 0.03;
		
		long Npixels = Nx*Ny;
		
		float *img1 = (float *)malloc(Npixels*sizeof(float));
		
		
		for (long i = 0; i < Npixels; i++)
		{
			img1[i] = img[i] + bias;
		}
		
		sprintf(buffer, "rm -f %s >/dev/null", name);
		if (system(buffer))
			printf("Could not delete the file %s\n", name);
		fitsfile *fk;
		fits_create_file(&fk, name, &status);
		fits_error(status);
		long nelem1  = (long)Nx * Ny * Nc;
//		long naxes[3] = {Ny, Nx, Nc};
//		fits_create_img(fk, FLOAT_IMG, 3, naxes, &status);
		long naxes[2] = {Ny, Nx};
		fits_create_img(fk, FLOAT_IMG, 2, naxes, &status);
		fits_error(status);
		long fpixel = 1;
		fits_write_img(fk, TFLOAT, fpixel, nelem1, img1, &status);
		fits_error(status);
		fits_close_file(fk, &status);		
		
		free(img1);
		
		return;
	}
	

/* ------------------------------------------------ */

