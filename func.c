#include "reconv.h"

/* ------------------------------------------------ */


void fft_image(int Nx, int Ny,
                double *img0,
                fftw_complex *F0)
{
    fftw_complex *in = fftw_malloc(sizeof(fftw_complex)*Nx*Ny);
    fftw_plan p;

    /* Image0 */
    for (int i=0;i<Nx*Ny;i++) {
        in[i][0] = img0[i];
        in[i][1] = 0.0;
    }
    p = fftw_plan_dft_2d(Nx, Ny, in, F0, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);

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
	
    fftw_plan p = fftw_plan_dft_2d(Nx, Ny, K, k_spatial,
                                  FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);

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
    fftw_plan p = fftw_plan_dft_2d(Nx, Ny, k_spatial, K,
                                  FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
}

/* ------------------------------------------------ */

void convolve_image(int Nx, int Ny,
                    fftw_complex *F0,
                    fftw_complex *K,
                    double *out)
{
    fftw_complex *Fout = fftw_malloc(sizeof(fftw_complex)*Nx*Ny);

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
    fftw_plan p = fftw_plan_dft_2d(Nx, Ny, Fout, Fout,
                                  FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);

    double norm = 1.0 / (Nx * Ny);
    for (int i=0;i<Nx*Ny;i++)
        out[i] = Fout[i][0] * norm;

    fftw_free(Fout);
}

/* ------------------------------------------------ */


void fft_images_padded(int Nx, int Ny,
                       int Px, int Py,
                       double *img0,
                       fftw_complex *F0,
					   int Pad)					  
{
	// First pad the image with Pad width, then direct FFT -> F0
	
    int P = Px * Py;

    double *p0 = calloc(P, sizeof(double));

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

    fftw_complex *in = fftw_malloc(sizeof(fftw_complex)*P);
    fftw_plan plan;

    /* Image0 */
    for (int i=0;i<P;i++) {
        in[i][0] = p0[i];
        in[i][1] = 0.0;
    }
	// Direct 2D FFT transform in (padded img0 with zero imaginary part) -> F0.
    plan = fftw_plan_dft_2d(Px, Py, in, F0,
                            FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    fftw_free(in);
    free(p0);
}


/* ------------------------------------------------ */


void crop_image_centered(int Nx, int Ny,
                         int Px, int Py,
                         double *padded,
                         double *out)
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


void sigma_clipping(double *image, long plane_pixels, double Nsigma, double *p0, double *sgm, long *Npix)
// Doing sigma-cliipping for the input monochrome image. Outputs: offset p0, std sgm, number of good pixels Npix.
{
  *p0 = 0.0;
  *sgm = 1e12;
  *Npix = -1;
  long Npix_old = -2;
  int k = 0;

  if (verbose)
	printf("Sigma clipping:\n");

  while (*Npix != Npix_old)
  {

    k++;
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

	if (verbose)
		printf("%d %e %e %ld\n", k, *sgm, *p0, *Npix);
 
	// Bias subtraction
    for (long i = 0; i < plane_pixels; i++)
    {
		image[i] = image[i] - *p0;
	}  

  return;
}


/* ------------------------------------------------ */

  double scaling (double *image1, double *master, long plane_pixels, double p1_low)
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

	if (verbose)
		printf("Scaling:\n");
	
	// Finding scaling S
	double S = 1.0;
	double sgm = 1e12;
	long Npix = -1;
	long Npix_old;
	int k=0;
	// 3-sigma clipping
	do
	{
		k++;
		double sum = 0.0;
		double sum2 = 0.0;
		double sumW = 0.0;
		Npix_old = Npix;
		Npix = 0;
		for (long j=0; j<Nj; j++)
		{
			double Sj = Y[j]/X[j];  // Explicitly using the fact that both images should have bias=0
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

	if (verbose)
	    printf("%d %e %e %ld\n", k, sgm, S, Npix);
	  
	return S;
  }

/* ------------------------------------------------ */

	void gauss_blur(int Nx, int Ny, double* img, double sgm)
	// Applying gaussian blur to img, with sgm radius
	{
		long N = Nx * Ny;
		
		// Fill an image with a Gaussian:
		double *G = malloc(sizeof(double) * N);
		double sum = 0.0;
		double sgm2 = sgm*sgm;
		for (int x=0; x<Nx; x++)
		{
			int dx = (x <= Nx / 2) ? x : x - Nx;
			for (int y=0; y<Ny; y++)
			{
				int dy = (y <= Ny / 2) ? y : y - Ny;
				double r2 = (double)(dx*dx + dy*dy);
				if (r2 < 400.0)
				{
					double val = exp(-r2 / (2.0 * sgm2));
					G[x*Ny + y] = val;
					sum += val;
				}
				else
					G[x*Ny + y] = 0.0;
			}
		}
		
		for (long i = 0; i < N; i++)
			G[i] /= sum;

		fftw_complex *FG = fftw_malloc(sizeof(fftw_complex)*N);			
		fftw_complex *FI = fftw_malloc(sizeof(fftw_complex)*N);			
		
		fft_image(Nx, Ny, img, FI);		
		fft_image(Nx, Ny, G, FG);		
		
		convolve_image(Nx, Ny, FG, FI, img);
		
		free(G);
		fftw_free(FI);
		fftw_free(FG);
		
		return;		
	}



/* ------------------------------------------------ */
