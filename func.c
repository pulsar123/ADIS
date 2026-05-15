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
	#pragma omp threadprivate(Fout)

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
	{
        out[i] = Fout[i][0] * norm;
	}

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
	#pragma omp threadprivate(p0,in)
	
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
           if (fabs(p - *p0) < Nsigma * *sgm && p>MASK)
             {
              sum += p;
              sum2 += p * p;
              (*Npix)++;
             }
	}
	
	if (*Npix > 0)
	{
		*p0 = sum / *Npix;
		*sgm = sqrt(sum2 / *Npix - *p0 * *p0);	
	}
	else
	{
		*p0 = 0;
		*sgm = 1;
	}

  }

	// Bias subtraction
    for (long i = 0; i < plane_pixels; i++)
    {
		if (image[i] > MASK)
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
		#pragma omp threadprivate(G,R,FG,FI)
		
		int Pad = (int)(10*sgm);
		int Px = Nx + 2*Pad;
		int Py = Ny + 2*Pad;
		long P = Px * Py;		

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
		char buffer[100];
		int status=0; 
		float bias = 0.03;
		
		long Npixels = Nx*Ny;
		
		float *img1 = (float *)malloc(Npixels*sizeof(float));
		
		
		for (long i = 0; i < Npixels; i++)
		{
			if (img[i] > MASK)
				img1[i] = img[i] + bias;
			else
				img1[i] = bias;
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

		sprintf(buffer, "1500:800:1600:900:-1.0:Cloud 0:");
		fits_write_key(fk, TSTRING, "ANNOTATE", buffer, NULL, &status);

		long fpixel = 1;
		fits_write_img(fk, TFLOAT, fpixel, nelem1, img1, &status);
		fits_error(status);
		fits_close_file(fk, &status);		
		
		free(img1);
		
		return;
	}
	

/* ------------------------------------------------ */


void image_bw(float *image, long Npix, int Nc)
{
    for (long i = 0; i < Npix; i++)
    {
		float r = image[i];
		float g = image[Npix + i];
		float b = image[2*Npix + i];
		if (r>MASK && g>MASK && b>MASK)
			// The colors are RGB; storing the B&W image into the red channel
			image[i] = 0.25*(r + 2*g + b);
			else
			image[i] = MASK0;
	}

	return;
}
/* ------------------------------------------------ */

void crop(float *img, int *Nx, int *Ny, long *Npix, float crop_fraction)
{
	if (crop_fraction <= 0.0 || crop_fraction >= 1.0)
		return;
		
	// width of the crop area:
	int dx = (int)(*Nx*(1.0-crop_fraction)/2.0+0.5);
	int dy = (int)(*Ny*(1.0-crop_fraction)/2.0+0.5);
	printf("Crop width: %d x %d\n", dy, dx);
	
	if (dx==0 && dy==0)
		return;
	
	for (int ix=dx; ix<*Nx-dx; ix++)
		for (int iy=dy; iy<*Ny-dy; iy++)
		{
			long i_old = ix* *Ny + iy;
			long i_new = (ix-dx)* (*Ny-2*dy) + iy-dy;
			img[i_new] = img[i_old];
		}
		
	*Nx = *Nx - 2*dx;
	*Ny = *Ny - 2*dy;
	*Npix = *Nx * *Ny;
	
	return;
}



/* ------------------------------------------------ */


void rebin(int i_image, float *buf0, int *Nx, int *Ny, long *Npix, float** h_image)
{
	if (i_image == 0)
	{
		// We use cudaMallocHost instead of malloc to put the array in the pinned host memory
//		ERR( cudaMallocHost((void**)h_image, sizeof(float)* *Npix) )
	}
	
	for (long i=0; i<*Npix; i++)
	{
		(*h_image)[i] = buf0[i];
	}
	
	return;
	
}
/* ------------------------------------------------ */

	void subtract_background(int i_image, int N_images, float *img, int Nx, int Ny, int NTx, int NTy, float bias)
	/*  Subtracting background from an image defined on NTx x NTy tiles (not exactly square), 
	using bilinear interpolation. Within each tile, the background level is estimated using
	3-sigma clipping algorithm.
		
	The background array G has the dimensions NTx+2 x NTy+2 (extra border tiles on all 4 sides,
	to make bilinear interpolation seamless).
	
	The tiles corresponding to the actual image have the indices 1...NTx, 1...NTy.
	
	The border tiles have the same values as the adjacent real image tiles. The four border corner
	tiles have the same values as the corresponding corner image tiles. (So each real image corner
	tile is surrounded by 3 border tiles with the same value as the image tile.)
	
	Each tile ix,iy covers this pixel range: jx=(int)((ix-1)*delta_x+0.5), ..., jx<(int)(ix*delta_x+0.5).
	(Same for jy.)
	
	Conversion from pixel index to tile index (also works for negative jx, jy):  
	
	ix = floor(jx/delta_x) + 1;
	iy = floor(jy/delta_y) + 1;
	
	Pixel coordinates of the center of each ix,iy tile (can be negative):
	
	jx0=floor((ix-0.5)*delta_x+0.5);
	jy0=floor((iy-0.5)*delta_y+0.5);
	
	Top left tile index from pixel coordinates:
	
	ix0 = floor(jx/delta_x+0.5);
	iy0 = floor(jy/delta_y+0.5);
	
	*/
	{
		static int NTiles;
		static float *B;
		static float delta_x;
		static float delta_y;
		
	if (i_image == 0)
	{
		NTiles = (NTx+2) * (NTy+2);
		B = (float *)malloc(NTiles*sizeof(float));
		
		delta_x = (float)Nx / (float)NTx;
		delta_y = (float)Ny / (float)NTy;
	}

	double sum_all = 0.0;
	double sum2_all = 0.0;
	long Npix_all = 0;
	double sum, sum2;

	// Computing background value for each tile using 3-sigma clipping
	// Border regions: ix=0, ix=NTx+1; iy=0, iy=NTy+1
	for (int ix=1; ix<NTx+1; ix++)
	{
		// Range of pixel coordinates corresponding to this tile:
		int jx1 = (int)((ix-1)*delta_x+0.5);
		int jx2 = (int)(ix*delta_x+0.5);
		for (int iy=1; iy<NTy+1; iy++)
			{
				int jy1 = (int)((iy-1)*delta_y+0.5);
				int jy2 = (int)(iy*delta_y+0.5);
				
				// 3-sigma clipping to compute the background level for the current tile ix,iy:
				double p0 = 0.0;
				double sgm = 1e12;
				long Npix = -1;
				long Npix_old = -2;
				while (Npix != Npix_old)
					{
						Npix_old = Npix;
						sum = 0.0;
						sum2 = 0.0;
						Npix = 0;
						// Cycling through all the pixels in the current tile:
						for (int jx=jx1; jx<jx2; jx++)
							for (int jy=jy1; jy<jy2; jy++)
								{
									double p = img[jx*Ny + jy];
									if (fabs(p - p0)<3*sgm && p>MASK)
										{
											sum += p;
											sum2 += p * p;
											Npix++;
										}								
								}
						p0 = sum / Npix;
						sgm = sqrt(sum2 / Npix - p0 * p0);	
					}  //while loop
					
				sum_all = sum_all + sum;
				sum2_all = sum2_all + sum2;
				Npix_all = Npix_all + Npix;
				
				B[ix*(NTy+2) + iy] = p0; // Memorizing the tile's background level
				
				// Filling out the border regions:
				if (ix==1)
					B[iy] = p0;
				if (ix==NTx)
					B[(NTx+1)*(NTy+2) + iy] = p0;
				if (iy==1)
					B[ix*(NTy+2)] = p0;
				if (iy==NTy)
					B[ix*(NTy+2) + NTy+1] = p0;
		
			}
		}  // double tile loop ix,iy
		
		// The four border corners:
		int ix0, iy0, i0, ix1, iy1, i1;
		// Top left:
		ix0 = 0; iy0 = 0; i0 = ix0*(NTy+2) + iy0;
		ix1 = 1; iy1 = 1; i1 = ix1*(NTy+2) + iy1;
		B[i0] = B[i1];
		// Top right:
		ix0 = 0; iy0 = NTy+1; i0 = ix0*(NTy+2) + iy0;
		ix1 = 1; iy1 = NTy; i1 = ix1*(NTy+2) + iy1;
		B[i0] = B[i1];
		// Bottom left:
		ix0 = NTx+1; iy0 = 0; i0 = ix0*(NTy+2) + iy0;
		ix1 = NTx; iy1 = 1; i1 = ix1*(NTy+2) + iy1;
		B[i0] = B[i1];
		// Bottom right:
		ix0 = NTx+1; iy0 = NTy+1; i0 = ix0*(NTy+2) + iy0;
		ix1 = NTx; iy1 = NTy; i1 = ix1*(NTy+2) + iy1;
		B[i0] = B[i1];
		

		// Subtracting the background
		// (Bilinear interpolation model for background)
		for (int jx=0; jx<Nx; jx++)
		{
			// Top left tile index:
			int ix0 = floor(jx/delta_x+0.5);
			int ix1 = ix0 + 1;
			// Pixel coordinate of the tile's center:
			int jx0=floor((ix0-0.5)*delta_x+0.5);
			// Center of the next tile:
			int jx1=floor((ix0+0.5)*delta_x+0.5);
			for (int jy=0; jy<Ny; jy++)
			{
				int iy0 = floor(jy/delta_y+0.5);
				int iy1 = iy0 + 1;
				int jy0=floor((iy0-0.5)*delta_y+0.5);
				int jy1=floor((iy0+0.5)*delta_y+0.5);
				
				// Linear interpolation along X axis (downwards):
				float B0 = ((float)(jx-jx0))/((float)(jx1-jx0)) * (B[ix1*(NTy+2)+iy0] - B[ix0*(NTy+2)+iy0]) + B[ix0*(NTy+2)+iy0];
				float B1 = ((float)(jx-jx0))/((float)(jx1-jx0)) * (B[ix1*(NTy+2)+iy1] - B[ix0*(NTy+2)+iy1]) + B[ix0*(NTy+2)+iy1];
				
				// Linear interpolation along Y axis (horizontal) 
				// background value corresponding to the current pixel jx,jy
				float Bp = ((float)(jy-jy0))/((float)(jy1-jy0)) * (B1-B0) + B0;
				
				if (img[jx*Ny+jy] > MASK)
					// Subtracting the bilinear interpolated value of the background:
					img[jx*Ny+jy] = img[jx*Ny+jy] - Bp + bias;
			}
		}
		
	if (i_image == N_images-1)
	{
		free(B);
	}

		return;
	}

/* ------------------------------------------------ */
	
int date2mjd (int yr, int mn, int dy) {
	// Gregorian date -> Modified Julian Date (https://github.com/mdwarfgeek/lib/blob/master/mjd.c)
  int a, m, rv;

  a = yr - (12 - mn) / 10;
  m = (mn + 9) % 12;

  rv  = (1461 * (a + 4712)) / 4;
  rv += (306 * m + 5) / 10;
  rv += dy - 2399904;
  rv -= (3 * ((a + 4900) / 100)) / 4;

  return(rv);
}

/* ------------------------------------------------ */

void compute_histogram(float *image, long Npix, float sgm, float *p_min_std, long *hist)
{
	// Initializing hist
	for (int j=BIN_MIN; j<=BIN_MAX; j++)
		hist[j-BIN_MIN] = 0;
	
	// Computing the histogram
	for (long i=0; i<Npix; i++)
	{
		if (image[i] > MASK)
		{
			int bin = (int)(image[i]/sgm / D_SGM);
	//		printf("%e %e %e\n", image[i], image[i]-p0, (image[i]-p0)/sgm);
			if (bin >= BIN_MIN && bin < BIN_MAX)
				hist[bin-BIN_MIN]++;
			else if (bin >= BIN_MAX)
				hist[bin-BIN_MIN]++;
		}
	}
	
/*
	for (int j=BIN_MIN; j<=BIN_MAX; j++)
		printf("%d %ld\n",j,hist[j-BIN_MIN]);
	exit(0);
	*/
	
	if (hist[BIN_MAX-BIN_MIN] > NPIX_MAX)
	{
		printf("hist > NPIX_MAX in compute_histogram!\n");
		printf("Increase BIN_MAX or NPIX_MAX\n");
		exit(1);
	}
		
	// Finding the critical bin value when Npixels <= Npix_max
	long Npixels = 0;
	int j0 = 0;
	for (int j=BIN_MAX; j>=BIN_MIN; j--)
	{
		long Npix1 = Npixels + hist[j-BIN_MIN];
		if (Npix1 > NPIX_MAX)
			break;
		Npixels = Npix1;
		j0 = j;
	}
	
	printf("Npixels=%ld\n",Npixels);
	*p_min_std = (j0+BIN_MIN)*D_SGM;
	
	return;
	
}

/* ------------------------------------------------ */

void borders(float *img, int Nx, int Ny, int BW)
{
		
	for (int x=0; x<Nx; x++)
	{
		for (int y=0; y<Ny; y++)
		{
			if (x<BW || x>=Nx-BW || y<BW || y>=Ny-BW)
				img[x*Ny+y] = MASK0;
		}
	}
	return;
}