/* ADIS: Asteroid Discovery in Image Sequences

  This program generates synthetic (artificial) image sequence, with noise and a fake "asteroid".
  
  On Ubuntu:
  
  sudo apt install libgsl-dev gsl-bin
  gcc synthetic_data.c -lgsl -lgslcblas -lfftw3 -lcfitsio -lm -o synthetic_data

*/



#include "reconv.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

int main(int argc, char **argv)
{
	int status;
	fitsfile *f0;
	double mjd_last = 0.0;
    long naxes[3];
	int naxis;
	double mjd0 = 0.0;
	char fits_name[100];
	char buffer[120];

	// Initializing the GSL random number generator
	gsl_rng *r;
    r = gsl_rng_alloc(gsl_rng_mt19937);
	gsl_rng_set(r, 1);
	
	int x0 = 1000;
	int y0 = 700;
	float Jx = 300;
	float Jy = -200;
	float sgm = 1e-4;
	float bias = 0.003;
	float snr = 10.0;
	float FWHM = 7.7;
	int synth_bg = 1;
	
	float sgm_blur = FWHM / sqrt(8*log(2.0));
	
	int N_images = argc - 1;
	int j0 = 1;
	
	// Last image
	char *file0 = argv[argc-1];
	fits_open_file(&f0, file0, READONLY, &status);
	fits_error(status);
	mjd_last = MJD_FITS(f0); // DATE-OBS for the last image
	fits_get_img_dim(f0, &naxis, &status);
	fits_error(status);
	fits_get_img_size(f0, naxis, naxes, &status);
	fits_error(status);
    int Nx = naxes[1]; // X axis is vertical
    int Ny = naxes[0]; // Y axis is horizontal
    int Nc = 3;
	long Npix = Nx * Ny;
	long nelem1  = 3*Npix;
	fits_close_file(f0, &status);		
	fits_error(status);
	
	float *img = (float *)malloc(3*Npix*sizeof(float));
	
	printf("\n");
	
	for (int i_image=0; i_image<N_images; i_image++)
	{

		// Reading the next input FITS file
		char *file0 = argv[j0+i_image];
		fits_open_file(&f0, file0, READONLY, &status);
		fits_error(status);
		double mjd = MJD_FITS(f0); // DATE-OBS
		if (i_image == 0)
			mjd0 = mjd;

		// Replacing background with Gassian noise if requested:
		if (synth_bg)
			for (long i = 0; i < Npix; i++)
			{
				double value = bias + gsl_ran_gaussian(r, (double)sgm);
				img[i] = value;
				img[Npix + i] = value;
				img[2*Npix + i] = value;
			}
			else
			{
				fits_read_img(f0, TFLOAT, 1, nelem1, NULL, img, NULL, &status);
				fits_error(status);
			}
	
		// Coordinates of the fake asteroid's center:
		float x_ast = x0 + (mjd-mjd0)/(mjd_last-mjd0) * Jx;
		float y_ast = y0 + (mjd-mjd0)/(mjd_last-mjd0) * Jy;
		
		printf("Image %s: x=%f, y=%f\n", file0, x_ast, y_ast);
		
		int x1 = x_ast - 6*sgm_blur;
		int x2 = x_ast + 6*sgm_blur;
		int y1 = y_ast - 6*sgm_blur;
		int y2 = y_ast + 6*sgm_blur;
		if (x1 < 0)
			x1 = 0;
		if (x2 >= Nx)
			x2 = Nx-1;
		if (y1 < 0)
			y1 = 0;
		if (y2 >= Ny)
			y2 = Ny-1;
		
		// Fake asteroid is a Gaussian truncated at 6*sgm
		for (int x=x1; x<=x2; x++)
		{
			for (int y=y1; y<=y2; y++)
			{
				// Pixel's distance from the asteroid's center:
				float r = sqrt(pow((x-x_ast),2) + pow((y-y_ast),2));
				if (r <= 6*sgm_blur)
				{
					// At the center, the SNR for the asteroid is snr, with given sgm
					float signal = snr*sgm*exp(-0.5*pow((r/sgm_blur),2));
					img[x*Ny+y] = img[x*Ny+y] + signal;
					img[Npix + x*Ny+y] = img[Npix + x*Ny+y] + signal;
					img[2*Npix + x*Ny+y] = img[2*Npix + x*Ny+y] + signal;
				}
			}
		}
		
	// Writing the output FITS file
	sprintf(fits_name,"synth_%s", file0);	
	sprintf(buffer, "rm -f %s >/dev/null", fits_name);
	if (system(buffer))
		printf("Could not delete the file %s\n", fits_name);
	fitsfile *fk;		
	fits_create_file(&fk, fits_name, &status);
	fits_error(status);
	// Copying the whole header from the input image to the output one:
	fits_copy_header(f0, fk, &status);
	fits_error(status);	
	long fpixel = 1;
	fits_write_img(fk, TFLOAT, fpixel, nelem1, img, &status);
	fits_error(status);
	fits_close_file(fk, &status);		
	fits_close_file(f0, &status);					
	}  // Images loop
	


	gsl_rng_free(r);

	return 0;
}