#include "reconv.h"
#include "asteroid_search.h"


void Is_GPU_present()
{
  int devid, devcount;
  /* find number of device in current "context" */
  cudaGetDevice(&devid);
  /* find how many devices are available */
  if (cudaGetDeviceCount(&devcount) || devcount==0)
    {
      printf ("No CUDA devices!\n");
      exit (1);
    }
  else
    {
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties (&deviceProp, devid);
      printf ("Device count, devid: %d %d\n", devcount, devid);
      printf ("Device: %s\n", deviceProp.name);
      printf ("Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
      printf ("Maximum treads per block: %d\n", deviceProp.maxThreadsPerBlock);
      printf ("Maximum block grid dimensions: %d x %d x %d\n\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    }
  return;
}



/* ------------------------------------------------ */
/*
void fits_error(int status)
{
    if (status) {
        fits_report_error(stderr, status);
        exit(EXIT_FAILURE);
    }
}
*/
/* ------------------------------------------------ */

void image_bw(float *image, long Npix, int Nc)
{
    for (long i = 0; i < Npix; i++)
    {
		// The colors are RGB; storing the B&W image into the red channel
		image[i] = 0.25*(image[i] + 2*image[Npix + i] + image[2*Npix + i]);
	}

	return;
}
/* ------------------------------------------------ */


void crop(float *img, int *Nx, int *Ny, long *Npix, float crop_fraction)
{
	if (crop_fraction <= 0.0 || crop_fraction >= 1.0)
		return;
		
	// width of the crop area:
	int dx = int(*Nx*(1.0-crop_fraction)/2.0+0.5);
	int dy = int(*Ny*(1.0-crop_fraction)/2.0+0.5);
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
		ERR( cudaMallocHost((void**)h_image, sizeof(float)* *Npix) )
	}
	
	for (long i=0; i<*Npix; i++)
	{
		(*h_image)[i] = buf0[i];
	}
	
	return;
	
}
/* ------------------------------------------------ */

	void subtract_background(int i_image, float *img, int Nx, int Ny, int NTx, int NTy)
	/*  Subtracting background from an image defined on NTx x NTy tiles (not exactly square), 
	using bilinear interpolation. Within each tile, the background level is estimated using
	3-sigma clipping algorithm.
	
	We also normalize the brightness values by the global (averaged over all tiles) background std value.
	
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
//				printf("ix=%d, iy=%d\n", ix, iy);
				while (Npix != Npix_old)
					{
						Npix_old = Npix;
						double sum = 0.0;
						double sum2 = 0.0;
						Npix = 0;
						// Cycling through all the pixels in the current tile:
						for (int jx=jx1; jx<jx2; jx++)
							for (int jy=jy1; jy<jy2; jy++)
								{
									double p = img[jx*Ny + jy];
									if (fabs(p - p0) < 3*sgm)
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
		
/*  Printing the background matrix
		for (int ix=0; ix<NTx+2; ix++)
		{
			for (int iy=0; iy<NTy+2; iy++)
			{
				printf("%e ", B[ix*(NTy+2) + iy]);
			}
			printf("\n");
		}
*/		

		// Global (computed over all the tiles) bias and std:
		double p0_all = sum_all / Npix_all;
		double sgm_all = sqrt(sum2_all / Npix_all - p0_all * p0_all);	

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
				float B0 = float(jx-jx0)/float(jx1-jx0) * (B[ix1*(NTy+2)+iy0] - B[ix0*(NTy+2)+iy0]) + B[ix0*(NTy+2)+iy0];
				float B1 = float(jx-jx0)/float(jx1-jx0) * (B[ix1*(NTy+2)+iy1] - B[ix0*(NTy+2)+iy1]) + B[ix0*(NTy+2)+iy1];
				
				// Linear interpolation along Y axis (horizontal) 
				// background value corresponding to the current pixel jx,jy
				float Bp = float(jy-jy0)/float(jy1-jy0) * (B1-B0) + B0;
				
				// Subtracting the bilinear interpolated value of the background,
				// and normalzing it by sgm_all:
				img[jx*Ny+jy] = (img[jx*Ny+jy] - Bp) / sgm_all;
			}
		}
		
		return;
	}

/* ------------------------------------------------ */


	
	void dump_fits (int Nx, int Ny, int Nc, float *img, const char *name)
	// Dump a 2D image into a FITS file (for debugging)
	{
		int status=0; 
		
		fitsfile *fk;
		fits_create_file(&fk, name, &status);
		fits_error(status);
		long nelem1  = (long)Nx * Ny * Nc;
//		long naxes[3] = {Ny, Nx, Nc};
//		fits_create_img(fk, FLOAT_IMG, 3, naxes, &status);
		long naxes[2] = {Ny, Nx};
		fits_create_img(fk, FLOAT_IMG, 2, naxes, &status);
		fits_error(status);
		fits_write_img(fk, TFLOAT, 1, nelem1, img, &status);
		fits_error(status);
		fits_close_file(fk, &status);		
		
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
	
__global__ void motion_search_cuda (int Ix1, int Iy1, int Jx, int Jy, float MQ, float p_min)
{
	// The main compute kernel: searches for moving objects over all images,
	// all allowed base image tiles, for the given motion vector (Jx,Jy) in MQ units
	// Using the maximum 1024 threads per block, to cover 32x32 pixel tiles
	
	
	
	
	return;
}




/* ------------------------------------------------ */

	