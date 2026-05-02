#include "reconv.h"
#include "asteroid_search.h"

// CUDA related functions


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
	
__global__ void motion_search_cuda (float **d_image, int N_images, size_t pitch, int Ix1, int Iy1, int Jx, int Jy, float MQ, float p_min, float *d_dt, float *d_test_image, struct List *d_list, int save_image, unsigned int *d_Pixel_counter, int Nx, int Ny)
{
	// The main compute kernel: searches for moving objects over all images,
	// all allowed base image tiles, for the given motion vector (Jx,Jy) in MQ units
	// Using the maximum 1024 threads per block, to cover 32x32 pixel tiles
	// Each kernel performs one full image stacking, over all base tiles.
	
	__shared__ float base_tile[NB][NB];  // Keeps the base tile pixels
	__shared__ float image_tile[NB+1][NB+2]; // Keeps tiles of all the rest of images; +1 columns to avoid bank conflicts (last column will not be used)
	
	// Base tile indexes:
	// Change in the interval Ix1..Ix2, Iy1..Iy2 (inclusive)
	int Ixb = blockIdx.x + Ix1;
	int Iyb = blockIdx.y + Iy1;

	// Coordinates of the pixels within the tile:
	int jx = threadIdx.y;
	int jy = threadIdx.x; // Fastest changing index

	// Base image pixel coordinates:
	int ix = jx + NB*Ixb;
	int iy = jy + NB*Iyb;  // Fastest changing index
	
	int counter = 0;
		
	// Loping over all images sequentially:
	for (int i=0; i<N_images; i++)
	{
		// Motion vector for the image i in pixels:
		float dx = d_dt[i] * Jx*MQ;
		float dy = d_dt[i] * Jy*MQ;
		// Pixel coordinates of the base transposed pixel:
		int ix0 = ix + floor(dx);
		int iy0 = iy + floor(dy); // Fastest changing index
		// Fractional pixel shift (>=0):
		float fdx = dx - floor(dx);
		float fdy = dy - floor(dy);		

		if (i==0 && threadIdx.x<NB && threadIdx.y<NB)
		{
			// Base 31x31 tile initialization
			// For the base tile only, skipping the last column and last row, because there is no interpolation:
			// Pointer to the start of the image's row:
			if (ix>=0 && ix<Nx && iy>=0 && iy<Ny)
			{
				float *row = (float *)((char*)d_image[i] + ix * pitch);
				float p = row[iy];  // Coalesced read
				if (p > MASK)
				{
					base_tile[jx][jy] = p;
					counter++;
				}
				else
					base_tile[jx][jy] = 0.0;
			}
//			else
//				printf("base: %d %d %d %d\n", ix, iy, Jx, Jy);
		}
		
		if (i > 0)
		// 32x32 image_tile initialization
		{
			// Pointer to the start of the image's row:
			if (ix0>=0 && ix0<Nx && iy0>=0 && iy0<Ny)
			{
				float *row = (float *)((char*)d_image[i] + ix0 * pitch);
				// Memory bug!!!:
				image_tile[jx][jy] = row[iy0]; // Coalesced read
			}
//			else
//				printf("reg: %d %d %e %e %d %d\n", ix0, iy0, dx, dy, Jx, Jy);
		}
		
		__syncthreads(); // Required because of the tile initialization by all threads ^^
		
		if (i > 0 && threadIdx.x<NB && threadIdx.y<NB)
		// Linear pixel interpolation:
		{
			if (image_tile[jx][jy]>MASK && image_tile[jx][jy+1]>MASK && image_tile[jx+1][jy]>MASK && 	image_tile[jx+1][jy+1]>MASK)
			{
				base_tile[jx][jy] += 
					image_tile[jx][jy]     * (1.0-fdx) * (1.0-fdy) +
					image_tile[jx][jy+1]   * (1.0-fdx) *      fdy  +
					image_tile[jx+1][jy]   *      fdx  * (1.0-fdy) +
					image_tile[jx+1][jy+1] *      fdx  *      fdy ;											
				counter++;
			}
		}
		
		__syncthreads(); 		
	}
	
	// Normalizing the pixel value, only using non-masked pixels:
	if (threadIdx.x<NB && threadIdx.y<NB)
		if (counter > 0)
			base_tile[jx][jy] = base_tile[jx][jy] / counter;
		else
			base_tile[jx][jy] = MASK0;


	if (save_image==0 && threadIdx.x<NB && threadIdx.y<NB)
		// Saving brighest pixels (>p_min) in the image stack
		if (base_tile[jx][jy] > p_min)
		{
			int ii = atomicInc(d_Pixel_counter, MAX_PIXELS);
			if (ii < MAX_PIXELS)
			{
				d_list[ii].Jx = Jx;
				d_list[ii].Jy = Jy;
				d_list[ii].ix = ix;
				d_list[ii].iy = iy;
				d_list[ii].p = base_tile[jx][jy];
			}
		}
	
	if (save_image)
		if (threadIdx.x<NB && threadIdx.y<NB)
		{
			float *row = (float *)((char*)d_test_image + ix * pitch);
			row[iy] = base_tile[jx][jy]; // Coalesced write
		}

	return;
}




/* ------------------------------------------------ */

	
__global__ void subtract_master_image (float **d_image, int N_images, size_t pitch, int Ix1, int Iy1, float *master_image, int Nx, int Ny)
{
	
	__shared__ float master_tile[32][33];
	__shared__ float image_tile[32][33];
	
	// tile indexes:
	int Ixb = blockIdx.x;
	int Iyb = blockIdx.y;

	// Coordinates of the pixels within the tile:
	int jx = threadIdx.y;
	int jy = threadIdx.x; // Fastest changing index

	// image pixel coordinates:
	int ix = jx + 32*Ixb;
	int iy = jy + 32*Iyb;  // Fastest changing index
	
	if (ix >= Nx || iy >= Ny)
		return;

	// Reading the master image:
	float *row = (float *)((char*)master_image + ix * pitch);
	master_tile[jx][jy] = row[iy] / N_images; // Coalesced read
	
	__syncthreads();
		
	// Loping over all images sequentially:
	for (int i=0; i<N_images; i++)
	{	
		float *row = (float *)((char*)d_image[i] + ix * pitch);
		image_tile[jx][jy] = row[iy]; // Coalesced read

		__syncthreads();
		
		// Subtracting the master image from every image, saving it back:
		row[iy] = image_tile[jx][jy] - master_tile[jx][jy];
		
		__syncthreads(); 		
	}

	return;
}




/* ------------------------------------------------ */



	
void find_kernel_parameters(int Jx, int Jy, float MQ, int Nx, int Ny, dim3 *Grid_size, int *Ix1, int *Iy1)
{
				
	// Motion vector in pixels:
	float jx = Jx*MQ;
	float jy = Jy*MQ;
	
	// Maximum range of base image pixels usable for this motion vector:
	// ix_min...ix_max, iy_min...iy_max (inclusive)
	int ix_min = 0;
	int ix_max = Nx - 1;
	int iy_min = 0;
	int iy_max = Ny - 1;
	if (jx < 0)
		ix_min = ceil(-jx);
	else
		ix_max = Nx - 1 - ceil(jx);
	if (jy < 0)
		iy_min = ceil(-jy);
	else
		iy_max = Ny - 1 - ceil(jy);
	
	// Now we can get the ranges for the base tile indexes (inclusive),
	// corresponding to the current motion vector:
	*Ix1 = (ix_min+NB-1) / NB;
	int Ix2 = (ix_max-NB+1) / NB;
	*Iy1 = (iy_min+NB-1) / NB;
	int Iy2 = (iy_max-NB+1) / NB;
	
	// To be double checked:
	if (Jx==0 && Jy==0)
	{
		Ix2++;
		Iy2++;
	}
					
	// The grid of blocks is for Ix and Iy parameters:
	*Grid_size = dim3(Ix2 - *Ix1 + 1, Iy2 - *Iy1 + 1);
	
	return;
}	
/* ------------------------------------------------ */
	void cluster_analysis(struct List *list, unsigned int Pixel_counter, int *Cluster_index)
	/* Carrying out cluster analysis in 4D on list.
	Output: Cluster_index vector containing cluster associations for each pixel.
	*/
	{
		int *members = (int *)malloc(Pixel_counter*sizeof(int));
		int *next_members = (int *)malloc(Pixel_counter*sizeof(int));
		
		for (int i=0; i<Pixel_counter; i++)
		{
			Cluster_index[i] = -1; // Initially no cluster is assigned
		}
		
		int i_max, del, pixel_is_neighbour, N_next;
		int counter = -1;
		
		// Outer (cluster) loop: 
		do
		{
			// Finding the brightest pixel for the next cluster
			float p_max = -1e30;
			i_max = -1;
			for (int i=0; i<Pixel_counter; i++)
			{
				float p = list[i].p;
				if (Cluster_index[i]==-1 && p>p_max)
				{
					p_max = p;
					i_max = i;
				}
			}
			
			if (i_max == -1)
				break;  // We ran out of unassigned pixels
			
			counter++;  // Incrementing the cluster counter
			Cluster_index[i_max] = counter;
			
			// Initial list of cluster members contains only the brightest pixel:
			members[0] = i_max;			
			int N_members = 1;
			
			// The while loop to go over iterations of members
			do
			{
				N_next = 0;
				// Finding all cluster members iteratively
				for (int i=0; i<Pixel_counter; i++)
				{
					if (Cluster_index[i] == -1)
					{
						pixel_is_neighbour = 0;
						for (int j=0; j<N_members; j++)
						{
							// Computing the closeness parameter
							int cl = 0;
							
							del = abs(list[members[j]].Jx-list[i].Jx);
							if (del < 2)
								cl = cl + del;
							else
								continue;
								
							del = abs(list[members[j]].Jy-list[i].Jy);
							if (del < 2)
								cl = cl + del;
							else
								continue;
								
							del = abs(list[members[j]].ix-list[i].ix);
							if (del < 2)
								cl = cl + del;
							else
								continue;
								
							del = abs(list[members[j]].iy-list[i].iy);
							if (del < 2)
								cl = cl + del;
							else
								continue;
								
							// Accepting the pixel as the new cluster member if it's close enough:
							if (cl>0 && cl <= CL_MAX)
							{
								pixel_is_neighbour = 1;
								break;
							}											
						}
						
						if (pixel_is_neighbour == 1)
						{							
							Cluster_index[i] = counter;
							N_next++;
							next_members[N_next-1] = i;												
						}										
					}
				}
				
				// Copying the next_members list to current members list:
				if (N_next > 0)
				{
					N_members = N_next;
					for (int j=0; j<N_members; j++)
					{
						members[j] = next_members[j];
					}
				}			
			}
			while(N_next > 0);
			
			printf("Found cluster %d\n", counter);
			
		}
		while(i_max != -1);
		
		printf("Found %d clusters\n", counter+1);

		return;
		
	}
/* ------------------------------------------------ */


