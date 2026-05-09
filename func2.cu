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
		if (counter > N_images*0.75)  // !!! 0
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

	void cluster_analysis_cuda(struct List *list, unsigned int Pixel_counter, int *h_cloud)
	// Carrying out the cluster analysis on GPU
	{
		// We need to move from a structure to simple arrays to achieve coalesced memory access on GPU:		
		int *h_ix;
		cudaMallocHost(&h_ix, Pixel_counter*sizeof(int));
		int *h_iy;
		cudaMallocHost(&h_iy, Pixel_counter*sizeof(int));
		int *h_Jx;
		cudaMallocHost(&h_Jx, Pixel_counter*sizeof(int));
		int *h_Jy;
		cudaMallocHost(&h_Jy, Pixel_counter*sizeof(int));

		int *d_ix;
		cudaMalloc(&d_ix, Pixel_counter*sizeof(int));
		int *d_iy;
		cudaMalloc(&d_iy, Pixel_counter*sizeof(int));
		int *d_Jx;
		cudaMalloc(&d_Jx, Pixel_counter*sizeof(int));
		int *d_Jy;
		cudaMalloc(&d_Jy, Pixel_counter*sizeof(int));
		int *d_cloud;
		cudaMalloc(&d_cloud, Pixel_counter*sizeof(int));
		int *d_members;
		cudaMalloc(&d_members, Pixel_counter*sizeof(int));
		
		cudaMemcpy(d_ix, h_ix, Pixel_counter*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_iy, h_iy, Pixel_counter*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Jx, h_Jx, Pixel_counter*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Jy, h_Jy, Pixel_counter*sizeof(int), cudaMemcpyHostToDevice);

		int N_members;
		unsigned int N_members1;
		int ii, jj;
		
		int Block_size;
		int N_cloud = 0;
		int Block_size0 = 512;
		if (Pixel_counter < Block_size0)
			Block_size = Pixel_counter;
		else
			Block_size = Block_size0;
		int N_blocks = (Pixel_counter+Block_size-1)/Block_size;

		
		do
		{
			// Pre-initializing i_free_pixel to -1
			ii = -1;
			cudaMemcpyToSymbol(i_free_pixel, &ii, sizeof(int), 0, cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();

			// Finding a non-assigned pixel from the list - the start of the new cloud
			// the result is in device variable i_free_pixel
			find_free_pixel<<<N_blocks,Block_size>>>(d_cloud, Pixel_counter);
			cudaMemcpyFromSymbol(&ii, i_free_pixel, sizeof(int), 0, cudaMemcpyDeviceToHost);
			if (ii == -1)
				// No more clusters
				break;
				
			// We are starting a new cloud (with one pixel for now, with the index ii)
			N_cloud++;
			N_members = 1;
			jj = 0;
			
			do
			// finding the cloud members iteratively
			{				
				N_members1 = 0;
				cudaMemcpyToSymbol(d_N_members1, &N_members1, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
				cudaDeviceSynchronize();

				// Searching for the neighbours of the current list of 4D pixels
				// Using one thread per d_members element (initially it's just one element)
				find_neighbours<<<N_blocks, Block_size>>>(jj, N_members, N_cloud, Pixel_counter, ii, d_ix, d_iy, d_Jx, d_Jy, d_cloud, d_members);

				cudaMemcpyFromSymbol(&N_members, d_N_members1, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);				
				jj++;
			}
			while (N_members > 0);
						
		}
		while (true);
		
		cudaMemcpy(h_cloud, d_cloud, Pixel_counter*sizeof(unsigned int), cudaMemcpyDeviceToHost);
		
		cudaFree(h_ix);
		cudaFree(h_iy);
		cudaFree(h_Jx);
		cudaFree(h_Jy);
		cudaFree(d_ix);
		cudaFree(d_iy);
		cudaFree(d_Jx);
		cudaFree(d_Jy);
		cudaFree(d_cloud);
		cudaFree(d_members);

		printf("Found %d clusters on GPU\n", N_cloud);

	}
/* ------------------------------------------------ */

__global__ void find_free_pixel(int *d_cloud, unsigned int Pixel_counter)
// The kernel to find a non-zero element in d_cloud, and store its index in the
// device variable i_free_pixel, which needs to be pre-initialized to -1.
{
	if (i_free_pixel != -1)
		return;
	
	int i_pixel = threadIdx.x + blockIdx.x*blockDim.x;
	
	if (i_pixel >= Pixel_counter)
		return;
	
	if (d_cloud[i_pixel] == 0)
	{
		atomicCAS(&i_free_pixel, -1, i_pixel);
	}
}

/* ------------------------------------------------ */

__global__ void find_neighbours(int jj, int N_members, int N_cloud, unsigned int Pixel_counter, int ii, int *d_ix, int *d_iy, int *d_Jx, int *d_Jy, int *d_cloud, int *d_members)
/* Identifying cloud memebers based on the freshly found cloud members in d_members.
   Each thread processes a different element of the d_cloud list.
*/
{
	// Global pixel index for the d_cloud vector
	int i = threadIdx.x + blockIdx.x*blockDim.x;

	if (i >= Pixel_counter)
		return;

	// Skipping the pixels which have already been assigned to a cloud:
	if (d_cloud[i] != -1)
		return;

	int ix = d_ix[i];
	int iy = d_iy[i];
	int Jx = d_Jx[i];
	int Jy = d_Jy[i];	

	bool inCloud = false;
	int i0;

	// Loop over all freshely added members:
	for (int j=0; j<N_members; j++)
	{
		if (jj == 0)
			// We just starting a new cloud, with only one member provided by ii index
			i0 = ii;
		else
			i0 = d_members[j];
		
		int dx  = abs(d_ix[i0] - ix);
		int dy  = abs(d_iy[i0] - iy);
		int dJx = abs(d_Jx[i0] - Jx);
		int dJy = abs(d_Jy[i0] - Jy);
		
		// Preliminary closeness test:
		if (dx<=1 && dy<=1 && dJx<=1 && dJy<=1)
		{
			// computing the closeness index
			int S = dx + dy + dJx + dJy;
			// Cluster membership criterion:
			if (S == 1)
				// S==1 means that only one coordinate differs by 1: no diagonal members
			{
				// This pixel belongs to the current cloud
				inCloud = true;
				break;
			}
		}
	}
	
	if (inCloud)
		// This pixel is the new member of the cloud
	{
		d_cloud[i] = N_cloud;
		int i1 = atomicInc(&d_N_members1, MAX_PIXELS);
		d_members[i1] = i;
	}
}


/* ------------------------------------------------ */

/*
__global__ void find_free_pixel(int *d_cloud, unsigned int Pixel_counter)
// In the device vector d_cloud, find the first (smallest index) zero element (with no cloud assigned)
// The result will be in i_free_pixel (should be initialized)
{
	__shared__ int warp_results[32];
	__shared__ s_cloud[1024];
	
	int i_pixel = threadIdx.x + blockIdx.x*blockDim.x;
	if (i_pixel >= Pixel_counter)
		return;
	
	s_cloud[threadIdx.x] = d_cloud[i_pixel];
	
	// Step 1; all warps participate. Within each warp, find the first zero-value element, store 
	// in a shared vector (one lement per warp).
    int warpId   = threadIdx.x / 32;

	// Each lane checks its element (out-of-bounds treated as non-zero)
	// The first condition is for a partial warp situation
	bool isZero = (i_pixel < Pixel_counter) && (s_cloud[threadIdx.x] == 0);

	// Collect results across all 32 lanes into a single 32-bit mask
	unsigned int mask = __ballot_sync(0xFFFFFFFF, isZero);

	// We are storing the thread index corresponding to the first zero element in this warp, to warp_results[warpId]
	if (mask != 0) 
		// __ffs returns 1-based index of lowest set bit
		warp_results[warpId] = warpId * 32 + __ffs(mask) - 1;
	else
		warp_results[warpId] = -1;
	
	__synthreads();
	
	// Step 2. Now that all warps produced one result, we find the smallest index between all these results
	// Only warp 0 should participate
	if (warpId == 0)
	{
		int val = warp_results[threadIdx.x];
		// This loop will go through 5 values for offset: 16, 8, 4, 2, 1
		for (int offset = 16; offset > 0; offset >>= 1) 
		{
			int other = __shfl_down_sync(0xFFFFFFFF, val, offset);
			if (other != -1)
				if (val == -1)
					val = other;
				else
					val = min(val, other);
		}
		
		// The 0 thread contains the result
		// (either the first zero element's index, or -1 if there are no 0 elements in this block)
		if (threadIdx.x==0 && val!=-1)
			// !!! Don't forget to initialize i_free_pixel=MAX_PIXELS !
			atomicMin(i_free_pixel,val);
	}
	
	return;
}
*/


/* ------------------------------------------------ */
