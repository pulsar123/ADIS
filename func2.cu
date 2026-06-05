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

double MJD_FITS (fitsfile *f0)
// Computing the MJD (Modified Julian calendar Date) value (in days) corresponding to the middle
// of the exposure, based on the FITS file header.
{
	char date_obs[30];
	int year, month, day, hour, minute;
    double second, exposure;	
	int status = 0;
		
	fits_read_key(f0, TSTRING, "DATE-OBS", date_obs, NULL, &status);
	fits_error(status);

	fits_read_key(f0, TDOUBLE, "EXPTIME", &exposure, NULL, &status);
	fits_error(status);

	fits_str2time(date_obs, &year, &month, &day,
			  &hour, &minute, &second, &status);
	fits_error(status);
	
	 // Modified Julian Date (UT) of the middle of exposure
	double mjd = exposure/2.0/86400 + date2mjd (year, month, day) + ((second/60.0+minute)/60.0+hour)/24.0;
	
	return mjd;
}	

	
/* ------------------------------------------------ */
	
__global__ void motion_search_cuda (float **d_image, int N_images, size_t pitch, int Ix1, int Iy1, int Jx,
	int Jy, float MQ, float p_min, float *d_dt, float *d_test_image, List d_list, int save_image,
	unsigned int *d_Pixel_counter, int Nx, int Ny, int crop, int *d_dx_offset, int *d_dy_offset)
{
	// The main compute kernel: searches for moving objects over all images,
	// all allowed base image tiles, for the given motion vector (Jx,Jy) in MQ units
	// Using the maximum 1024 threads per block, to cover 32x32 pixel tiles
	// Each kernel performs one full image stacking, over all base tiles.
	
	// In crop mode, we pretend that we are still dealing with full resolution images, only when
	// reading from /writing to image arrays we convert to local (cropped) indexes using the per-image
	// offsets dx_offset, dy_offset. Input parameters Nx, Ny, and pitch correspond to cropped images.
	// Base tile indexes (Ix1, Iy1) and motion vectors (Jx, Jy) correspond to a full resolution image.
	
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
	// Full resolution, even in crop mode
	int ix = jx + NB*Ixb;
	int iy = jy + NB*Iyb;  // Fastest changing index

	int counter = 0;
		
	// Loping over all images sequentially:
	for (int i=0; i<N_images; i++)
	{
	
		float dx, dy;
		// Motion vector for the image i in pixels:
		dx = d_dt[i] * Jx*MQ;
		dy = d_dt[i] * Jy*MQ;

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
			int ixt, iyt;
			if (crop)
			{
				// Switching from full-res to cropped coordinates
				ixt = ix - d_dx_offset[0];
				iyt = iy - d_dy_offset[0];
			}
			else
			{
				ixt = ix;
				iyt = iy;
			}
			if (ixt>=0 && ixt<Nx && iyt>=0 && iyt<Ny)
			{
				float *row = (float *)((char*)d_image[i] + ixt * pitch);
				float p = row[iyt];  // Coalesced read
				if (p > MASK)
				{
					base_tile[jx][jy] = p;
					counter++;
				}
				else
					base_tile[jx][jy] = 0.0;
			}
		}
		
		if (i > 0)
		// 32x32 image_tile initialization
		{
			int ixt, iyt;
			if (crop)
			{
				ixt = ix0 - d_dx_offset[i];
				iyt = iy0 - d_dy_offset[i];
			}
			else
			{
				ixt = ix0;
				iyt = iy0;
			}
			// Pointer to the start of the image's row:
			if (ixt>=0 && ixt<Nx && iyt>=0 && iyt<Ny)
			{
				float *row = (float *)((char*)d_image[i] + ixt * pitch);
				image_tile[jx][jy] = row[iyt]; // Coalesced read
			}
			else
				image_tile[jx][jy] = MASK0;
		}
		
		__syncthreads(); // Required because of the tile initialization by all threads ^^
		
		if (i > 0 && threadIdx.x<NB && threadIdx.y<NB)
		// Linear pixel interpolation:
		{
			if (image_tile[jx][jy]>MASK && image_tile[jx][jy+1]>MASK && image_tile[jx+1][jy]>MASK && image_tile[jx+1][jy+1]>MASK)
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
		if (counter > N_images*0.75)  // !!! Only using the pixels where at least 75% of the images have a non-masked value
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
				d_list.Jx[ii] = Jx;
				d_list.Jy[ii] = Jy;
				d_list.ix[ii] = ix;
				d_list.iy[ii] = iy;
				d_list.p[ii] = base_tile[jx][jy];
			}
		}
	
	if (save_image)
		if (threadIdx.x<NB && threadIdx.y<NB)
		{
			if (crop)
			{
				// Switching from full-res to cropped coordinates
				ix = ix - d_dx_offset[0];
				iy = iy - d_dy_offset[0];
			}
			if (ix>=0 && ix<Nx && iy>=0 && iy<Ny)
			{
				float *row = (float *)((char*)d_test_image + ix * pitch);
				row[iy] = base_tile[jx][jy]; // Coalesced write
			}
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



	
void find_kernel_parameters(int Jx, int Jy, float MQ, int Nx, int Ny, dim3 *Grid_size, int *Ix1, int *Iy1, int crop, int X0, int Y0)
{
	// In crop mode,we always use the same base tiles, regardless of the motion vector
	// The cropped box spans these ranges of native base image pixels:
	// h_dx_offset[0]...h_dx_offset[0]+Nx-1
	// h_dy_offset[0]...h_dy_offset[0]+Ny-1
	// Convert these to the native base tile indexes by /NB
	if (crop)
	{
		// Starting tile indexes (could be partial):
		*Ix1 = X0 / NB;
		*Iy1 = Y0 / NB;
		// Ending tile indexes (could be partial):
		int Ix2 = (X0+Nx-1) / NB;
		int Iy2 = (Y0+Ny-1) / NB;
		// Number of tiles along each dimension:
		*Grid_size = dim3(Ix2-*Ix1+1,Iy2-*Iy1+1);
		return;
	}
				
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
	void cluster_analysis(List h_list, unsigned int Pixel_counter, int *Cluster_index, int *N_cloud)
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
				float p = h_list.p[i];
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
							
							del = abs(h_list.Jx[members[j]]-h_list.Jx[i]);
							if (del < 2)
								cl = cl + del;
							else
								continue;
								
							del = abs(h_list.Jy[members[j]]-h_list.Jy[i]);
							if (del < 2)
								cl = cl + del;
							else
								continue;
								
							del = abs(h_list.ix[members[j]]-h_list.ix[i]);
							if (del < 2)
								cl = cl + del;
							else
								continue;
								
							del = abs(h_list.iy[members[j]]-h_list.iy[i]);
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
			
//			printf("Found cluster %d\n", counter);
			
		}
		while(i_max != -1);
		
		printf("Found %d clusters\n", counter+1);
		*N_cloud = counter + 1;

		return;
		
	}
/* ------------------------------------------------ */

	void cluster_analysis_cuda(List d_list, unsigned int Pixel_counter, int *h_cloud, int *N_cloud)
	/* Carrying out the cluster analysis on GPU. The results (cloud ID for each pixel from the input d_list) are stored
	in h_cloud.
	*/
	{

		int *d_cloud;
		cudaMalloc(&d_cloud, Pixel_counter*sizeof(int));
		int *d_members;
		cudaMalloc(&d_members, Pixel_counter*sizeof(int));		
		// NN should be large enough to accomodate all intermediate results of the reduction:
		int NN = Pixel_counter/FIND_MAXIMUM_BS * 2;
		// The scratch vectors to store the results of the intermediate steps of the reduction:
		float *d_vec;
		cudaMalloc(&d_vec, NN*sizeof(float));
		int *d_index;
		cudaMalloc(&d_index, NN*sizeof(int));
		
		int N_members;
		unsigned int N_members1;
		int ii;
		
		int Block_size;
		int Block_size0 = FIND_MAXIMUM_BS;
		if (Pixel_counter < Block_size0)
			Block_size = Pixel_counter;
		else
			Block_size = Block_size0;
		int N_blocks = (Pixel_counter+Block_size-1)/Block_size;

		// Initially d_cloud elements are all set to 0 (no cloud association):
		init_d_cloud<<<N_blocks,Block_size>>>(d_cloud, Pixel_counter);

		int counter = -1;
		
		// Cloud loop
		do
		{
			counter++;

			// Finding the brightest unassigned pixel on GPU - it will be the start of a new cloud
			// This is tricky as it is done in parallel, using a binary tree reduction at the block level,
			// conisting with one or more iterative reduction kernel calls ("steps"), to handle vectors of an arbitrary length
			int step = 0;
			int offset0;
			int offset1;
			float *d_vec1;
			int N, N_blocks2, Block_size2;
			do
			{
				if (step == 0)
				{
					// During the first step, the source of the data (pixel brightness) is the input d_list.p vector:
					d_vec1 = d_list.p;
					N = Pixel_counter;
					Block_size2 = Block_size;
					N_blocks2 = N_blocks;
					offset0 = 0;
					offset1 = 0;
				}
				else
				{
					// For subsequent steps, the source of the data is a segment of the scratch vector d_vec, with indices offset0...offset1
					d_vec1 = &d_vec[offset0];
					N = N_blocks2; // Each block from the previous iteration becomes a thread in the new iteration
					offset1 = offset1 + N;
					if (N < Block_size0)
						Block_size2 = N;
					else
						Block_size2 = Block_size0;
					N_blocks2 = (N+Block_size2-1)/Block_size2; // Updated number of blocks for the current iteration
				}
				// One stage of a binary reduction:
				find_maximum<<<N_blocks2,Block_size2>>> (step, N, d_vec1, &d_index[offset0], d_cloud, &d_vec[offset1], &d_index[offset1], counter, d_members);
				
				step++;
				offset0 = offset1;
			}
			while(N_blocks2 > 1);
			
			// 0-th element of d_members was initialized above in the last find_maximum iteration
			cudaMemcpy(&ii, d_members, sizeof(int), cudaMemcpyDeviceToHost);
			if (ii == -1)
				// No more clusters
				{
					counter--;
					break;
				}

//			printf("Cloud %d, brightest pixel=%d\n", *N_cloud, ii);
				
			// We are starting a new cloud (with one pixel for now, with the index ii)
			N_members = 1;
			
			do
			// finding the cloud members iteratively
			{				
				N_members1 = 0;
				cudaMemcpyToSymbol(d_N_members1, &N_members1, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
				cudaDeviceSynchronize();

				// Searching for the neighbours of the current list of 4D pixels
				// Using one thread per d_members element (initially it's just one element)
				find_neighbours<<<N_blocks, Block_size>>>(N_members, counter, Pixel_counter, d_list, d_cloud, d_members);

				cudaMemcpyFromSymbol(&N_members, d_N_members1, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);				
			}
			while (N_members > 0);
						
		}
		while (true);
		
		cudaMemcpy(h_cloud, d_cloud, Pixel_counter*sizeof(unsigned int), cudaMemcpyDeviceToHost);
		
		cudaFree(d_cloud);
		cudaFree(d_members);
		cudaFree(d_vec);
		cudaFree(d_index);
		
		*N_cloud = counter + 1;

		printf("Found %d clusters on GPU\n", *N_cloud);

	}
/* ------------------------------------------------ */

__global__ void init_d_cloud(int *d_cloud, unsigned int Pixel_counter)
{	
	int i_pixel = threadIdx.x + blockIdx.x*blockDim.x;
	
	if (i_pixel >= Pixel_counter)
		return;

	d_cloud[i_pixel] = -1;
}
/* ------------------------------------------------ */

__global__ void find_maximum (int step, int N, float *vec, int *index, int *d_cloud, float *vec_out, int *index_out, int N_cloud, int *d_members)
/*
	One step in the iterative search for the brightest unassigned pixel, using a binary reduction method.
	   mask: input mask vector only needed for the step=0
*/
{
	__shared__ float svec[FIND_MAXIMUM_BS];
	__shared__ int sindex[FIND_MAXIMUM_BS];
	
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	
	if (i >= N)
		return;
	
	svec[threadIdx.x] = vec[i];
	
	if (step == 0)
		// Initializing index vector during the first step:
		sindex[threadIdx.x] = i;
		else
		sindex[threadIdx.x] = index[i];

	if (step==0 && d_cloud[i]!=-1)
		// For the assigned pixels, setting the vec value to a very low value
		// (This needs to be done only during the first step)
		svec[threadIdx.x] = -1e30;
	
	__syncthreads();
	
	// Binary reduction routine:
	int nTotalThreads = FIND_MAXIMUM_BS;
	while(nTotalThreads > 1)
	{
		int halfPoint = nTotalThreads / 2; // Number of active threads
		if (threadIdx.x < halfPoint)
		{
			int i2 = i + halfPoint;
			if (i2 < N) // Skipping the fictitious threads on the right side
			{
				int thread2 = threadIdx.x + halfPoint;
				float temp = svec[thread2];
				if (temp > svec[threadIdx.x])
				{
					// Memorizing both the larger svec value, and the corresponding index sindex
					svec[threadIdx.x] = temp;
					sindex[threadIdx.x] = sindex[thread2];
				}
			}
		}
		__syncthreads();
		nTotalThreads = halfPoint; // Reducing the binary tree size by two
	}
	
	if (threadIdx.x == 0)
	{
		if (gridDim.x == 1)
		// The final reduction kernel call
		{
			if (svec[0] > -1e29)
				// We found a new brightest pixel
			{
				// The index of the brightest pixel:
				int i_pixel = sindex[0];
				// Marking the brightest pixel as the first pixel in the new cloud with the index N_cloud:
				d_cloud[i_pixel] = N_cloud;
				// Adding the pixel to the start of the d_members vector:
				d_members[0] = i_pixel;
			}
			else
				// We ran out of pixels
			{
				d_members[0] = -1;
			}
		}
		else
		// For intermediate steps, we store the partial reductions results as segments in vectors vec_out and index_out
		{
			vec_out[blockIdx.x] = svec[0];
			index_out[blockIdx.x] = sindex[0];
		}
	}
	
}

/* ------------------------------------------------ */

__global__ void find_neighbours(int N_members, int N_cloud, unsigned int Pixel_counter, List d_list, int *d_cloud, int *d_members)
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

	int ix = d_list.ix[i];
	int iy = d_list.iy[i];
	int Jx = d_list.Jx[i];
	int Jy = d_list.Jy[i];	

	bool inCloud = false;
	int i0;
	
	// Loop over all freshely added members:
	for (int j=0; j<N_members; j++)
	{
		i0 = d_members[j];
		
		int dx  = abs(d_list.ix[i0] - ix);
		int dy  = abs(d_list.iy[i0] - iy);
		int dJx = abs(d_list.Jx[i0] - Jx);
		int dJy = abs(d_list.Jy[i0] - Jy);
		
		// Preliminary closeness test:
		if (dx<=1 && dy<=1 && dJx<=1 && dJy<=1)
		{
			// computing the closeness index
			int cl = dx + dy + dJx + dJy;
			// Cluster membership criterion:
			if (cl>0 && cl<=CL_MAX)
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

void cloud_stats (List h_list, unsigned int h_Pixel_counter, int N_cloud, int *Cluster_index,
 Cloud *cloud, float sgm, float MQ, int finetune, int rebin, int d_rebin, float p_min)
// Computing basic stats for the brightest clouds
{
	int NC;
	if (N_cloud < ICLOUD_STATS_MAX)
		NC = N_cloud;
	else
		NC = ICLOUD_STATS_MAX;
	
	FILE *fp;	
	
	if (finetune)
		fp = fopen("stats_fine.dat", "w");
	else
		fp = fopen("stats.dat", "w");

	for (int icloud=0; icloud<NC; icloud++)
	{
		float pmax = -1e30;
		int imax = -1;
		int ix_max = -1e7;
		int iy_max = -1e7;
		int Jx_max = -1e7;
		int Jy_max = -1e7;
		int ix_min = 1e7;
		int iy_min = 1e7;
		int Jx_min = 1e7;
		int Jy_min = 1e7;
		int N = 0;
		float mass = 0.0;
		for (int i=0; i<h_Pixel_counter; i++)
		{
			if (Cluster_index[i] == icloud)
			{
				N++;
				mass = mass + h_list.p[i];
				if (h_list.p[i] > pmax)
				{
					pmax = h_list.p[i];
					imax = i;
				}
				if (h_list.ix[i] > ix_max)
					ix_max = h_list.ix[i];
				if (h_list.iy[i] > iy_max)
					iy_max = h_list.iy[i];
				if (h_list.Jx[i] > Jx_max)
					Jx_max = h_list.Jx[i];
				if (h_list.Jy[i] > Jy_max)
					Jy_max = h_list.Jy[i];
				if (h_list.ix[i] < ix_min)
					ix_min = h_list.ix[i];
				if (h_list.iy[i] < iy_min)
					iy_min = h_list.iy[i];
				if (h_list.Jx[i] < Jx_min)
					Jx_min = h_list.Jx[i];
				if (h_list.Jy[i] < Jy_min)
					Jy_min = h_list.Jy[i];
			}
			
		}
		
		if (N == 0)
		{
			printf("Cluster %d has zero members!\n", icloud);
			continue;
		}
		
		// Jx, Jy are converted to pixels
		cloud[icloud].pmax = pmax;
		cloud[icloud].imax = imax;
		cloud[icloud].ix = h_list.ix[imax];
		cloud[icloud].iy = h_list.iy[imax];
		cloud[icloud].Jx = MQ * h_list.Jx[imax];
		cloud[icloud].Jy = MQ * h_list.Jy[imax];
		cloud[icloud].ix_min = ix_min;
		cloud[icloud].iy_min = iy_min;
		cloud[icloud].Jx_min = MQ * Jx_min;
		cloud[icloud].Jy_min = MQ * Jy_min;
		cloud[icloud].ix_max = ix_max;
		cloud[icloud].iy_max = iy_max;
		cloud[icloud].Jx_max = MQ * Jx_max;
		cloud[icloud].Jy_max = MQ * Jy_max;
		cloud[icloud].N = N;
		cloud[icloud].mass = mass;				
		
		// Switching to original (before rebinning) pixel size:
		float rad_xy = rebin*MQ*sqrt(pow((ix_max-ix_min)/2.0,2) + pow((ix_max-ix_min)/2.0,2));
		float rad_J = rebin*MQ*sqrt(pow((Jx_max-Jx_min)/2.0,2) + pow((Jx_max-Jx_min)/2.0,2));
		fprintf(fp, "%4d %11e %4d %4d %8.2f %8.2f %11e %7d %11e %8.2f %8.2f\n", 
		icloud, pmax/sgm,
		rebin*h_list.ix[imax]+d_rebin,
		rebin*h_list.iy[imax]+d_rebin,
		rebin*MQ*h_list.Jx[imax], rebin*MQ*h_list.Jy[imax],
		p_min,
		N, mass, rad_xy, rad_J);
	}
	
	fclose(fp);
	return;
}

/* ------------------------------------------------ */
	
	void save_cloud_fits (int Nx_ini, int Ny_ini, int Nx, int Ny, int Nc, float *img, const char *name,
		const char *name0, Cloud *cloud, int icloud, float sgm, int rebin, int d_rebin, double bias,
		int crop, int X00, int Y00)
		// Saving the stacked image as a fits file, doing upscaling (if rebin>1)
	{
		char buffer[100];
		int status=0; 
		
		float masked_value = bias - 4*sgm;
		
		// The file will be using the native (unbinned) resolution:
		long Npix_ini = Nx_ini*Ny_ini;		
		float *img1 = (float *)malloc(Npix_ini*sizeof(float));
				
	if (rebin > 1)
	{
		// Looping over the binned pixels:
		for (int ix=0; ix<Nx; ix++)
			for (int iy=0; iy<Ny; iy++)
				// Looping over the original resolution pixels
				{
					for (int jx=0; jx<rebin; jx++)
					{
						int ix_ini = ix*rebin + jx;
						for (int jy=0; jy<rebin; jy++)
						{
							int iy_ini = iy*rebin + jy;
							// Handling potential incomplete rows and columns at the end:
							if (ix_ini<Nx_ini && iy_ini<Ny_ini)
							{
								// Using one binned pixel to paint over the whole rebin x rebin area
								// of the native resolution image
								if (img[ix*Ny+iy] > MASK)
									img1[ix_ini*Ny_ini+iy_ini] = img[ix*Ny+iy] + bias;
								else
									img1[ix_ini*Ny_ini+iy_ini] = masked_value;  // Making masked areas darker
							}
						}
					}
				}
	}

	else
	{
		if (crop)
		{
			for (int i=0; i < Npix_ini; i++)
				img1[i] = masked_value;

			for (int ix=0; ix<Nx; ix++)
			{
				int ix_ini = ix + X00;
				for (int iy=0; iy<Ny; iy++)
				{
					int iy_ini = iy + Y00;
					
					if (ix_ini>=0 && ix_ini<Nx_ini && iy_ini>=0 && iy_ini<Ny_ini)
						if (img[ix*Ny+iy] > MASK)
							img1[ix_ini*Ny_ini+iy_ini] = img[ix*Ny+iy] + bias;
						else
							img1[ix_ini*Ny_ini+iy_ini] = masked_value;	
				}
			}
			
		}
		
		else
		{
			for (long i = 0; i < Npix_ini; i++)
			{
				if (img[i] > MASK)
					img1[i] = img[i] + bias;
				else
					img1[i] = masked_value;  // Making masked areas darker
			}
		}
	}
		
		sprintf(buffer, "rm -f %s >/dev/null", name);
		if (system(buffer))
			printf("Could not delete the file %s\n", name);
		fitsfile *fk, *f0;
		
		fits_open_file(&f0, name0, READONLY, &status);
		fits_error(status);
		
		fits_create_file(&fk, name, &status);
		fits_error(status);
		
		fits_copy_header(f0, fk, &status);
		fits_error(status);

		int Box_size = 50; // box size in pixels
		int BS2 = Box_size/2;
		sprintf(buffer, "%d;%d;%d;%d;-1.0;Cloud %d;", 
		rebin*cloud[icloud].iy+d_rebin-BS2, 
		rebin*cloud[icloud].ix+d_rebin-BS2,
		rebin*cloud[icloud].iy+d_rebin+BS2,
		rebin*cloud[icloud].ix+d_rebin+BS2,
		icloud);
		fits_write_key(fk, TSTRING, "ANNOTATE", buffer, NULL, &status);
		
		long nelem1  = (long)Nx_ini * Ny_ini * Nc;
		long fpixel = 1;
		fits_write_img(fk, TFLOAT, fpixel, nelem1, img1, &status);
		fits_error(status);
		fits_close_file(fk, &status);		
		fits_close_file(f0, &status);		
		
		free(img1);
		
		return;
	}
	
/* ------------------------------------------------ */

__global__ void erase_image (float *image, size_t pitch, int Nx, int Ny, double bias)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	
	if (x>=Nx || y>=Ny)
		return;

	float *row = (float *)((char*)image + x * pitch);
	row[y]  = bias;
	
	return;
}

/* ------------------------------------------------ */

void create_mosaic (int Nx, int Ny, List list, unsigned int Pixel_counter, int *Cluster_index, int NCmax, const char *name0, Cloud *cloud)
// Creating a single fits files mosaic.fit which displays the detection pixels corresponding to the top
// NCmax clouds, with corresponding ANNOTA boxes (for ASTAP)
{
	char buffer[100];
	const char *name = "mosaic.fit";
	int status=0; 
	float bias = 0.0;
		
	float *mosaic = (float *)malloc(3*Nx*Ny*sizeof(float));	
	
	int Box_size = 50; // box size in pixels
	int BS2 = Box_size/2;

	long Npix = Nx *  Ny;
	
	// Initializing the fits image to the bias level
	for (int i=0; i<3*Npix; i++)
		mosaic[i] = bias;
	
	
	// Saving the significant pixels from the top NCmax cloud to the image
	for (int i=0; i<Pixel_counter; i++)
	{
		if (Cluster_index[i] < NCmax)
		{
			int x = list.ix[i];
			int y = list.iy[i];
			int ic = Cluster_index[i];			
			float p = list.p[i]/cloud[ic].pmax;
			float r = ic / 3;
			float g = ic % 3;
			float b = (ic+2) % 2;
			float s = r + g + b;
			mosaic[x*Ny+y] = p*r/s + bias;
			mosaic[Npix + x*Ny+y] = p*g/s + bias;
			mosaic[2*Npix + x*Ny+y] = p*b/s + bias;
		}
	}
		
	sprintf(buffer, "rm -f %s >/dev/null", name);
	if (system(buffer))
		printf("Could not delete the file %s\n", name);
	fitsfile *fk, *f0;
	
	fits_open_file(&f0, name0, READONLY, &status);
	fits_error(status);
	
	fits_create_file(&fk, name, &status);
	fits_error(status);
	
//	fits_copy_header(f0, fk, &status);
//	fits_error(status);
		
	long naxes[3];
	int bitpix, naxis_in;
	fits_get_img_param(f0, 2, &bitpix, &naxis_in, naxes, &status);
	fits_error(status);
	naxes[2] = 3;
	fits_create_img(fk, bitpix, 3, naxes, &status);
	fits_error(status);
	int naxis_out = 3;
    fits_write_key(fk, TINT, "NAXIS", &naxis_out,
                    "number of array dimensions", &status);
	fits_error(status);
	
	int nkeys, i;
	char card[FLEN_CARD];
	fits_get_hdrspace(f0, &nkeys, NULL, &status);
	
	for (i = 1; i <= nkeys; i++) {
		fits_read_record(f0, i, card, &status);
		fits_error(status);

		/* Skip structural keywords */
		if (strncmp(card, "SIMPLE  ", 8) == 0) continue;
		if (strncmp(card, "BITPIX  ", 8) == 0) continue;
		if (strncmp(card, "NAXIS   ", 8) == 0) continue;
		if (strncmp(card, "NAXIS1  ", 8) == 0) continue;
		if (strncmp(card, "NAXIS2  ", 8) == 0) continue;
		if (strncmp(card, "NAXIS3  ", 8) == 0) continue;
		if (strncmp(card, "EXTEND  ", 8) == 0) continue;
		if (strncmp(card, "END     ", 8) == 0) continue;

		fits_write_record(fk, card, &status);
		fits_error(status);
	}	
	
	// Creating NCmax annotation boxes
	for (int icloud=0; icloud<NCmax; icloud++)
	{
		sprintf(buffer, "%d;%d;%d;%d;-1.0;%d;", cloud[icloud].iy-BS2, cloud[icloud].ix-BS2,
		cloud[icloud].iy+BS2,cloud[icloud].ix+BS2,icloud);
		fits_write_key(fk, TSTRING, "ANNOTATE", buffer, NULL, &status);	
	}	
	
	long nelem1  = (long)Nx * Ny * 3;
	long fpixel = 1;
	fits_write_img(fk, TFLOAT, fpixel, nelem1, mosaic, &status);
	fits_error(status);
	fits_close_file(fk, &status);		
	fits_close_file(f0, &status);		
	free(mosaic);

	return;
}
/* ------------------------------------------------ */


__global__ void compute_histogram (List d_list, unsigned int Pixel_counter, float p_min, float del_sgm, int *d_hist)
// Computing pixel brightness histogram for d_list pixels above p_min value.
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;

	if (i >= Pixel_counter)
		return;
	
	int bin = (d_list.p[i] - p_min) / del_sgm; // bin=0 contains p=p_min...p_min+del_sgm, etc
	
	if (bin < 0)
		bin = 0;
	else if (bin > NBIN-1)
		bin = NBIN - 1;
	
	atomicAdd(&d_hist[bin], 1);

	return;
}


/* ------------------------------------------------ */

void cropping(float *buf0, int Nx_ini, int Ny_ini, int Nx, int Ny, double bias, int X0, int Y0,
 float *image)
// Crop an image of size Nx x Ny from the image buf0 of size Nx_ini x Ny_ini, with
// X0,Y0 shift. Also subtract bias value. Save to image.
// Can only be used in finetune mode, and when no rebinning is used.
{	
	
	for (int ix=0; ix<Nx; ix++)
	{
		int ix_ini = ix + X0;
		for (int iy=0; iy<Ny; iy++)
		{
			int iy_ini = iy + Y0;
			
			if (ix_ini>=0 && ix_ini<Nx_ini && iy_ini>=0 && iy_ini<Ny_ini)
			{
				float p = buf0[ix_ini*Ny_ini+iy_ini];
				if (p > MASK )
					image[ix*Ny+iy] = p - bias;
				else
					image[ix*Ny+iy] = MASK0;
			}
			else
				// Areas outside of the original image are masked:
				image[ix*Ny+iy] = MASK0;
			
		}
	}
	
	return;
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


/*
__global__ void find_free_pixel (int *d_cloud, unsigned int Pixel_counter, int N_cloud, int *d_members)
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
		int old = atomicCAS(&i_free_pixel, -1, i_pixel);
		if (old == -1)
		{
			d_cloud[i_pixel] = N_cloud;
			d_members[0] = i_pixel;
		}
	}
}
*/
/* ------------------------------------------------ */
