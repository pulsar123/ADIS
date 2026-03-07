Reconv: the utility to remove static objects (stars, galaxies etc) from astronomical image sequences, using
the stacked image as the master image. Uses FFT reconvolution. Only RGB images are supported at the moment.

Requires libraries cfitsio and fftw3, and gcc compiler. Should be straightforward to compile and run under WSL (Windows), Linux, and Mac.

The image series needs to be calibrated, registered (on stars), and cropped (to get rid of edge artifacts if present). It is also a good idea to do plate solving for the whole series, as it will not be possible in the output destarred images. 

The master image is the above image sequence stacked. Enable "Normalize" option during stacking, to avoid pixel clipping.

To compile (adjust -I parameter accordingly):

gcc -O3 -Wall reconv.c func.c -I /usr/include/cfitsio/ -lfftw3 -lcfitsio -lm -o reconv

Now there is also a multi-threading (OpenMP) option available in the code. I achieve ~2x speedup (vs the serial option). For my 20-megapixels RGB FITS images, the multi-threading timing is 10 seconds. It is ~20 seconds for the serial version. For comparison, StarNet takes 67 seconds on my PC (and it's using the GPU).

To compile the code as muilti-threaded, add -fopenmp to the avove command:

gcc -fopenmp -O3 -Wall reconv.c func.c -I /usr/local/include -L /usr/local/lib -lfftw3 -lcfitsio -lm -lz -lcurl -o reconv

Also you may need to compile Cfitsio library yourself, using "./configure --enable-reentrant" command before running "make" (this is to make the image read fuunction thread-safe).

Options:

-i input_image: input FITS image: individual image from the series

-m master_image: input FITS image: stacked from the whole series

-o output_image: output FITS image with static objects (stars etc) removed. FITS header is copied from the individual input image

-R kernel_radius: radius of the kernel (in pixels) used during reconvolution. Adjust to minimize the star artifacts in the output image. Good initiall guess is ~1.6xFWHM of the stars in the master image.

Optional arguments:

-bias value: the value to use as a bias in the output image (on 0..1 scale). Zero by default

-blur sigma: experimental feature: Gaussian blur the master image with sigma pixels before processing. Disabled by default.

-bmax value: erase brightest pixels (which are above bias+value*std in the master image) in the output image. This is to minimize artifacts from the brightest stars. Disabled by default.

-k image_name: a debugging option. Writes out the kernel computed in the code as a fits file

-no_rescale: a debugging option. Sets the brightness scale=1 during master image subtraction

-v : verbose

The basic algorithm is as follows:

1) Read the individual image I and the master image M

2) Repeated for each color channel: R, G, B

  2a) Extract the channel from I->IC and M->MC
  
  2b) [Optional Gaussian blur of MC]
  
  2c) Iterative 3-sigma clipping to compute and subtract the bias for both IC and MC
  
  2d) Zero-pad IC and MC borders R pixels wide, compute direct FFT: IC->IF, MC->MF
  
  2e) Derive the kernel via complex division: KF=IF/MF, and then inverse FFT KF->K (this kernel is such that when we convolve MC with the kernel K, we get IC)
  
  2f) Erase to zero the kernel K beyond R pixels from the center. The transition is smooth, using a cubic spline with zero derivatives at both ends. Pixels up to R/2 away from the center are not modified. Outside R/2 the pixels are gradually driven to zero, with all pixels = 0 beyond R pixels.
  
  2g) Use FFT to convolve MC with the truncated kernel, the result is M1. Now the master image is blurred in exactly the same way as the individual image (on small scales).
  
  2h) Compute brightness scale S between M1 and IC (should be close to 1). This is an average brightness ratio in log space for all pixels which are brighter than Nsigma_cutoff=5 std units above zero. The averaging uses variable weighing, with zero weight for the darkest pixels, and maximum weight for the brightest ones (to minimize the impact of very noisy and numerous dark pixels).
  
  2i) The output image channel is computed as OC=IC-S*M1 .
  
  2j) The final optional step is to erase (to the bias value) all pixels in OC which are bmax std units above the bias in the convolved master image (M1). The erasure uses a cubic spline for a smooth transition. This is to deal with artifacts from brightest stars.
  
3) Copy FITS header from I to the output image O, and write O to the disk.

This procedure modifies the master image in such way that it becomes as close as possible to the individual image (in terms of blurness and brightness), and then subtracts the modified master image from the individual image. This ensures all faint and intertmediate brightness static objects are erased. Brightest stars leave some residual signal, which can be handled by the -bmax option (erasing brightest pixels). The only remaining signal is from transient objects or events - slowly moving asteroids and comets, satellites, cosmic rays etc.

The output imaging sequence can now be used for a fully automated discovery of moving objects in the series (using GPU acceleration to make this feaseable). I am working on such code now.

I provide a simple BASH script destar.sh which can be used to apply reconv to every image in the sequence.
