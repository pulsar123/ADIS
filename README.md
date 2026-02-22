Reconv: the utility to remove static objects (stars, galaxies etc) from astronomical image sequences, using
the stacked image as the master image. Uses FFT reconvolution.


To compile:

gcc -O3 -Wall reconv.c func.c -I /usr/include/cfitsio/ -lfftw3 -lcfitsio -lm -o ~/Macro-scripts/reconv

