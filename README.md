# ADIS: Asteroid Discovery in Imaging Sequences.

This package is aimed at amateur astrophotographers. Its purpose is discovery (serendipitous or
targeted) of faint/invisible moving star-like objects in imaging sequences taken with a telescope and an astro
camera. These could be asteroids, faint comets, potentially interstellar visitors. In most cases, the objects are too faint to show up in individual images.

The software is command line only (no GUI). For now, the only input images suppoted are RGB fits files (from OSC cameras). (Though it'd be a very easy fix to support monochromatic cameras as well.)

The package consists of two principal steps/programs:

1) **reconv**: removing all static objects in the imaging sequence (stars, galaxies, nebulae, sky glow
background etc). The majority of this is subtracted (so the data is preserved), with the rest (like residual star artifacts) masked out. At the end, the imaging sequence should look like plain noise, with some masked areas, and occasional visible moving objects - like artificial satellites
and meteors.
2) **asteroid_search**: the most compute intensive part. Brute search for moving objects in the imaging sequence,
by trying lots (thousands to millions) of different motion vectors, then stacking the whole image sequence along 
each vector, and looking for statistically significant brighter pixels. (This is sometimes called "synthetic tracking", though personally I don't like the term.) At the end, 4D clustering analysis
is performed, to find the moving objects ("clouds" in 4D space). The 4D space consists of the x,y pixel coordinates in the first image, and the motion vector pixel coordinates (Jx,Jy). The code is using GPU acceleration to 
make this practical. I am using CUDA, so **only NVIDIA GPUs are supported**.

I tested the code in two envoronments: WSL (Linux emulator under Windows), and Linux. Apparently MacOS no longer supports CUDA, so this will not work on MacOS unfortunately.

It requires libraries cfitsio and fftw3, gcc compiler, and CUDA environment (including nvcc compiler). Requires a fairly capable NVIDIA GPU, I'd say RTX 2080 or better, with at least 8GB of GPU RAM.

Installing pre-requisites (shown for Ubuntu, which is the default OS in WSL):

```
sudo apt install build-essential

sudo apt install libcfitsio-dev

sudo apt install libfftw3-dev
```

To install CUDA, follow this link: https://developer.nvidia.com/cuda-downloads . If you are using WSL (under Windows), choose "Linux" option, and then "WSL-Ubuntu" option. Next, choose version 2.0, next: choose "deb-local". (Here is the [direct link](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local).) Then simply follow the instructions.

To compile (assuming cfitsio and fftw3 were installed in standard locations; you need the development
versions of the libraries):

```
gcc -O2 reconv.c func.c -lfftw3 -lcfitsio -lm -lz -o reconv

nvcc -arch=sm_75 -O2 asteroid_search.cu func.c func2.cu -lfftw3 -lcfitsio -lm -lz -o asteroid_search
```

Adjust the -arch argument for your GPU card. For my RTX 2080 the CUDA capability is 7.5, so it is
-arch=sm_75. For the other card I tested, H100, it is -arch=sm_90. For other NVIDIA cards, check out [this link](https://developer.nvidia.com/cuda/gpus).

For more details, check the wiki page: https://github.com/pulsar123/Asteroid_detector/wiki .
